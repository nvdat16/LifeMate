"""Reload the adapter, generate test predictions and export before/after reports."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from contextlib import nullcontext

import torch
from peft import PeftModel

from .artifacts import adapter_sha256
from .dataset import read_waveform
from .metrics import corpus_metrics, score_pair
from .modeling import load_base_model


def generate_text(current_model, relative_audio, args, data, context):
    inputs = context.processor.feature_extractor(
        read_waveform(data.audio_root, relative_audio), sampling_rate=16000,
        return_tensors='pt', return_attention_mask=True,
    )
    inputs = {key: value.to('cuda:0', dtype=context.compute_dtype if value.is_floating_point() else value.dtype)
              for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast('cuda', dtype=context.compute_dtype):
        generated = current_model.generate(
            **inputs, language='vi', task='transcribe', num_beams=1, do_sample=False,
            max_new_tokens=args.max_new_tokens, return_timestamps=False,
            return_dict_in_generate=True, use_cache=True,
        ).sequences
    if generated[0, -1].item() != context.processor.tokenizer.eos_token_id:
        raise RuntimeError('Chạm giới hạn token trước EOS, cần kiểm tra decoding.')
    return context.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def evaluate_test(label, model, args, data, context, adapter_dir, disable_adapter=False):
    rows = data.work_splits['test']
    output = data.run_dir / f'phowhisper-large-{label}.jsonl'
    config = {
        'base_model': args.base_model, 'revision': context.resolved_revision,
        'source_sha256': data.manifest['source_sha256'], 'split_fingerprint': data.split_fingerprint,
        'adapter_sha256': adapter_sha256(adapter_dir), 'label': label,
        'use_4bit': args.use_4bit, 'dtype': str(context.compute_dtype),
        'max_new_tokens': args.max_new_tokens, 'language': 'vi', 'num_beams': 1,
        'smoke_test': args.smoke_test,
    }
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    expected = {r['clip_id']: r for r in rows}
    completed = {}
    if output.exists():
        with output.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                result = json.loads(line)
                cid = result['clip_id']
                if result.get('config_id') != config_id or cid not in expected or cid in completed:
                    raise ValueError(f'File test cũ khác config hoặc ID không hợp lệ: {output}')
                if result.get('reference_text') != expected[cid]['audio_text'] or not isinstance(result.get('asr_text'), str):
                    raise ValueError(f'File test cũ có transcript không hợp lệ: {cid}')
                completed[cid] = result
    pending = [r for r in rows if r['clip_id'] not in completed]
    if pending:
        # A complete final JSON object without newline must not join with the next row.
        if output.exists() and output.stat().st_size:
            with output.open('rb') as stream:
                stream.seek(-1, 2)
                needs_newline = stream.read(1) != b'\n'
            if needs_newline:
                with output.open('a', encoding='utf-8') as stream:
                    stream.write('\n')
        with model.disable_adapter() if disable_adapter else nullcontext():
            generate_text(model, pending[0]['audio'], args, data, context)
            torch.cuda.synchronize()
            with output.open('a', encoding='utf-8') as stream:
                for index, row in enumerate(pending, 1):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    text = generate_text(model, row['audio'], args, data, context)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-start
                    result = {
                        'clip_id': row['clip_id'], 'id': row['id'], 'kind': row['kind'],
                        'audio': row['audio'], 'reference_text': row['audio_text'], 'asr_text': text,
                        'audio_seconds': row['audio_seconds'], 'inference_seconds': seconds,
                        'model': args.base_model + (' + LoRA' if not disable_adapter else ' (base only)'),
                        'config_id': config_id, 'config': config,
                        **score_pair(row['audio_text'], text),
                    }
                    stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                    completed[row['clip_id']] = result
                    if index == 1 or index % 25 == 0 or index == len(pending):
                        print(label, index, '/', len(pending), flush=True)
    return [completed[r['clip_id']] for r in rows]


def write_comparison(args, data, context, before, after):
    assert [r['clip_id'] for r in before] == [r['clip_id'] for r in after]
    comparison = []
    for label, results in [('before', before), ('after', after)]:
        for kind in ('overall', 'current', 'history'):
            subset = [r for r in results if kind == 'overall' or r['kind'] == kind]
            if subset:
                metrics = corpus_metrics([r['reference_text'] for r in subset], [r['asr_text'] for r in subset])
                comparison.append({'model': label, 'group': kind, **metrics})
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    with (data.run_dir / 'before_after.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    (data.run_dir / 'before_after.json').write_text(json.dumps({
        'evaluation_status': data.manifest['evaluation_status'], 'smoke_test': args.smoke_test,
        'baseline_precision': 'NF4 4-bit' if args.use_4bit else str(context.compute_dtype),
        'results': comparison,
    }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    differences = [{'clip_id': a['clip_id'], 'reference': a['reference_text'], 'before': a['asr_text'], 'after': b['asr_text'],
                    'wer_before': a['wer'], 'wer_after': b['wer'], 'delta_wer': b['wer']-a['wer']}
                   for a, b in zip(before, after)]
    with (data.run_dir / 'per_clip_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(differences[0]))
        writer.writeheader(); writer.writerows(differences)


def run_evaluation(args, data, context, adapter_dir):
    base_model = load_base_model(args, context)
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    # Preview validation only, after the adapter has been selected.
    preview = data.work_splits['validation'][0]
    print('Chuẩn:', preview['audio_text'])
    print('LoRA: ', generate_text(model, preview['audio'], args, data, context))
    before = evaluate_test('before', model, args, data, context, adapter_dir, disable_adapter=True)
    after = evaluate_test('after', model, args, data, context, adapter_dir)
    write_comparison(args, data, context, before, after)
