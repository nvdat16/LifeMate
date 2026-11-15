#!/usr/bin/env python3
"""Merge a trained PhoWhisper LoRA run, convert it for faster-whisper and upload to Hugging Face.

Only model weights, configs and aggregate metrics are uploaded. Split manifests, transcripts,
predictions and per-clip CSVs in the run directory are never copied to the export folders.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Files allowed into an upload. Everything else is refused, so transcripts cannot leak.
UPLOAD_PATTERNS = ['*.json', '*.safetensors', '*.bin', '*.txt', '*.model', 'README.md']
PROCESSOR_FILES = ['tokenizer.json', 'preprocessor_config.json']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Gộp LoRA → (tùy chọn) CTranslate2 → upload Hugging Face (mặc định private).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--run-dir', type=Path, required=True, help='Thư mục run, ví dụ training/asr/outputs/phowhisper-medium-lora-v1.')
    parser.add_argument('--repo-id', required=True,
                        help='Tên repo gốc, ví dụ PhoWhisper-medium-zaic hoặc user/PhoWhisper-medium-zaic. '
                             'Tạo <repo>-lora và <repo>-ct2-<quantization>.')
    parser.add_argument('--quantization', default='int8_float16',
                        choices=['int8', 'int8_float16', 'int8_bfloat16', 'float16', 'bfloat16', 'float32'],
                        help='Quantization CTranslate2 cho faster-whisper.')
    parser.add_argument('--skip-ct2', action='store_true', help='Chỉ upload adapter LoRA, không gộp/chuyển CT2.')
    parser.add_argument('--upload-merged', action='store_true', help='Upload thêm model transformers đã gộp (<repo>-merged, ~1.5GB).')
    parser.add_argument('--public', action='store_true', help='Tạo repo public. Mặc định private vì model train trên dữ liệu cuộc thi.')
    parser.add_argument('--no-upload', action='store_true', help='Chỉ gộp/chuyển đổi local, không upload.')
    parser.add_argument('--force', action='store_true', help='Ghi đè thư mục export cũ.')
    return parser.parse_args(argv)


def load_run(run_dir):
    run_dir = run_dir.expanduser().resolve()
    config_path, adapter_dir = run_dir / 'training_config.json', run_dir / 'adapter'
    if not config_path.is_file() or not (adapter_dir / 'adapter_config.json').is_file():
        raise SystemExit(f'{run_dir} không phải run đã train xong (thiếu training_config.json hoặc adapter/).')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('smoke_test'):
        raise SystemExit('Đây là run smoke test; không upload.')
    result_path = run_dir / 'training_result.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.is_file() else {}
    metrics_path = run_dir / 'before_after.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))['results'] if metrics_path.is_file() else []
    return run_dir, adapter_dir, config, result, metrics


def model_card(kind, repo_id, config, result, metrics, quantization=None):
    rows = '\n'.join(f"| {m['model']} | {m['group']} | {m['wer']:.4f} | {m['cer']:.4f} | {m['clips']} |" for m in metrics)
    table = ('| Model | Nhóm | WER | CER | Clips |\n|---|---|---|---|---|\n' + rows) if rows else '_Chưa có before_after.json._'
    usage = {
        'lora': f"""```python
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

base = AutoModelForSpeechSeq2Seq.from_pretrained("{config['base_model']}", revision="{config['resolved_base_revision']}")
model = PeftModel.from_pretrained(base, "{repo_id}")
processor = AutoProcessor.from_pretrained("{repo_id}")
```""",
        'ct2': f"""```python
from faster_whisper import WhisperModel

model = WhisperModel("{repo_id}", device="cuda", compute_type="{quantization}")
segments, _ = model.transcribe("audio.wav", language="vi", beam_size=1)
print(" ".join(s.text for s in segments))
```""",
        'merged': f"""```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="{repo_id}")
print(asr("audio.wav", generate_kwargs={{"language": "vi", "task": "transcribe"}})["text"])
```""",
    }[kind]
    title = {'lora': 'LoRA adapter', 'ct2': f'CTranslate2 ({quantization}) cho faster-whisper', 'merged': 'Model đã gộp LoRA'}[kind]
    best_wer = result.get('best_validation_wer')
    return f"""---
language: vi
base_model: {config['base_model']}
library_name: {'peft' if kind == 'lora' else ('ctranslate2' if kind == 'ct2' else 'transformers')}
pipeline_tag: automatic-speech-recognition
tags: [whisper, phowhisper, vietnamese, lora]
---

# {repo_id.split('/')[-1]} — {title}

Fine-tune LoRA của `{config['base_model']}` (commit `{config['resolved_base_revision']}`) cho LifeMate.

## Cấu hình train

- LoRA r={config['lora_r']}, alpha={config['lora_alpha']}, dropout={config['lora_dropout']}, target {', '.join(config['target_modules'])} (encoder + decoder)
- Batch {config['batch_size']} × grad accum {config['gradient_accumulation']}, lr {config['learning_rate']}, {config['epochs']} epoch, seed {config['seed']}
- Precision: {config['compute_dtype']}{' + NF4 4-bit' if config['use_4bit'] else ''}
- Best validation WER: {f'{best_wer:.4f}' if best_wer is not None else 'n/a'}

## Đánh giá trên test nội bộ (greedy decoding)

{table}

WER/CER chuẩn hóa NFC, chữ thường, bỏ dấu câu; giữ dấu tiếng Việt và chữ số. Test tách từ tập train nên là đánh giá nội bộ.

## Sử dụng

{usage}
"""


def merge(adapter_dir, config, output_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    print('Gộp LoRA vào', config['base_model'], '@', config['resolved_base_revision'])
    base = AutoModelForSpeechSeq2Seq.from_pretrained(
        config['base_model'], revision=config['resolved_base_revision'], dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    generation = model.generation_config
    generation.forced_decoder_ids = None
    generation.language, generation.task = 'vi', 'transcribe'
    model.save_pretrained(output_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(adapter_dir).save_pretrained(output_dir)
    del model, base


def convert_ct2(merged_dir, output_dir, quantization):
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        raise SystemExit('Thiếu ctranslate2: pip install ctranslate2 (hoặc chạy với --skip-ct2).')
    print('Chuyển CTranslate2', quantization, '->', output_dir)
    copy_files = [name for name in PROCESSOR_FILES if (merged_dir / name).is_file()]
    TransformersConverter(str(merged_dir), copy_files=copy_files).convert(str(output_dir), quantization=quantization, force=True)


def upload(api, folder, repo_id, private):
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type='model')
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type='model',
                      allow_patterns=UPLOAD_PATTERNS, commit_message='Upload PhoWhisper LoRA export')
    print('Đã upload:', f'https://huggingface.co/{repo_id}', '(private)' if private else '(public)')


def main(argv=None):
    args = parse_args(argv)
    run_dir, adapter_dir, config, result, metrics = load_run(args.run_dir)

    api = None
    repo_base = args.repo_id
    if not args.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            user = api.whoami()['name']
        except Exception:
            raise SystemExit('Chưa đăng nhập Hugging Face. Chạy: hf auth login (token có quyền write) hoặc đặt HF_TOKEN.')
        if '/' not in repo_base:
            repo_base = f'{user}/{repo_base}'

    export_root = run_dir / 'export'
    if export_root.exists():
        if not args.force:
            raise SystemExit(f'{export_root} đã tồn tại. Thêm --force để làm lại.')
        shutil.rmtree(export_root)
    export_root.mkdir()

    # Adapter: copy only the adapter/processor files written by save_pretrained.
    lora_dir = export_root / 'lora'
    shutil.copytree(adapter_dir, lora_dir)
    lora_repo = f'{repo_base}-lora'
    (lora_dir / 'README.md').write_text(model_card('lora', lora_repo, config, result, metrics), encoding='utf-8')
    targets = [(lora_dir, lora_repo)]

    if not args.skip_ct2 or args.upload_merged:
        merged_dir = export_root / 'merged'
        merge(adapter_dir, config, merged_dir)
        if args.upload_merged:
            merged_repo = f'{repo_base}-merged'
            (merged_dir / 'README.md').write_text(model_card('merged', merged_repo, config, result, metrics), encoding='utf-8')
            targets.append((merged_dir, merged_repo))
        if not args.skip_ct2:
            ct2_dir = export_root / f'ct2-{args.quantization}'
            convert_ct2(merged_dir, ct2_dir, args.quantization)
            ct2_repo = f"{repo_base}-ct2-{args.quantization.replace('_', '-')}"
            (ct2_dir / 'README.md').write_text(model_card('ct2', ct2_repo, config, result, metrics, args.quantization), encoding='utf-8')
            targets.append((ct2_dir, ct2_repo))

    for folder, repo_id in targets:
        print('Export:', folder, '->', repo_id)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                print('   ', path.relative_to(folder), f'{path.stat().st_size/1e6:.1f} MB')
    if args.no_upload:
        print('Bỏ qua upload (--no-upload).')
        return
    for folder, repo_id in targets:
        upload(api, folder, repo_id, private=not args.public)


if __name__ == '__main__':
    main()
