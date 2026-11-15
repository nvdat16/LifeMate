"""Read competition transcripts/WAVs and prepare split manifests without GPU dependencies."""
from __future__ import annotations

import csv
import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path

from .artifacts import file_sha256, restore_run
from .config import PROJECT_ROOT
from .metrics import metric_text
from .splitting import audit_splits, balanced_group_split, group_and_deduplicate, split_statistics


@dataclass(frozen=True)
class PreparedData:
    run_dir: Path
    audio_root: Path
    manifest: dict
    split_fingerprint: str
    work_splits: dict[str, list[dict]]


def find_train(root=None):
    root = Path(root).expanduser() if root else (Path('/kaggle/input') if Path('/kaggle/input').is_dir() else PROJECT_ROOT)
    if root.is_file():
        return root.resolve()
    candidates = []
    for path in sorted(root.rglob('train.jsonl')):
        # Prepared clip manifests are outputs, not conversation-level sources.
        if (path.parent / 'split_manifest.json').exists():
            continue
        with path.open(encoding='utf-8') as stream:
            line = next((line for line in stream if line.strip()), None)
        if line:
            first = json.loads(line)
            if 'audio' in first and 'audio_text' in first:
                candidates.append(path.resolve())
    if len(candidates) != 1:
        raise ValueError(f'Hãy truyền --data-root cụ thể; tìm được: {candidates}')
    return candidates[0]


def load_speaker_map(path=None):
    mapping = {}
    if path is None:
        return mapping
    path = Path(path).expanduser()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream)) if path.suffix.lower() == '.csv' else [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        speaker = row.get('speaker_id')
        if speaker is None or not str(speaker).strip():
            raise ValueError('speaker_id thiếu trong speaker map.')
        if not row.get('clip_id') and not row.get('audio'):
            raise ValueError('Speaker map cần clip_id hoặc audio cho từng dòng.')
        for field in ('clip_id', 'audio'):
            key = row.get(field)
            if key:
                namespaced = (field, str(key))
                if namespaced in mapping and mapping[namespaced] != str(speaker):
                    raise ValueError(f'Speaker map xung đột: {namespaced}')
                mapping[namespaced] = str(speaker)
    return mapping


def read_clips(source, audio_root, speaker_map=None):
    speaker_map = speaker_map or {}
    clips, sample_ids = [], set()
    audio_root = Path(audio_root).resolve()
    with Path(source).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            sid = item['id']
            if sid in sample_ids:
                raise ValueError(f'ID mẫu trùng: {sid}')
            sample_ids.add(sid)
            turns = [(f'{sid}:current', 'current', None, item)]
            turns += [(f'{sid}:history:{i}', 'history', i, turn)
                      for i, turn in enumerate(item.get('history', [])) if turn.get('role') == 'user']
            for cid, kind, history_index, turn in turns:
                text = turn.get('audio_text')
                if not isinstance(text, str) or not metric_text(text):
                    raise ValueError(f'{cid}: thiếu transcript chuẩn.')
                path = (audio_root / turn['audio']).resolve()
                if not path.is_relative_to(audio_root):
                    raise ValueError(f'{cid}: audio nằm ngoài audio root.')
                with wave.open(str(path), 'rb') as wav:
                    if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, 'NONE'):
                        raise ValueError(f'{path}: cần WAV PCM mono 16 kHz/16-bit.')
                    frames = wav.getnframes()
                    duration = frames / 16000
                    if not 0 < duration <= 30:
                        raise ValueError(f'{path}: duration={duration}; không tự cắt audio > 30 giây.')
                    pcm = wav.readframes(frames)
                    if len(pcm) != frames*2:
                        raise ValueError(f'{path}: WAV thiếu dữ liệu.')
                candidates = [value for value in (
                    turn.get('speaker_id'), speaker_map.get(('clip_id', cid)), speaker_map.get(('audio', turn['audio']))
                ) if value is not None and str(value).strip()]
                speakers = {str(value) for value in candidates}
                if len(speakers) > 1:
                    raise ValueError(f'{cid}: speaker metadata xung đột.')
                clips.append({
                    'clip_id': cid, 'id': sid, 'kind': kind, 'history_index': history_index,
                    'audio': turn['audio'], 'audio_text': text,
                    'audio_seconds': duration, 'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                    'speaker_id': next(iter(speakers)) if speakers else None,
                })
    if not clips:
        raise ValueError('Không có audio.')
    return clips


def prepare_data(args):
    """Validate audio and persist the reproducible, leakage-audited split."""
    ratios = dict(zip(('train', 'validation', 'test'), args.split_ratios))
    run_dir = args.output_root.expanduser().resolve() / (args.run_name + ('-smoke' if args.smoke_test else ''))
    source = find_train(args.data_root)
    audio_root = Path(args.audio_root).expanduser().resolve() if args.audio_root else source.parent
    all_clips = read_clips(source, audio_root, load_speaker_map(args.speaker_map))
    groups, duplicates, _ = group_and_deduplicate(all_clips, args.deduplicate_audio)
    splits = balanced_group_split(groups, ratios, args.seed, args.split_restarts)
    audit_splits(splits)
    stats = split_statistics(splits)
    print('Source:', source)
    print('Audio:', len(all_clips), '| PCM unique:', len({r['pcm_sha256'] for r in all_clips}))
    print('Duplicate groups:', len(duplicates))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if all(r['speaker_id'] is None for r in all_clips):
        print('Không có speaker_id: đây là split theo hội thoại/PCM, chưa chứng minh speaker-disjoint.')
    current_ratio = sum(r['kind']=='current' for rows in splits.values() for r in rows)/sum(len(rows) for rows in splits.values())
    for stat in stats:
        if abs(stat['current_ratio']-current_ratio) > args.kind_ratio_tolerance:
            print('Tỷ lệ current/history lệch mục tiêu, ưu tiên giữ nhóm:', stat['split'])

    manifest = {
        'format': 'phowhisper_lora_group_split_v1', 'seed': args.seed, 'ratios': ratios,
        'source_sha256': file_sha256(source), 'deduplicate_pcm': args.deduplicate_audio,
        'grouping': ['conversation id', 'known speaker_id', 'exact PCM SHA-256'],
        'evaluation_status': 'internal test previously inspected through train benchmarks',
        'duplicates': duplicates, 'statistics': stats, 'splits': splits,
    }
    split_fingerprint = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if args.restore_run_dir is not None:
        restore_run(args.restore_run_dir, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / 'split_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous != manifest:
            raise ValueError('Run cũ dùng split khác. Đổi --run-name để tạo run mới.')
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name, rows in splits.items():
        with (run_dir / f'{name}.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    smoke_limits = {'train': 32, 'validation': 8, 'test': 8}
    work_splits = {name: rows[:smoke_limits[name]] for name, rows in splits.items()} if args.smoke_test else splits
    print('Run:', run_dir, '| Smoke test:', args.smoke_test)
    return PreparedData(run_dir, audio_root, manifest, split_fingerprint, work_splits)
