"""Artifact fingerprints, restoring Trainer runs and ZIP packaging."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_sha256(directory):
    parts = [(path.name, file_sha256(path)) for path in sorted(Path(directory).glob('adapter*')) if path.is_file()]
    if not parts:
        raise ValueError('Chưa có adapter đã lưu.')
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def restore_run(source, destination):
    """Copy a Trainer run and relocate its recorded best-checkpoint paths."""
    source, destination = source.expanduser().resolve(), destination.resolve()
    if not source.is_dir():
        raise ValueError(f'Không tìm thấy run cần khôi phục: {source}')
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('--restore-run-dir cần thư mục đích mới nằm ngoài run nguồn. '
                         'Nếu đã copy, bỏ tùy chọn này để resume.')
    for name in ('split_manifest.json', 'training_config.json'):
        if not (source / name).is_file():
            raise ValueError(f'Run cần khôi phục thiếu {name}.')
    shutil.copytree(source, destination)
    for state_path in (destination / 'checkpoints').rglob('trainer_state.json'):
        state = json.loads(state_path.read_text(encoding='utf-8'))
        best = state.get('best_model_checkpoint')
        if best:
            # Trainer stores absolute paths, which change across machines/sessions.
            name = best.replace('\\', '/').rstrip('/').split('/')[-1]
            candidate = destination / 'checkpoints' / name
            if not name.startswith('checkpoint-') or not candidate.is_dir():
                raise ValueError(f'Run khôi phục thiếu best checkpoint: {best}')
            state['best_model_checkpoint'] = str(candidate)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def package_run(run_dir, zip_full_run=False):
    package = run_dir.parent / f'{run_dir.name}-adapter-results.zip'
    with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(run_dir.rglob('*')):
            if path.is_file() and 'checkpoints' not in path.relative_to(run_dir).parts:
                bundle.write(path, arcname=str(path.relative_to(run_dir)))
    print('Adapter:', run_dir / 'adapter')
    print('Báo cáo:', run_dir)
    print('ZIP:', package)

    if zip_full_run:
        full_package = shutil.make_archive(str(run_dir.parent / (run_dir.name+'-resume')), 'zip', root_dir=run_dir)
        print('ZIP resume:', full_package)
