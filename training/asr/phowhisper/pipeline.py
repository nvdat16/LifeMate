"""Coordinate data preparation, training, evaluation and packaging."""
from __future__ import annotations

import gc
import os

from .artifacts import package_run
from .config import parse_args
from .data import prepare_data


def main(argv=None):
    args = parse_args(argv)
    data = prepare_data(args)
    if args.prepare_only:
        print('Đã chuẩn bị dữ liệu. Chạy lại cùng lệnh, bỏ --prepare-only để train.')
        return

    # Set visibility before importing any GPU module; preserve scheduler allocation.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('Script này dùng một GPU. Chạy bằng python, không dùng torchrun.')

    import torch
    from .trainer import train
    from .evaluation import run_evaluation

    context, adapter_dir = train(args, data)
    # Trainer/model references stay within train(), allowing a fresh adapter reload.
    gc.collect()
    torch.cuda.empty_cache()
    run_evaluation(args, data, context, adapter_dir)
    package_run(data.run_dir, zip_full_run=args.zip_full_run)
