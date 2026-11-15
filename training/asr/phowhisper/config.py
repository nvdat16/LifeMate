"""Command-line options and repository paths."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASR_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Fine-tune PhoWhisper-large LoRA: chia dữ liệu, train, đánh giá và đóng gói.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-root', default=os.environ.get('ZAIC_DATA_ROOT'),
                        help='Thư mục chứa train/train.jsonl, thư mục train, hoặc file JSONL. '
                             'Mặc định: ZAIC_DATA_ROOT, /kaggle/input, rồi project root.')
    parser.add_argument('--audio-root', type=Path, help='Gốc để nối đường dẫn audio trong JSONL; mặc định là thư mục chứa JSONL.')
    parser.add_argument('--speaker-map', type=Path, help='CSV/JSONL có speaker_id và clip_id hoặc audio.')
    parser.add_argument('--output-root', type=Path,
                        default=Path('/kaggle/working/phowhisper_lora') if Path('/kaggle/working').is_dir() else ASR_ROOT / 'outputs',
                        help='Thư mục chứa các run.')
    parser.add_argument('--run-name', default='phowhisper-large-lora-v1', help='Tên thư mục run; đổi tên khi đổi cấu hình.')
    parser.add_argument('--restore-run-dir', type=Path, help='Copy cả run cũ sang output trước khi resume; đích phải chưa tồn tại.')
    parser.add_argument('--resume-from-checkpoint', type=Path, help='Checkpoint thuộc run hiện tại; mặc định tự tìm checkpoint cuối.')
    parser.add_argument('--base-model', default='vinai/PhoWhisper-large', help='Model Hugging Face hoặc đường dẫn local.')
    parser.add_argument('--base-revision', default='main', help='Revision cho run mới; run cũ dùng commit đã lưu.')
    parser.add_argument('--gpu', default='0', help='Một GPU CUDA; giữ CUDA_VISIBLE_DEVICES nếu đã được đặt.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-ratios', type=float, nargs=3, default=(0.8, 0.1, 0.1),
                        metavar=('TRAIN', 'VALIDATION', 'TEST'))
    parser.add_argument('--split-restarts', type=int, default=24)
    parser.add_argument('--deduplicate-audio', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--kind-ratio-tolerance', type=float, default=0.05)
    parser.add_argument('--prepare-only', action='store_true', help='Kiểm tra WAV và lưu split; không cần torch, CUDA hay Internet.')
    parser.add_argument('--smoke-test', action='store_true', help='3 optimizer steps, tối đa 32/8/8 clip; thêm hậu tố -smoke vào tên run.')
    parser.add_argument('--use-4bit', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=440)
    parser.add_argument('--zip-full-run', action='store_true', help='Đóng gói thêm checkpoints/ và optimizer để resume.')
    args = parser.parse_args(argv)
    if not args.run_name or args.run_name in {'.', '..'} or '/' in args.run_name or '\\' in args.run_name:
        parser.error('--run-name phải là một tên thư mục, không phải đường dẫn.')
    if not 0 < args.max_new_tokens <= 440:
        parser.error('--max-new-tokens cần trong 1..440.')
    for name in ('split_restarts', 'lora_r', 'lora_alpha', 'batch_size', 'gradient_accumulation_steps', 'learning_rate', 'epochs'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'--{name.replace("_", "-")} phải lớn hơn 0 và hữu hạn.')
    if not 0 <= args.lora_dropout < 1 or not 0 <= args.kind_ratio_tolerance <= 1:
        parser.error('--lora-dropout cần trong [0, 1); --kind-ratio-tolerance cần trong [0, 1].')
    if any(not math.isfinite(r) or r <= 0 for r in args.split_ratios) or not math.isclose(sum(args.split_ratios), 1.0):
        parser.error('--split-ratios phải gồm ba số > 0, tổng bằng 1.')
    if ',' in args.gpu or not args.gpu.strip():
        parser.error('--gpu phải chọn đúng một GPU.')
    return args
