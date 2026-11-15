# Tổ chức project LifeMate: training và agent

**Trạng thái: cấu trúc chính đã triển khai.** Training được tách khỏi agent runtime, còn dữ liệu cuộc thi dùng một `DATA_ROOT` bên ngoài source code.

## Cấu trúc hiện tại

```text
zaic-2026/
├── main.py                         # Chạy inference/evaluation
├── src/lifemate/                   # Agent runtime và adapters inference
├── training/
│   ├── asr/README.md               # ASR training chưa có trainer
│   ├── data_prep/                  # Chuẩn bị, repair, distill, export
│   ├── llm/data/sft/               # Bộ dữ liệu SFT
│   ├── llm/data/distillation/      # Trajectory teacher/repair
│   ├── llm/scripts/                # Trainer và checkpoint evaluation
│   └── paths.py                    # DATA_ROOT và path dùng chung
├── scripts/research/               # Script nghiên cứu/phân tích theo mẫu
├── scripts/                        # Tiện ích/portal
├── docs/                           # Đặc tả và hướng dẫn
└── experiments/                    # Kết quả chạy thử
```

## Phân ranh trách nhiệm

- `src/lifemate/` và `main.py` dành cho inference, không phải huấn luyện.
- `training/llm/` có dataset, trainer LLM và nơi lưu checkpoint.
- `training/data_prep/` tạo và sửa dữ liệu trước khi train.
- `training/asr/` mới ghi nhận vị trí dự kiến. Repo chưa có ASR trainer; PhoWhisper chỉ được gọi qua adapter inference ở `src/lifemate/adapters/asr/`.
- Script phân tích một lần nằm trong `scripts/research/`; chúng không thuộc luồng chạy agent bắt buộc.

## Đường dẫn dữ liệu

`ZAIC_DATA_ROOT` là biến môi trường tùy chọn trỏ tới thư mục chứa `train/` và `public_test/`. `main.py` cũng nhận `--data-root`. Mặc định là project root.

```powershell
$env:ZAIC_DATA_ROOT = 'D:\path\to\train+public_test'
python main.py --split public_test --mode ollama --output submission.jsonl
```

Script huấn luyện dùng chung path module `training/paths.py`; không commit bộ dữ liệu cuộc thi.

## Việc còn lại trước lần train thật

1. Cài dependency GPU phù hợp (PyTorch/Unsloth/TRL) và chạy `python training/llm/scripts/train_sft.py --help`.
2. Xác nhận model nền, sequence length, batch size, GPU memory, output và checkpoint manifest trước khi chạy train.
3. Chạy một lượt nhỏ trên validation cố định, sau đó đánh giá trên cùng IDs khi so sánh bản chuẩn/harmonized.
4. Chỉ tạo ASR trainer nếu kết quả PhoWhisper inference cho thấy cần fine-tune; đo WER trước/sau trên tập holdout.
5. Trước khi đóng gói cuộc thi, kiểm tra định dạng submission và yêu cầu model/offline hiện hành.

## Lưu ý bảo mật

Một số script từng có khóa API viết trực tiếp trong mã; các giá trị dự phòng đó đã được gỡ. Dùng biến môi trường `OPENAI_API_KEY`. Nếu khóa cũ là khóa thật, hãy thu hồi và tạo khóa mới.
