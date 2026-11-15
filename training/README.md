# Training

Khu vực huấn luyện tách khỏi agent runtime. Runtime và inference vẫn ở `src/lifemate/`; lệnh chính của agent là `main.py`.

## Thành phần

- `data_prep/`: tạo và chuẩn hóa SFT, sinh hoặc sửa trajectory teacher.
- `llm/data/sft/`: dữ liệu SFT ChatML/ShareGPT.
- `llm/data/distillation/`: trajectory và bản repair/trung gian.
- `llm/scripts/train_sft.py`: fine-tune LLM bằng Unsloth.
- `llm/scripts/evaluate_checkpoint.py`: đánh giá checkpoint.
- `llm/outputs/`: nơi lưu checkpoint train; không commit.
- `asr/scripts/main.py`: fine-tune PhoWhisper-large bằng LoRA, chia dữ liệu, đánh giá WER/CER và xuất adapter.
- `asr/outputs/`: checkpoint và báo cáo ASR; không commit. ASR inference nằm trong `src/lifemate/adapters/asr/`.

## Dữ liệu cuộc thi

Đặt `ZAIC_DATA_ROOT` tới thư mục chứa `train/` và `public_test/` trước khi chạy script cần đọc dữ liệu nguồn. Nếu bỏ trống, project root được dùng làm mặc định.

```powershell
$env:ZAIC_DATA_ROOT = 'D:\path\to\train+public_test'
```

## Fine-tune LLM

```powershell
python training/llm/scripts/train_sft.py --help
```

Trainer mặc định dùng `llm/data/sft/train_chatml.jsonl` và `val_chatml.jsonl`, lưu output dưới `training/llm/outputs/`. Kiểm tra help trước khi chạy GPU để xem model, batch size và các tham số hiện tại.

Script chuẩn bị dữ liệu được liệt kê trong `data_prep/`; xem README của thư mục để biết mục đích. Các script gọi API cần biến môi trường `OPENAI_API_KEY`.

## Fine-tune ASR

Chạy toàn bộ pipeline từ script thay vì từng cell notebook:

```bash
python -m pip install -r training/asr/requirements.txt
python training/asr/scripts/main.py --data-root "train+public_test" --smoke-test
python training/asr/scripts/main.py --data-root "train+public_test"
```

Train cần một GPU NVIDIA/CUDA. Dùng `--prepare-only` để kiểm tra audio/chia dữ liệu trên CPU, `--help` để xem tham số. Xem [ASR README](asr/README.md) cho cấu hình, resume và output.
