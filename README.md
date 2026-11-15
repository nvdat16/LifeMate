# LifeMate — Zalo AI Challenge 2026

Agent tiếng Việt nhận yêu cầu bằng giọng nói, gọi công cụ trong môi trường cuộc thi và xuất kết quả theo định dạng yêu cầu.

## Bắt đầu

- [Bản đồ project](docs/project_map.md): chức năng từng thư mục.
- [Kế hoạch tổ chức](docs/project_reorganization_plan.md): cấu trúc đích và trạng thái triển khai.
- [Hướng dẫn training](training/README.md): chuẩn bị dữ liệu, train LLM và fine-tune ASR bằng script.
- [Đặc tả LifeMate](docs/lifemate_specs.md): schema, công cụ và chấm điểm.

## Cấu trúc chính

| Thư mục | Chức năng |
|---|---|
| `src/lifemate/` | Agent runtime: workflow, adapters, sandbox, ASR/LLM inference, evaluation và UI |
| `training/` | Dữ liệu LLM, script chuẩn bị dữ liệu, trainer và nơi lưu checkpoint |
| `scripts/research/` | Script phân tích theo từng lỗi/mẫu; không phải entry point của agent |
| `scripts/` | Lệnh tiện ích, tích hợp portal và khởi chạy |
| `docs/` | Đặc tả, hướng dẫn và báo cáo |
| `experiments/` | Kết quả chạy thử đã lưu |

## Chạy agent

```powershell
python main.py --split public_test --mode ollama --model qwen2.5:1.5b --output submission.jsonl
```

Nếu bộ dữ liệu cuộc thi không nằm ngay trong thư mục project, đặt `ZAIC_DATA_ROOT` trỏ đến thư mục chứa `train/` và `public_test/`, hoặc truyền `--data-root`:

```powershell
$env:ZAIC_DATA_ROOT = 'D:\path\to\train+public_test'
python main.py --split public_test --mode ollama --model qwen2.5:1.5b --output submission.jsonl
```

Dùng `python main.py --help` để xem các tùy chọn. Không đưa bộ dữ liệu cuộc thi hoặc khóa API vào Git.

### Dùng OpenAI API

Đặt `OPENAI_API_KEY` trong môi trường chạy (trên Kaggle, lưu bằng Kaggle Secrets rồi đưa secret vào biến môi trường của phiên Notebook). Sau đó:

```bash
python main.py --split public_test --mode openai --model "REPLACE_WITH_MODEL_ID" --output submission.jsonl
```

Endpoint mặc định là `https://api.openai.com/v1`; dùng `--base-url` nếu gọi một endpoint tương thích khác. Chế độ này cần Internet và phát sinh chi phí API theo tài khoản/model. Không ghi khóa vào file notebook, mã nguồn hoặc dòng lệnh.
