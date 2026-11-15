# Bản đồ thư mục LifeMate

Trang này chỉ ra nơi cần mở theo từng loại công việc.

## Cấu trúc nhanh

```text
zaic-2026/
├── main.py                         # Entry point chạy agent/evaluation
├── src/lifemate/                   # Mã runtime của agent
├── training/
│   ├── asr/                        # Trainer PhoWhisper LoRA, báo cáo và checkpoint ASR
│   ├── data_prep/                  # Chuẩn bị, distill, repair và export dữ liệu
│   └── llm/
│       ├── data/                   # SFT và dữ liệu distillation
│       ├── scripts/                # Train/evaluate LLM
│       └── outputs/                # Checkpoint đầu ra (không commit)
├── scripts/research/               # Script phân tích một lần
├── scripts/                        # Tiện ích dự án/portal
├── docs/                           # Đặc tả và hướng dẫn
├── experiments/                    # Kết quả chạy thử
└── tests/                          # Bộ kiểm tra dự án
```

## Chức năng từng phần

| Đường dẫn | Chứa gì | Khi nào mở |
|---|---|---|
| `src/lifemate/domain/` | Kiểu dữ liệu và khái niệm lõi | Khi tìm hiểu sample, trajectory, tool call |
| `src/lifemate/ports/` | Giao diện giữa agent và adapters | Khi thay backend hoặc tích hợp thành phần |
| `src/lifemate/adapters/` | Kết nối LLM, ASR, sandbox, dataset và exporter | Khi cấu hình hoặc sửa tích hợp |
| `src/lifemate/workflow/` | Điều phối hội thoại, lập kế hoạch, gọi tool và guardrail | Khi thay đổi hành vi agent |
| `src/lifemate/evaluation/` | Metric và scorer nội bộ | Khi đọc kết quả đánh giá |
| `src/lifemate/ui/` | Giao diện xem trajectory | Khi cần xem dữ liệu trực quan |
| `training/data_prep/` | Tạo, chuẩn hóa và xuất dữ liệu | Trước khi train |
| `training/llm/data/sft/` | Các bản SFT ChatML/ShareGPT | Chọn tập train và validation |
| `training/llm/data/distillation/` | Teacher trajectories, bản repair và dữ liệu trung gian | Truy nguồn mẫu SFT |
| `training/llm/scripts/` | Trainer Unsloth và đánh giá checkpoint | Train/evaluate LLM |
| `training/asr/` | Trainer PhoWhisper-large LoRA và đánh giá WER/CER | Fine-tune ASR; inference adapter nằm ở `src/lifemate/adapters/asr/` |
| `scripts/research/` | Phân tích/kiểm tra theo mẫu cụ thể | Tra cứu kết quả nghiên cứu cũ |
| `experiments/` | Log/JSON đầu ra của các lần thử | So sánh các lượt chạy |

## Luồng dữ liệu và runtime

```text
audio + context + history
        ↓
ASR adapter → workflow/agent → sandbox tools
                              ↓
                tool calls + state/result
                              ↓
                    exporter → submission
```

Teacher/repair trajectories nằm trong `training/llm/data/distillation/`; dữ liệu hội thoại dùng để fine-tune nằm trong `training/llm/data/sft/`. Trainer hiện đọc bộ chuẩn `train_chatml.jsonl` và `val_chatml.jsonl`; harmonized là biến thể riêng, cần chọn rõ trước khi train.

## Dữ liệu cuộc thi

`ZAIC_DATA_ROOT` là thư mục gốc chứa `train/` và `public_test/`. Có thể đặt biến môi trường này hoặc truyền `--data-root` cho `main.py`. Mặc định chương trình dùng thư mục project.

Ví dụ PowerShell:

```powershell
$env:ZAIC_DATA_ROOT = 'D:\path\to\train+public_test'
python main.py --split public_test --mode ollama --output submission.jsonl
```

## Điểm bắt đầu

- Sửa hành vi agent: `src/lifemate/`.
- Chuẩn bị dữ liệu: `training/data_prep/`.
- Fine-tune/evaluate LLM: `training/llm/scripts/`.
- Fine-tune/evaluate ASR: `training/asr/scripts/main.py` ([hướng dẫn](../training/asr/README.md)).
- Tìm hiểu thay đổi tổ chức: [project_reorganization_plan.md](project_reorganization_plan.md).
