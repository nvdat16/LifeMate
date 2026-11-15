# Data preparation

Các script trong thư mục này là bước chuẩn bị trước khi train, không phải agent runtime.

- `prepare_finetune_dataset.py`, `export_sft_dataset.py`: tạo/xuất dữ liệu SFT.
- `harmonize_sft_dataset.py`, `synchronize_unified_system_prompt.py`, `apply_targeted_ask_missing_info_prompt.py`: chuẩn hóa biến thể dữ liệu.
- `distill_train_dataset.py`, `run_luna_10_samples.py`, `run_single_agent_gpt6_terra.py`, `run_terra_on_failed_samples.py`, `run_sol_on_remaining_26.py`: sinh hoặc sửa trajectories bằng teacher.
- `merge_master_dataset.py`, `merge_sol_repairs.py`: tổng hợp output.

Dữ liệu SFT và distillation được lưu ở `training/llm/data/`. Dữ liệu cuộc thi được đọc từ `ZAIC_DATA_ROOT` (thư mục chứa `train/` và `public_test/`). Kiểm tra `--help` hoặc phần argparse của script trước khi chạy vì một số script ghi output.
