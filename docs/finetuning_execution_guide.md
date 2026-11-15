# Hướng Dẫn Thực Thi Huấn Luyện Fine-Tuning (SFT) Cho Mô Hình LifeMate
**Cuộc thi:** Zalo AI Challenge 2026 — Track 01: Agentic Voice AI  
**Mục tiêu:** Huấn luyện mô hình Single-Agent sub-5B đạt điểm State Match tiệm cận 100%, phản hồi < 2.5s / sample trên GPU NVIDIA A30.

---

## 1. TỔNG QUAN TẬP DỮ LIỆU ĐÃ CHUẨN HÓA

Tập dữ liệu SFT được lưu tại [`training/llm/data/sft/`](../training/llm/data/sft/). Bản harmonized mới nhất đạt 100% State Match và điểm trung bình 86.55%:

| Tên File | Số Lượng Mẫu | Định Dạng Chuẩn | Mục Đích Sử Dụng |
| :--- | :---: | :---: | :--- |
| [`harmonized_train_chatml.jsonl`](../training/llm/data/sft/harmonized_train_chatml.jsonl) | **1,450 mẫu (95%)** | ChatML (`messages`) | Tập huấn luyện chính thức (Train Split) cho Unsloth / HuggingFace `SFTTrainer`. |
| [`harmonized_val_chatml.jsonl`](../training/llm/data/sft/harmonized_val_chatml.jsonl) | **76 mẫu (5%)** | ChatML (`messages`) | Tập thẩm định (Validation Split) kiểm tra State Match sau mỗi 0.5 Epoch. |
| [`harmonized_dataset_info.json`](../training/llm/data/sft/harmonized_dataset_info.json) | — | JSON Metadata | Thống kê phân phối độ dài token (Trung bình: 5,941 tokens, Max: 10,751 tokens). |

---

## 2. LỰA CHỌN MÔ HÌNH NỀN TẢNG (BASE MODEL)

* **Model khuyến nghị:** `unsloth/Qwen2.5-3B-Instruct`
  * **Số tham số:** 3.09 tỷ tham số $\rightarrow$ **Tuân thủ 100% trần 5.0 tỷ tham số của BTC**.
  * **Tại sao là Qwen2.5-3B?** Dòng Qwen2.5 hiện là model mã nguồn mở số 1 thế giới ở khả năng sinh JSON có cấu trúc (Structured Outputs) và Function Calling. Kích thước 3B cho phép tốc độ suy luận cực nhanh (~1.1s – 1.8s / lượt trên GPU A30 24GB).

---

## 3. THIẾT KẾ HUẤN LUYỆN ĐẶC BIỆT: RESPONSE-ONLY LOSS MASKING

Trong các tác vụ Agentic, một sai lầm phổ biến là cho mô hình học mất mát (loss) trên toàn bộ chuỗi ký tự. Điều này khiến mô hình tốn 80% dung lượng tham số chỉ để cố học thuộc lòng schema của 20+ tool và các bảng dữ liệu trả về từ môi trường (Observation).

Pipeline trong script [`training/llm/scripts/train_sft.py`](../training/llm/scripts/train_sft.py) áp dụng **Response-Only Loss Masking**:
* **Mask bỏ (Label = -100):** System Prompt, User Voice Request, và các khối `OBSERVATIONS FROM TOOL EXECUTION`.
* **Chỉ tính Loss (Compute Gradient):** Các khối phản hồi của Assistant:
  ```json
  {"thought": "...", "tool_calls": [...], "is_finished": ...}
  ```
$\rightarrow$ **Kết quả:** Mô hình học tập trung 100% năng lực vào việc: *Khi gặp tình huống $X$ và dữ liệu $Y$, thì suy luận (thought) như thế nào và gọi tool gì.*

---

## 4. CÁC BƯỚC THỰC HIỆN HUẤN LUYỆN (EXECUTION STEPS)

### Bước 1: Cài đặt môi trường Unsloth & MLflow

**Cách 1: Cài đặt tiêu chuẩn trên Kaggle (Khuyên dùng)**
```bash
pip install unsloth mlflow
```

**Cách 2: Cài đặt chuẩn khi có xung đột PyTorch / CUDA trên Kaggle**
```bash
pip install pip3-autoremove
pip-autoremove torch torchvision torchaudio -y
pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
pip install unsloth mlflow
```

### Bước 2: Chạy Huấn Luyện Tối Ưu cho Qwen3-4B-Instruct-2507 (với RSLoRA & MLflow)

Khởi chạy script huấn luyện với các siêu tham số đã tối ưu hóa cho Kaggle:

```bash
python training/llm/scripts/train_sft.py \
    --model_name "unsloth/Qwen3-4B-Instruct-2507" \
    --max_seq_length 8192 \
    --load_in_4bit True \
    --batch_size 1 \
    --grad_accum 16 \
    --epochs 3 \
    --lr 1e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.08 \
    --weight_decay 0.01 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --use_rslora True \
    --neftune_noise_alpha 5.0 \
    --mask_user_and_tool_turns True \
    --eval_steps 25 \
    --save_total_limit 2 \
    --report_to mlflow \
    --mlflow_experiment "lifemate-zaic-2026" \
    --mlflow_run_name "qwen3-4b-sft-r32-lr1e4-seq8192" \
    --save_lora_dir "/kaggle/working/lora_adapter" \
    --export_gguf q8_0
```

* **Các cơ chế bảo vệ tối ưu:**
  * **Multi-turn Response-only Masking:** Chỉ tính loss trên lượt trả lời của Assistant, bỏ qua 85% token của System Schema và Sandbox Tool Results.
  * **RSLoRA & NEFTune:** Ổn định rank hội tụ, chống hiện tượng học vẹt các tên người/địa chỉ cụ thể trong tập train.
  * **Bảo hiểm chống tràn ổ cứng Kaggle:** Tự động lưu LoRA Adapter trước (~130MB an toàn), dọn dẹp các checkpoint tạm trước khi xuất file `q8_0` GGUF (~4.2GB).

### Bước 3: Xem Dashboard Đồ Thị MLflow

Để mở giao diện trực quan theo dõi đồ thị loss và siêu tham số:
```bash
mlflow ui --port 5000
```
Truy cập vào trình duyệt tại: `http://localhost:5000`

---

## 5. THẨM ĐỊNH CHECKPOINT TRÊN TẬP VALIDATION (76 MẪU)

Sau khi hoàn tất quá trình huấn luyện, chạy script đánh giá để tìm ra checkpoint có điểm số thực tế cao nhất trên môi trường Sandbox:

```bash
python training/llm/scripts/evaluate_checkpoint.py \
    --checkpoint_dir "training/llm/outputs/qwen2.5_3b_lifemate_sft/checkpoint-360" \
    --val_file "training/llm/data/sft/harmonized_val_chatml.jsonl"
```

* Tiêu chí lựa chọn Checkpoint vô địch:
  1. Tỷ lệ **State Match** cao nhất ($\ge 95\%$).
  2. Điểm **Tool Call F1** cao nhất ($\ge 85\%$).
  3. Tỷ lệ cú pháp JSON hợp lệ đạt **100%**.

---

## 6. ĐÓNG GÓI VÀ DEPLOY SUY LUẬN TỐC ĐỘ CAO (vLLM / OLLAMA)

Script đã tự động gộp (merge) trọng số LoRA vào base model và xuất thư mục 16-bit:
* Đường dẫn: `training/llm/outputs/qwen2.5_3b_lifemate_sft/merged_16bit/`

### Khởi chạy Inference Engine với vLLM (Độ trễ < 1.5s):
```bash
vllm serve training/llm/outputs/qwen2.5_3b_lifemate_sft/merged_16bit \
    --port 8000 \
    --max-model-len 11000 \
    --gpu-memory-utilization 0.85
```

### Chạy đánh giá toàn bộ 634 mẫu `public_test` để xuất kết quả nộp bài:
```bash
python main.py --pipeline single --model-url "http://localhost:8000/v1"
```
File `submission.json` sinh ra sẽ sẵn sàng 100% để nộp thẳng lên Leaderboard của BTC.
