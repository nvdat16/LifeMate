# Kế Hoạch Hành Động & Lộ Trình Fine-Tuning Mô Hình LifeMate (Zalo AI Challenge 2026)

> **Mục tiêu:** Xây dựng hệ thống Agentic Voice AI đạt điểm tuyệt đối (State Match 100%), phản hồi dưới 3 giây/mẫu trên GPU NVIDIA A30, và tuân thủ 100% điều lệ kỹ thuật của BTC (< 5B LLM, < 6B toàn hệ thống).  
> **Chiến lược cốt lõi:** Chuyển dịch từ Multi-Agent cồng kềnh sang **Single Fine-Tuned Agent (KISS Principle)** — Chưng cất tri thức (Knowledge Distillation) từ Frontier Models (GPT-6.1-Terra / GPT-5.6-Luna) vào model mã nguồn mở sub-5B.

---

## 📌 BẢNG DANH MỤC CÔNG VIỆC CẦN THỰC HIỆN (ACTION ITEMS)

| STT | Hạng Mục Công Việc | Công Cụ / Model | Trạng Thái / Mục Tiêu |
| :---: | :--- | :--- | :--- |
| **1** | Chạy Pipeline sinh dữ liệu ReAct Trajectory trên 1.545 mẫu train | `gpt-5.6-luna` / `gpt-6.1-terra` | Xuất file `golden_react_train.jsonl` (State match 100%) |
| **2** | Tự động hóa bộ lọc Verifier kiểm chứng state_hash | `zaic_tools.ToolEnv` | Chỉ nhận mẫu đạt hash khớp 100% với Gold BTC |
| **3** | Chuẩn bị dữ liệu huấn luyện (Data Formatting cho SFT) | Python / ShareGPT / ChatML | Định dạng hội thoại multi-turn (User $\rightarrow$ Assistant ReAct $\rightarrow$ Observation $\rightarrow$ Assistant) |
| **4** | Xây dựng pipeline Fine-tuning LoRA với Unsloth | `Unsloth` / `SFTTrainer` | Lưu checkpoint đều đặn mỗi **0.5 epoch** |
| **5** | Đánh giá so sánh các Checkpoints trên tập validation (50 mẫu) | `LocalScorer` | Chọn checkpoint có điểm F1 và State Hash cao nhất |
| **6** | Đóng gói model đã merge và nạp vào Single Agent Pipeline | `Ollama` / `vLLM` GGUF/16bit | Tốc độ suy luận $\le 2.5\text{s}$ / sample |
| **7** | Chạy toàn bộ 634 mẫu `public_test` và xuất submission | `main.py --pipeline single` | Nộp bài lên Leaderboard hàng ngày |

---

## 🔬 GIAI ĐOẠN 1: SINH DỮ LIỆU SYNTHETIC TỪ FRONTIER MODEL

### 1.1 Tại sao cần sinh dữ liệu thay vì dùng thẳng `train.jsonl` của BTC?
- **Nhãn của BTC là danh sách phẳng (Flat List):** BTC chỉ cung cấp mảng các tool sau cùng, không có phân chia lượt thoại (Turn 1 tra cứu $\rightarrow$ Turn 2 điền tham số).
- **Thiếu phần suy luận (Chain of Thought / Thought):** BTC không giải thích *tại sao* lại gọi tool đó. Thiếu `thought`, model sẽ không hiểu logic điều kiện (ví dụ: *chỉ dời lịch khi thời tiết báo mưa, trời không mưa thì thôi*).
- **BTC cho phép hoàn toàn hợp lệ:** Tab Rule cho phép: *"Synthetic data may be generated with other services or models (LLM, TTS, …) during development, but these must not be used at inference."*

### 1.2 Thiết kế Pipeline sinh dữ liệu tự động (`scripts/generate_golden_trajectories.py`)
- **Teacher Model khuyến nghị:**
  - `gpt-5.6-luna`: Tốc độ cực nhanh (~12s / sample), chi phí rẻ, sinh JSON nén gọn, tư duy quyết đoán.
  - `gpt-6.1-terra`: Dùng cho các mẫu phức tạp có điều kiện rẽ nhánh sâu (4+ tool calls).
- **Cơ chế Verifier đóng vòng (Closed-loop Verification):**
  1. Đọc từng mẫu trong `train.jsonl` (câu lệnh người dùng + full schema của installed apps).
  2. Teacher Model chạy qua vòng lặp ReAct:
     - **Turn 1 (Read):** Lập kế hoạch + Gọi tool đọc (`weather.get_forecast`, `calendar.list_events`, `contacts.search`).
     - **Sandbox thật:** Trả về kết quả thực tế vào Observation.
     - **Turn 2 (Write):** Trích xuất ID từ observation và điền vào tool ghi (`calendar.update_event`, `sms.send`).
     - **Turn 3 (Finish):** Xuất `is_finished: true` và `final_response`.
  3. Lấy `state_hash` từ Sandbox và so khớp với Gold Hash của BTC:
     - **Khớp 100%:** Lưu trajectory vào tập dữ liệu vàng (`golden_react_train.jsonl`).
     - **Chưa khớp:** Tự động retry tối đa 2 lần với prompt tăng cường.

---

## ⚙️ GIAI ĐOẠN 2: THIẾT KẾ PIPELINE FINE-TUNING (SFT VỚI UNSLOTH)

### 2.1 Lựa chọn Base Model tối ưu nhất theo luật BTC

| Mô hình Base | Số Tham Số | Tuân Thủ Luật BTC | Ưu Điểm Nổi Bật | Đánh Giá |
| :--- | :---: | :---: | :--- | :---: |
| **`Qwen/Qwen2.5-3B-Instruct`** | **3.09B** | ✅ Tuyệt đối (< 5B) | Khả năng tuân thủ JSON và Function Calling số 1 thế giới ở phân khúc 3B; suy luận siêu tốc (1.2s/sample trên A30). | ⭐⭐⭐⭐⭐ **(Lựa chọn số 1)** |
| **`unsloth/Qwen3-4B-Instruct-2507`** | **4.02B** | ✅ Tuyệt đối (< 5B) | Phiên bản kiến trúc mới nhất của Qwen, dung lượng tri thức lớn hơn, hiểu ngữ cảnh phức tạp tốt. | ⭐⭐⭐⭐ |
| **`Qwen2.5-1.5B-Instruct`** | **1.54B** | ✅ Tuyệt đối (< 5B) | Cực nhẹ, train rất nhanh, nhưng có thể bị giảm độ chính xác ở các sample có 4+ tools. | ⭐⭐⭐ |

*Lưu ý: Dòng 7B (như Qwen2.5-7B) bị cấm tuyệt đối vì vi phạm trần 5B của BTC.*

---

### 2.2 Kiến Trúc & Cấu Hình Huấn Luyện (Training Hyperparameters)

- **Framework:** `Unsloth` (tối ưu bộ nhớ x2, tốc độ x5 so với HuggingFace thuần).
- **Phương pháp:** QLoRA 4-bit (chỉ tốn ~6GB VRAM trên Colab/Kaggle T4).
  - LoRA Rank $r = 16$ hoặc $r = 32$.
  - LoRA Alpha $\alpha = 32$.
  - Target Modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.
- **Cấu hình Epochs & Checkpointing (Theo yêu cầu lưu mỗi 0.5 Epoch):**
  - **Số Epoch:** 3 Epochs.
  - Với 1.500 mẫu, batch size = 2, gradient accumulation = 4 $\rightarrow$ 1 epoch $\approx$ 188 steps.
  - **Lưu Checkpoint mỗi 0.5 Epoch:** Đặt `save_steps = 94` steps (tương đương mỗi nửa epoch lưu 1 lần).
  - Cấu hình lưu trữ: `save_total_limit = 6` (lưu trọn vẹn 6 checkpoint trong suốt 3 epoch: Epoch 0.5, 1.0, 1.5, 2.0, 2.5, 3.0).
- **Learning Rate & Scheduler:**
  - Learning rate: `2e-4`.
  - LR Scheduler: `cosine` với warmup ratio `0.05`.
  - Max Sequence Length: `max_seq_length = 4096` (thoải mái chứa 14-20 tool schemas đầy đủ).

---

## 💡 CÁC Ý TƯỞNG & ĐÓNG GÓP CHIẾN LƯỢC MỚI (EXPERT INSIGHTS)

### 1. Kỹ thuật "Data Augmentation" chống Domain Shift cho 11 App mới ở Public Test
- **Vấn đề phát hiện hôm nay:** Tập Public Test có **11 app hoàn toàn mới** (`grocery`, `stocks`, `ecommerce`, `pantry`, `habit_tracker`, `recipes`, `parking`, `event_tickets`, `train_bus_tickets`, `public_transit`, `translator`) chưa từng có trong tập train!
- **Giải pháp:** Khi chuẩn bị dữ liệu train, **tuyệt đối không huấn luyện model nhớ tên app cố định**. 
  - Hãy dùng `gpt-5.6-luna` sinh thêm ~300 mẫu synthetic dựa trên chính schema của 11 app mới trong file `public_test_tools_catalog.json`.
  - *Lưu ý:* Việc dùng catalog public để sinh mẫu synthetic là **hoàn toàn hợp lệ**, miễn là không được gán nhãn trên các file `.wav` của `public_test`.

### 2. Kỹ thuật "Nhiễm bẩn ASR" (Acoustic Perturbation / Noise Injection)
- **Thực tế:** Tập `train` có sẵn văn bản chuẩn 100% (`audio_text`), nhưng tập `public_test` khi suy luận sẽ phải đi qua PhoWhisper. ASR chắc chắn sẽ có tỷ lệ sai sót nhỏ (thiếu dấu, từ địa phương Bắc/Trung/Nam, viết thường tên riêng).
- **Chiến thuật:** Khi đưa dữ liệu vào SFT, hãy áp dụng xác suất 20% tự động làm nhiễu nhẹ text (bỏ dấu viết hoa, chuyển chữ số thành chữ viết hoặc ngược lại, ví dụ "chín giờ" $\leftrightarrow$ "9:00", "Đà Nẵng" $\leftrightarrow$ "đà nẵng").
- **Hiệu quả:** Model nhỏ 3B sau khi train sẽ cực kỳ "lì lợm", không bị bỡ ngỡ trước văn bản ASR thực tế của tập test!

### 3. Cân bằng trọng số cho 2 kịch bản đặc thù: Out-of-Scope (7.2%) & Missing Info (14%)
- Trong tập train:
  - Có **7.2% mẫu Out-of-Scope** $\rightarrow$ Model phải học được phản xạ dứt khoát: nếu context không có app tương ứng $\rightarrow$ Xuất ngay `"tool_calls": []` và `is_finished: true`, tuyệt đối không cố bịa tool.
  - Có **14.0% mẫu Missing Info** $\rightarrow$ Model phải biết dừng lại và gọi `sys.ask_missing_info` thay vì đoán mò tham số.

### 4. Đóng gói Docker nộp bài theo chuẩn BTC
- Sử dụng framework **vLLM** hoặc **llama.cpp / Ollama** đã tối ưu hóa PagedAttention và FlashAttention-2.
- Trên GPU chấm thi của BTC (NVIDIA A30 24GB):
  - Model ASR PhoWhisper: ~1.5 GB VRAM.
  - Model LLM Qwen2.5-3B (đã fine-tune): ~6.0 GB VRAM.
  - Tổng VRAM chiếm dụng chỉ ~7.5 GB / 24 GB $\rightarrow$ Dư dả hơn 16 GB cho KV Cache!
  - Thời gian xử lý: Toàn bộ quá trình ASR + LLM ReAct chỉ tốn **~1.5 đến 2.5 giây / sample** (bỏ xa mốc timeout 30s của BTC, cầm chắc lợi thế Tie-Breaking).

---

## 📅 LỊCH TRÌNH THỰC HIỆN ĐỀ XUẤT

```
Tuần 1:
├── Ngày 1: [HOÀN THÀNH 100%] Chạy chưng cất dữ liệu ReAct trên 1.545 mẫu train bằng gpt-5.6-luna.
│   └── Kết quả thực tế: 1.545/1.545 mẫu | State Match: 94.37% (1.458/1.545) | Avg F1: 75.83% | 838 mẫu tuyệt đối 1.00!
├── Ngày 2: Data Augmentation (sinh bổ sung kịch bản cho 11 app mới & inject nhiễu ASR).
├── Ngày 3: Setup Colab/Kaggle chạy Unsloth Fine-tune Qwen2.5-3B-Instruct (lưu checkpoint mỗi 0.5 epoch).
└── Ngày 4: Chạy LocalScorer đánh giá 6 checkpoint để chọn phiên bản tốt nhất.

Tuần 2:
├── Ngày 5: Chạy full 634 mẫu public_test với model đã fine-tune qua Single Agent Pipeline.
└── Ngày 6: Đóng gói Docker container hoàn chỉnh theo đúng cấu hình chấm thi của BTC.
```

