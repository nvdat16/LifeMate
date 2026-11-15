# KIẾN TRÚC KỸ THUẬT & CHIẾN LƯỢC TRIỂN KHAI — ZALO AI CHALLENGE 2026 (LIFEMATE)

> **Dự án:** LifeMate — Voice-Driven Multi-step Agentic AI  
> **Tác giả:** Senior AI Systems Architect  
> **Ngày lập:** 29/10/2026  
> **Workspace:** thư mục checkout của `zaic-2026` (không phụ thuộc đường dẫn máy cá nhân)  

---

## MỤC LỤC
1. [Phân tích Cơ chế Chấm điểm & Trọng số Tối ưu (Metric-Driven Analysis)](#1-phân-tích-cơ-chế-chấm-điểm--trọng-số-tối-ưu)
2. [Kiến trúc End-to-End Pipeline (ASR + Multi-step Agent Loop)](#2-kiến-trúc-end-to-end-pipeline-asr--agent-loop)
3. [Chiến lược Huấn luyện & Fine-tuning (1,545 Mẫu Train)](#3-chiến-lược-huấn-luyện--fine-tuning)
4. [Thiết kế Cấu trúc Codebase (Modular Architecture & Stubs)](#4-thiết-kế-cấu-trúc-codebase-modular-architecture)
5. [Lộ trình Triển khai Chi tiết (Actionable Roadmap 29/10 - 23/11)](#5-lộ-trình-triển-khai-chi-tiết)

---

## 1. PHÂN TÍCH CƠ CHẾ CHẤM ĐIỂM & TRỌNG SỐ TỐI ƯU

### 1.1 Giải mã Hệ thống Điểm số & Phạt Lỗi (Scoring Mechanics)
Mỗi dòng trong `submission.jsonl` gồm ba trường: `id`, `tool_calls` (danh sách tuần tự), và `state_hash` (chuỗi SHA-256 từ canonical state). Replay engine của BTC sẽ tái hiện tuần tự `tool_calls` trên `state.json` gốc thông qua thư viện `zaic_tools`.

```
                    ┌───────────────────────────┐
                    │  submission: tool_calls   │
                    └─────────────┬─────────────┘
                                  │ Replay via zaic_tools
                                  ▼
┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│ state.json (gốc) ├───►│ Replayed Env Exec ├───►│ final_state_hash │
└──────────────────┘    └───────────────────┘    └────────┬─────────┘
                                                          │ So khớp SHA-256
                                                          ▼
                                                 ┌──────────────────┐
                                                 │ State Match (0/1)│
                                                 └──────────────────┘
```

#### Ma trận Trọng số & Rủi ro Phạt điểm:
1. **Sai lệch `state_hash` (Tử huyệt P0):**
   - Nếu `state_hash` sai khác dù chỉ 1 bit, toàn bộ điểm trạng thái của sample bị mất (0 điểm).
   - **Đặc điểm Canonical State:** Theo `zaic_tools/utils.py`, `canon()` loại bỏ:
     - Toàn bộ metadata `_meta`.
     - 5 App thông tin thời gian thực (`LIVE_APPS = {"weather", "gold_price", "fuel_price", "lottery", "exchange_rate"}`).
     - Các trường văn bản tự do (`FREE_TEXT_KEYS = {"address", "body", "content", "dropoff", "expression", "label", "location", "name", "note", "pickup", "subject", "title"}`).
     - Các scalar lists được sắp xếp (`sorted`), string được lower-case và collapse khoảng trắng (`"700 cm"` == `"700cm"`).
   - **Hệ quả kiến trúc:**
     - Các app tra cứu thông tin (như `weather.get_current`) có thể gọi an toàn mà **không sợ làm bẩn `state_hash`**.
     - Các trường tự do như nội dung tin nhắn (`sms.send.content`) hay địa chỉ điểm đến (`ride_hailing.book.to`) nếu có sai khác ký tự nhỏ sẽ không làm hỏng `state_hash` (nhưng ảnh hưởng token-F1 của trajectory).
     - Tuy nhiên, các giá trị định danh/logic cốt lõi (`device_id`, `amount`, `status`, `time`, `phone_number`, `contact_id`) ảnh hưởng trực tiếp tới `state_hash` → Tuyệt đối không được sai sót.

2. **Rejected Tool Call do vi phạm Schema (Double Penalty P0):**
   - Theo `zaic_tools/validate.py`, hàm `schema_error()` là cơ chế duy nhất khiến tool call bị từ chối (`{"ok": false, "error": "..."}`).
   - Khi bị reject:
     - `env.get_tool_calls()` **vẫn ghi nhận** call lỗi vào lịch sử nộp bài (làm loãng/sai trajectory F1).
     - State không đổi → dẫn tới chuỗi write tiếp theo có thể bị đứt gãy hoặc không đạt `state_hash` mục tiêu.

3. **Lỗi logic nghiệp vụ (Non-schema Errors):**
   - Trong `zaic_tools`, ID không tồn tại, số dư tài khoản không đủ, hoặc kết quả tìm kiếm rỗng **không gây exception** và **không bị reject schema**; tool trả về `{"ok": true, "done": false, "message": "..."}`.
   - Nếu Agent không đọc phản hồi này mà cứ tiến hành bước kế tiếp, state sẽ không thay đổi như kỳ vọng, làm sai `state_hash`.

4. **Kịch bản Ask Missing Info & Out-of-scope:**
   - **Ask Missing Info (15.4% dữ liệu):** Phải gọi duy nhất `sys.ask_missing_info` với arguments `{"tool_name": "<app>.<tool>", "params": ["<param_1>", ...]}`. Thiếu param hoặc sai tên tool sẽ mất điểm kịch bản này.
   - **Out-of-scope (24.4% dữ liệu):** Nếu người dùng yêu cầu việc ngoài khả năng của các app đã cài (ví dụ: đòi bay vũ trụ hoặc app chưa cấp quyền), phải trả về danh sách rỗng `tool_calls: []`. Bất kỳ 1 tool call thừa thãi nào trong trường hợp này đều khiến sample bị đánh tụt điểm nghiêm trọng.

---

### 1.2 Chiến lược Tận dụng Catalog Metadata & Guardrails
Để ngăn chặn hoàn toàn Rejected Calls và bảo toàn `state_hash`, toàn bộ luồng tương tác với môi trường phải đi qua lớp **Active Guardrail Engine**:

```
                         [LLM Tool Prediction]
                                   │
                                   ▼
                   ┌──────────────────────────────┐
                   │  Guardrail 1: App White-list │  (Chặn tool không thuộc context.installed_apps)
                   └──────────────┬───────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │  Guardrail 2: Schema Checker │  (env.validate(name, args))
                   └──────────────┬───────────────┘
                                  │ Thất bại (Schema Error)
                                  ├──► Tự động sửa/format (Auto-caster: date, int, enum)
                                  │    Nếu vẫn lỗi -> LLM Reflection Retry (Internal)
                                  ▼ Hợp lệ
                   ┌──────────────────────────────┐
                   │  Guardrail 3: Pre-check Ref  │  (Kiểm tra x-ref đã tồn tại từ read cache chưa)
                   └──────────────┬───────────────┘
                                  ▼
                         [env.call(name, args)]
```

- **Khai thác `kind: "read"` vs `kind: "write"`:**
  - Quy tắc bất di bất dịch: **Read-before-Write**. Không bao giờ gọi một write tool có chứa `x-ref` (ví dụ `device_id`, `contact_id`, `playlist_id`, `event_id`) khi chưa có bước `read` (`get_status`, `search`, `list`) tương ứng trước đó.
- **Khai thác `x-one-of`:**
  - Ví dụ `sms.send` yêu cầu `x-one-of: [["contact_id"], ["phone_number"]]`. Nếu danh bạ tìm thấy contact thì dùng `contact_id`; nếu số điện thoại xuất hiện trực tiếp trong audio transcript thì dùng `phone_number`. Bộ tiền xử lý tự động validate cấu trúc này.
- **Khai thác `x-match: "fuzzy"`:**
  - Với các trường nội dung text tự do, ưu tiên trích xuất verbatim hoặc chuẩn hóa bằng `_fold()` (bỏ dấu tiếng Việt, lowercase) để tối đa hóa Token-F1 mà không cần ép khuôn máy móc.

---

## 2. KIẾN TRÚC END-TO-END PIPELINE (ASR + AGENT LOOP)

Hệ thống được thiết kế theo mô hình tách bạch 2 tầng (Two-Stage Decoupled Architecture): **Perception (ASR) + Action (ReAct Agent)**.

```
                    ┌──────────────────────────────┐
                    │      Audio Request (.wav)    │
                    │   + History Audios (_u*.wav) │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       TẦNG 1: ASR CORE       │
                    │   (PhoWhisper / Whisper-v3)  │
                    └──────────────┬───────────────┘
                                   │ Raw Transcript
                                   ▼
                    ┌──────────────────────────────┐
                    │   Text Normalization Layer   │
                    │  (Số, ngày giờ, code-switch) │
                    └──────────────┬───────────────┘
                                   │ Clean Transcripts
                                   ▼
                    ┌──────────────────────────────┐
                    │  TẦNG 2: CORE REACT AGENT    │
                    │      (Qwen2.5-3B-Instruct)   │
                    │  Input: Prompt + Context     │
                    └──────────────┬───────────────┘
                                   │
             ┌─────────────────────┴──────────────────────┐
             │                                            │
             ▼                                            ▼
   [Loop: Plan & Call Read]                      [Out-of-Scope Detect]
             │                                            │
             ▼                                            ▼
   [Observe Tool Output]                             tool_calls = []
             │
             ▼
   [Decision: Missing Info?]
             ├── Yes ──► Call sys.ask_missing_info
             └── No  ──► Validate & Call Write Tools
                                   │
                                   ▼
                         [env.get_state_hash()]
```

### 2.1 Tầng ASR (Speech-to-Text)
- **Base Model Đề xuất:**
  1. **Lựa chọn 1 (Ưu tiên):** `vinai/PhoWhisper-large` — Huấn luyện chuyên sâu trên ngữ âm người Việt, vượt trội về nhận diện phương ngữ Bắc - Trung - Nam và độ kháng nhiễu môi trường.
  2. **Lựa chọn 2:** `openai/whisper-large-v3` (hoặc `whisper-large-v3-turbo` để tối ưu VRAM/latency) — Mạnh về code-switching (các từ tiếng Anh phổ biến trong digital life: "smart home", "turn on", "AC", "Grab", "Shopee", "ZaloPay").
- **Chiến lược xử lý Conversation History:**
  - Mỗi mẫu có thể có các file audio lịch sử `_u1.wav`, `_u2.wav`.
  - Không chạy tuần tự làm nghẽn pipeline: gom batch toàn bộ audio của sample vào 1 forward pass duy nhất của ASR.
  - Lưu cache transcription của `history` để tránh dịch lại nhiều lần.
- **ASR Post-Processing & Normalization:**
  - Quy đổi số tự nhiên bằng chữ sang chữ số: `"hai mươi sáu độ"` → `"26 độ"`, `"năm trăm nghìn"` → `"500000"`.
  - Chuẩn hóa thời gian tương đối dựa vào `context.current_time`: `"ba giờ chiều mai"` kết hợp với `current_time: 2026-11-28T07:10:00+07:00` → `2026-11-29T15:00:00+07:00`.
  - Từ điển định danh tiếng Anh thường gặp: `"gu gồ" -> "google"`, `"ai si" -> "ac"`, `"pét" -> "pass"`.

### 2.2 Tầng Core Agent (LLM Planner & Tool Caller)
- **Lựa chọn LLM Phù hợp Ràng buộc Phần cứng & Quy định BTC:**
  - **RÀNG BUỘC PHÁP LÝ BẮT BUỘC CỦA BTC (Tab Rule):** 
    - Tổng tham số toàn hệ thống: **< 6B parameters**.
    - Riêng LLM: **< 5B parameters**, context length tối đa **16,384 tokens**.
    - Giới hạn thời gian: **30 giây/sample**.
    - Phần cứng chấm Private: **1 GPU NVIDIA A30 (24GB), 16 vCPU Xeon Gold 6442Y, 64 GB RAM**.
  - **Lựa chọn tối ưu:** **`Qwen/Qwen2.5-3B-Instruct`** (3.09B parameters).
    - Cùng họ Qwen2.5 xuất sắc về Function Calling và tiếng Việt.
    - Kích thước 3.09B kết hợp với PhoWhisper (~0.5B - 1.5B) $\rightarrow$ Tổng tham số hệ thống **~3.6B - 4.6B < 6B** (Đạt chuẩn 100%!).
    - Tốc độ suy luận vượt trội (chỉ mất ~2-4s/sample, bỏ xa trần 30s của BTC).
    - Mức chiếm dụng VRAM chỉ khoảng **6 - 8 GB**, vận hành cực kỳ an toàn trên GPU NVIDIA A30 24GB.
- **Vòng lặp ReAct Động (Dynamic ReAct Loop):**
  - **Bước 1 (Lọc Catalog):** Chỉ inject schema của các tool nằm trong `context.installed_apps`. Điều này giảm context size từ 147 tools xuống còn ~10-25 tools cho mỗi sample, tăng độ chính xác chọn tool lên gấp 3 lần.
  - **Bước 2 (Scenario Classifier):** 
    - Nếu intent của user không khớp bất kỳ tool nào của installed_apps → Kết luận **Out-of-scope**, return `[]` ngay lập tức, tiết kiệm token.
    - Nếu thiếu thông tin thiết yếu không thể tra cứu được trong DB/Environment → Dự đoán `sys.ask_missing_info`.
  - **Bước 3 (ReAct Iteration):**
    - LLM sinh suy nghĩ (`Thought`) và lời gọi tool (`Action`).
    - Trước khi gửi đến `env.call()`, gọi `env.validate()`. Nếu validate fail, phản hồi lỗi schema ngay trong turn của ReAct để LLM tự sửa tham số trước khi ghi nhận vào lịch sử thật của `ToolEnv`.
    - Sau khi gọi read tool, quan sát kết quả (`Observation`) chứa danh sách `devices`, `contacts`, `events`...
    - LLM đọc kết quả, lấy ID chính xác (`device_id = "ac_living_room"`) và sinh lời gọi write tool (`smart_home.climate_set`).
  - **Bước 4 (Kết thúc & Hash):**
    - Khi LLM xuất ra token kết thúc hoặc không còn tool nào cần gọi, invoke `env.get_state_hash()` và trích xuất `env.get_tool_calls()`.

---

## 3. CHIẾN LƯỢC HUẤN LUYỆN & FINE-TUNING

### 3.1 Phân bổ Dữ liệu (Stratified Train/Val Split)
Tập `train.jsonl` có 1,545 mẫu với phân phối:
- **Execute (Write actions):** 930 mẫu (60.2%)
- **Out-of-scope (`tool_calls: []`):** 377 mẫu (24.4%)
- **Ask Missing Info (`sys.ask_missing_info`):** 238 mẫu (15.4%)

**Chiến lược Split:**
- Tỷ lệ: **90% Train (1,390 mẫu) / 10% Validation (155 mẫu)**.
- Thuật toán phân tầng đa nhãn (Multi-label Stratified Split):
  - Nhãn 1: Scenario Type (Execute / Ask / Out-of-scope).
  - Nhãn 2: App Groups chính xuất hiện trong label (Smart Home, Finance, Communication, Planning, Mobility...).
- Giữ cố định tập Validation này làm Benchmark cục bộ để đo lường tương quan với điểm Leaderboard công khai.

### 3.2 Pipeline Fine-tune ASR
- **Tập dữ liệu:** 1,545 cặp audio cuối + toàn bộ audio trong `history` có nhãn `audio_text` (tổng cộng ~2,800 đoạn thoại).
- **Kỹ thuật Huấn luyện:**
  - Base: `PhoWhisper-large`.
  - Phương pháp: LoRA trên Decoder Attention và MLP (rank r=16, alpha=32).
  - SpecAugment / Audio Perturbation: Thay đổi tốc độ phát (Speed perturbation 0.9x, 1.1x), thêm nhiễu nền giả lập môi trường xe hơi và phòng khách.
  - Mục tiêu: Giảm thiểu WER (Word Error Rate) trên các danh từ riêng, tên thiết bị tiếng Anh và các con số tài chính.

### 3.3 Pipeline Chuyển đổi Dữ liệu & SFT cho LLM
Để LLM học sâu cơ chế gọi tool đa bước, toàn bộ 1,545 mẫu train được chuyển thành dạng **Multi-turn Chat Completion with Function Calling**:

```json
[
  {"role": "system", "content": "You are LifeMate Voice Agent... Available Tools: [...] Context: {...}"},
  {"role": "user", "content": "Transcript: Tôi sắp về nhà, bật điều hòa phòng khách 26 độ và đặt xe đón tôi."},
  {"role": "assistant", "tool_calls": [{"name": "smart_home.get_status", "arguments": {}}]},
  {"role": "tool", "name": "smart_home.get_status", "content": "{\"ok\": true, \"devices\": [{\"id\": \"ac_living_room\", \"type\": \"ac\"}]}"},
  {"role": "assistant", "tool_calls": [
    {"name": "smart_home.climate_set", "arguments": {"device_id": "ac_living_room", "temperature": 26, "mode": "cool"}},
    {"name": "ride_hailing.book", "arguments": {"from": "227 Nguyễn Văn Cừ", "to": "123 Trần Hưng Đạo"}}
  ]}
]
```

- **Kỹ thuật Huấn luyện SFT:**
  - Base: `Qwen/Qwen2.5-3B-Instruct` (tuân thủ giới hạn < 5B parameters).
  - Format: ChatML với special tokens `<|im_start|>`, `<|im_end|>`.
  - Loss Masking: Chỉ tính Cross-Entropy loss trên phần token do Assistant sinh ra (`tool_calls` và `arguments`), mask toàn bộ token của User, System và Tool Result.
  - QLoRA (4-bit BitsAndBytes, r=32, alpha=64, target_modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Data Augmentation (Tăng cường Kháng Lỗi):**
  - **ASR Noise Injection:** Thay vì dùng 100% `audio_text` sạch của train, ta lấy ASR thô tự sinh từ audio và đưa vào prompt lúc train. Khi đó LLM học được cách tự sửa lỗi chính tả phát âm (ví dụ: ASR nghe nhầm "bật điều hoà" thành "bật đều hoà" thì LLM vẫn hiểu đúng intent).
  - **Paraphrasing:** Sử dụng synthetic variations để sinh thêm 2-3 cách nói khác nhau cho cùng một chuỗi tool call.

---

## 4. THIẾT KẾ CẤU TRÚC CODEBASE (MODULAR ARCHITECTURE)

Codebase được thiết kế theo tiêu chuẩn công nghiệp: phân tách rõ ràng trách nhiệm giữa Data, ASR, Agent Core, Training, Local Evaluation và Docker Packaging.

```text
zaic-2026/ (cấu trúc hiện tại xem tại [project_map.md](project_map.md))
├── docs/
│   ├── lifemate_specs.md             # Toàn bộ thông số & quy định kỹ thuật đã cào
│   └── implementation_strategy.md    # Tài liệu kiến trúc & chiến lược này
├── configs/
│   ├── asr_config.yaml               # Hyperparameters cho ASR (PhoWhisper)
│   ├── agent_config.yaml             # Prompt template, model path, generation config
│   └── train_config.yaml             # LoRA/QLoRA training parameters
├── lifemate/
│   ├── __init__.py
│   ├── data_processing/              # Module tiền xử lý và nạp dữ liệu
│   │   ├── __init__.py
│   │   ├── dataset.py                # DataLoader cho train.jsonl / public_test.jsonl
│   │   ├── audio_utils.py            # Đọc, resample và chuẩn hóa file WAV
│   │   ├── catalog_parser.py         # Lọc schema theo installed_apps
│   │   ├── normalizer.py             # Chuẩn hóa văn bản, ngày giờ, số
│   │   └── split.py                  # Stratified K-Fold split script
│   ├── asr/                          # Module Speech-to-Text
│   │   ├── __init__.py
│   │   ├── model.py                  # Wrapper cho PhoWhisper / Whisper inference
│   │   ├── history_transcriber.py    # Xử lý song song audio history
│   │   └── finetune_asr.py           # Script train ASR trên audio_text
│   ├── agent_runtime/                # Core Agent Execution Engine
│   │   ├── __init__.py
│   │   ├── agent.py                  # LifeMateAgent orchestrator
│   │   ├── llm_client.py             # Inference client (vLLM / HuggingFace Transformers)
│   │   ├── prompt_builder.py         # Xây dựng System & User prompts tối ưu
│   │   ├── react_loop.py             # Vòng lặp ReAct: Plan -> Validate -> Exec -> Hash
│   │   ├── guardrail.py              # Bộ lọc schema & logic trước khi gọi env.call
│   │   └── scenario_classifier.py    # Phân loại Execute / Ask Missing / Out-of-Scope
│   ├── training/                     # Module huấn luyện mô hình
│   │   ├── __init__.py
│   │   ├── format_sft_data.py        # Chuyển đổi tool_calls thành multi-turn chat
│   │   └── train_sft.py              # Script QLoRA SFT cho LLM
│   └── evaluation/                   # Module đánh giá nội bộ
│       ├── __init__.py
│       ├── scorer.py                 # Replay submission trên zaic_tools và tính điểm
│       └── metrics.py                # Tính State Match Rate, Tool F1, Token-F1
├── docker_submission/                # Cấu hình đóng gói vòng Private Test
│   ├── Dockerfile
│   ├── entrypoint.sh                 # Script chạy pipeline từ test.jsonl -> submission.jsonl
│   └── requirements.txt
├── scripts/                          # Các script phụ trợ / cào dữ liệu
├── run_inference.py                  # Entrypoint chính để chạy tạo submission.jsonl
└── run_eval.py                       # Entrypoint chạy benchmark trên tập validation
```

---

### Code Scaffold Stubs (Chi tiết Giao diện & Type Hints)

Dưới đây là thiết kế chi tiết các hàm giao diện chính (Stub functions kèm `#TODO` định hướng):

#### 1. `lifemate/agent_runtime/guardrail.py`
```python
"""Module kiểm tra và sửa lỗi schema trước khi gọi ToolEnv."""
from typing import Any, Dict, Tuple, Optional, Set

class ToolGuardrail:
    def __init__(self, catalog: Dict[str, Any], allowed_tools: Set[str]):
        """
        #TODO: Lưu trữ schema catalog của split hiện tại và danh sách allowed_tools
        của sample (lấy từ context.installed_apps + sys.ask_missing_info).
        """
        self.catalog = catalog
        self.allowed_tools = allowed_tools

    def pre_validate(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Kiểm tra schema nội bộ trước khi gọi env.validate().
        
        Args:
            tool_name: Tên tool dạng '<app>.<method>'
            arguments: Dict tham số do LLM sinh ra
            
        Returns:
            (is_valid, error_message, sanitized_arguments)
            
        #TODO:
        1. Kiểm tra tool_name có nằm trong self.allowed_tools hay không. Nếu không -> Báo lỗi.
        2. Tự động ép kiểu (Type Casting):
           - Ép chuỗi số thành int/float nếu parameter yêu cầu integer/number.
           - Ép date-time về định dạng ISO-8601 chuẩn regex.
        3. Kiểm tra ràng buộc 'x-one-of': Nếu thiếu cả hai nhóm param -> Cố gắng fallback hoặc báo lỗi.
        4. Trả về arguments đã được làm sạch (sanitized).
        """
        # TODO: Implement rule checking logic
        return True, None, arguments
```

#### 2. `lifemate/agent_runtime/react_loop.py`
```python
"""Vòng lặp ReAct điều khiển luồng tương tác giữa LLM và ToolEnv."""
from typing import Any, Dict, List, Tuple
# Import ToolEnv từ zaic_tools của split tương ứng

class ReActController:
    def __init__(self, llm_client: Any, guardrail: Any, max_steps: int = 6):
        """
        #TODO: Khởi tạo controller với LLM client, bộ guardrail và giới hạn số bước.
        """
        self.llm_client = llm_client
        self.guardrail = guardrail
        self.max_steps = max_steps

    def execute_sample(self, env: Any, transcript: str, context: Dict[str, Any], history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """
        Thực thi trọn vẹn 1 mẫu hội thoại, trả về (tool_calls, state_hash).
        
        Args:
            env: Phiên làm việc ToolEnv (được mở duy nhất một lần)
            transcript: Nội dung ASR của lượt nói cuối
            context: Dữ liệu bối cảnh người dùng
            history: Lịch sử các lượt nói trước đó
            
        Returns:
            Tuple[List[ToolCall], str]: tool_calls đã thực thi và state_hash cuối cùng
            
        #TODO:
        1. Kiểm tra Intent: Nếu là Out-of-Scope -> Return [], env.get_state_hash()
        2. Khởi tạo danh sách messages cho LLM bao gồm System prompt + Context + History + User transcript.
        3. Lặp qua các bước (tối đa max_steps):
           a. LLM dự đoán action tiếp theo (tool_name, arguments) hoặc quyết định kết thúc (DONE).
           b. Nếu LLM chọn DONE hoặc không sinh tool_call -> Break vòng lặp.
           c. Chạy qua guardrail.pre_validate(). Nếu sai -> Đưa lỗi vào prompt cho LLM sửa (chưa gọi env).
           d. Chạy env.validate(tool_name, arguments).
           e. Gọi env.call(tool_name, arguments) để ghi nhận kết quả và cập nhật state bộ nhớ.
           f. Nhận observation từ env.call(), append vào messages và tiếp tục bước sau.
        4. Trả về env.get_tool_calls() và env.get_state_hash().
        """
        # TODO: Implement step-by-step agent loop logic
        return env.get_tool_calls(), env.get_state_hash()
```

#### 3. `lifemate/evaluation/scorer.py`
```python
"""Module tái hiện chấm điểm nội bộ tương thích 100% với cách chấm của BTC."""
from typing import Dict, Any, List
from pathlib import Path

class LocalScorer:
    def __init__(self, split: str = "train", root_dir: str = "."):
        """
        #TODO: Nạp catalog và môi trường ban đầu từ <root_dir>/<split>/
        """
        self.split = split
        self.root_dir = Path(root_dir)

    def evaluate_submission(self, submission_file: str, ground_truth_file: str) -> Dict[str, float]:
        """
        Chấm điểm toàn diện file submission.jsonl.
        
        Returns:
            Dict chứa:
            - 'state_hash_accuracy': Tỷ lệ khớp hash tuyệt đối (P0)
            - 'tool_precision': Tỷ lệ tool gọi đúng
            - 'tool_recall': Tỷ lệ tool không bị sót
            - 'fuzzy_param_f1': Điểm Token-F1 trên các trường x-match: fuzzy
        
        #TODO:
        1. Đọc từng dòng của submission và ground truth theo ID tương ứng.
        2. Dùng Executor.replay() để chạy tool_calls trên state gốc.
        3. So khớp state_hash với label.final_state (qua zaic_tools.state_hash).
        4. So khớp trajectory của tool_calls và tính token-F1 trên các free text fields.
        5. Trả về báo cáo metrics tổng hợp.
        """
        # TODO: Implement full scoring replay logic
        return {"state_hash_accuracy": 0.0, "tool_f1": 0.0}
```

---

## 5. LỘ TRÌNH TRIỂN KHAI CHI TIẾT (ACTIONABLE ROADMAP)

Hạn chót quan trọng:
- **16/11/2026 (23:59):** Hạn chót nộp toàn bộ Model + Code huấn luyện + Inference code.
- **20/11/2026 (09:00):** BTC mở Private Test.
- **23/11/2026 (23:59):** Hạn chót nộp kết quả Private Test cuối cùng.

```
29/10               05/11               12/11         16/11         20/11         23/11
  │─── TUẦN 1 ────────│─── TUẦN 2 ────────│─── TUẦN 3 ────│── T4 ───────│─── TUẦN 5 ────│
  Scaffold Codebase   Baseline Agent      Fine-tuning     SUBMIT        PRIVATE TEST  
  Data EDA & Split    Zero-Shot LLM       ASR + QLoRA     MODELS &      Run Docker    
  ASR Benchmark       Local Scorer        Docker Build    CODE (23:59)  Final Submit  
```

### Chi tiết Kế hoạch Từng Giai đoạn:

| Tuần / Ngày | Hạng mục Công việc | Độ ưu tiên | Deliverables Cụ thể |
|---|---|:---:|---|
| **Tuần 1 (29/10 - 04/11)**<br>Nền tảng & Phân tích | 1. Dựng khung source code theo thiết kế modular.<br>2. Viết module `split.py` chia 90/10 Stratified Train/Val.<br>3. Benchmark nhanh các zero-shot ASR models (PhoWhisper vs Whisper-large-v3) trên `train.jsonl`.<br>4. Hoàn thiện `LocalScorer` để đo lường tự động. | **P0** | - Bộ khung code `lifemate/`<br>- File `val_split.jsonl` chuẩn<br>- Báo cáo WER ban đầu của ASR |
| **Tuần 2 (05/11 - 11/11)**<br>Baseline Pipeline & ReAct | 1. Xây dựng prompt template Function Calling chuẩn cho Qwen2.5-3B.<br>2. Hoàn thiện `ToolGuardrail` và `ReActController`.<br>3. Chạy Baseline End-to-End đầu tiên (ASR Zero-shot + Qwen2.5-3B Zero-shot).<br>4. Tạo file submission đầu tiên trên `public_test.jsonl` và nộp thử lên Leaderboard. | **P0** | - Submission v1 trên Leaderboard<br>- Baseline State Match Rate trên tập Val |
| **Tuần 3 (12/11 - 16/11)**<br>Fine-tuning & Optimization | 1. Fine-tune PhoWhisper với LoRA trên toàn bộ dữ liệu âm thanh cuộc thi.<br>2. Sinh dataset ChatML và huấn luyện QLoRA cho Qwen2.5-3B với kỹ thuật Masked Loss & ASR Noise Augmentation.<br>3. Tối ưu hóa Guardrail xử lý các case khó của `x-one-of` và `x-ref`.<br>4. Đóng gói Docker container thử nghiệm và test chạy offline.<br>5. **Nộp toàn bộ Code + Weights trước 23:59 ngày 16/11.** | **P0 (Critical)** | - Checkpoints ASR + LLM tốt nhất<br>- Docker image hoàn chỉnh<br>- Hoàn tất nộp bài Giai đoạn 1 |
| **Tuần 4 & 5 (17/11 - 23/11)**<br>Private Test & Final Push | 1. Tiếp nhận dữ liệu Private Test vào 9:00 ngày 20/11.<br>2. Chạy inference song song trên Docker container với GPU.<br>3. Kiểm tra tính toàn vẹn của `submission.jsonl` bằng `zaic_tools --validate`.<br>4. **Nộp file kết quả Private Test trước 23:59 ngày 23/11.** | **P0 (Final)** | - File `submission.jsonl` Private Test chính thức |

---

## 6. KẾT LUẬN & BƯỚC ĐI TIẾP THEO

Tài liệu này xác lập nền móng kỹ thuật toàn diện cho bảng đấu LifeMate. Kiến trúc kết hợp giữa **PhoWhisper + Qwen2.5-3B ReAct + Active Guardrail** là giải pháp tối ưu nhất về cả độ chính xác nhận diện tiếng Việt, khả năng suy luận công cụ đa bước và tuân thủ tuyệt đối giới hạn tài nguyên tính toán của Ban tổ chức.

**Hành động tức thời:**
1. Trình tài liệu này để Trưởng nhóm / User phê duyệt kiến trúc.
2. Ngay sau khi được phê duyệt, tiến hành sinh khung mã nguồn (codebase scaffolding) và dựng bộ nạp dữ liệu `data_processing/`.

