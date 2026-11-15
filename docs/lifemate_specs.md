# Zalo AI Challenge 2026 — LifeMate (Voice Agents) Technical Specs

> **Source:** Trích xuất trực tiếp từ cổng thông tin https://challenge.zalo.ai/portal/lifemate (đầy đủ các tab: Overview, Description, Evaluation, Data, Rule, Submission Guidance) + mã nguồn `zaic_tools` và bộ dev-kit local.  
> **Thời điểm đối chứng & cập nhật:** 29/10/2026  
> **Phiên bản catalog:** Schema 2.7.0  

---

## 1. TỔNG QUAN & MÔ TẢ BÀI TOÁN (OVERVIEW & DESCRIPTION)

### 1.1 Mục tiêu Bài toán
Xây dựng một AI Agent điều khiển bằng giọng nói (Voice-Driven Multi-step Agentic AI) có khả năng chuyển yêu cầu bằng tiếng Việt nói đời thường thành chuỗi hành động tương tác chính xác với các ứng dụng số (smart home, banking, ride-hailing, food delivery, email, v.v.).

Agent được đánh giá trên **kết quả hành động thực tế** (đưa trạng thái các ứng dụng về đúng yêu cầu của người dùng), không đánh giá trên câu trả lời văn bản.

### 1.2 Cấu trúc Đầu vào (Input) của mỗi mẫu
1. **Audio yêu cầu cuối (`audio`):** File `.wav` chứa giọng nói tiếng Việt của người dùng (có thể pha trộn từ tiếng Anh thông dụng, giọng 3 miền Bắc - Trung - Nam, giọng thật hoặc tổng hợp). Ví dụ: `audio/<id>/<id>.wav`.
2. **Ngữ cảnh người dùng (`context`):**
   - `current_time`: Thời điểm hiện tại chuẩn ISO-8601 (ví dụ `2026-11-28T07:10:00+07:00`). Dùng để phân giải các từ thời gian tương đối như "mai", "tối nay", "3 giờ chiều".
   - `timezone`: Múi giờ (ví dụ `Asia/Ho_Chi_Minh`).
   - `location`: Tọa độ vị trí hiện tại (`lat`, `lng`).
   - `user_profile`: Tên, địa chỉ nhà, địa chỉ cơ quan, thành phố hiện tại.
   - `installed_apps`: Danh sách các ứng dụng đã cài và schema các tool được cấp phép cho mẫu đó (`{app_name: {"tools": [tool_names]}}`). **Agent chỉ được phép gọi các tool nằm trong danh sách này**.
3. **Lịch sử hội thoại (`history`):** Các lượt thoại trước đó (nếu có). Gồm audio của user (`_u1.wav`, `_u2.wav`) và phản hồi/tool_calls của assistant. Các hành động trong history đã được áp dụng sẵn vào trạng thái ban đầu của môi trường.
4. **Môi trường giả lập (`ToolEnv`):** Lưu trữ dữ liệu app của người dùng tại `<split>_environment/<id>/state.json`. Agent **tuyệt đối không được đọc trực tiếp file này** mà chỉ tương tác qua `zaic_tools`.

### 1.3 Cấu trúc Đầu ra (Output)
Mỗi dòng trong file `submission.jsonl` (UTF-8 JSON Lines):
```json
{
  "id": "public_0001",
  "tool_calls": [
    {"name": "smart_home.get_status", "arguments": {}},
    {"name": "smart_home.climate_set", "arguments": {"device_id": "ac_living_room", "mode": "cool", "temperature": 26}},
    {"name": "ride.book", "arguments": {"from": "227 Nguyễn Văn Cừ, Quận 5, TP.HCM", "to": "123 Trần Hưng Đạo, Quận 5, TP.HCM"}}
  ],
  "state_hash": "9f2c4e7a1b8d3f60..."
}
```

### 1.4 Ba Kịch bản Xử lý Cốt lõi
- **Kịch bản 1: Thực thi hành động (Execute Actions):** Gọi chuỗi tools (Read $\rightarrow$ Write) để hoàn thành nhiệm vụ và đưa môi trường về trạng thái mong muốn.
- **Kịch bản 2: Hỏi thêm thông tin (Request Missing Info):** Khi yêu cầu bị thiếu tham số bắt buộc không thể tự tra cứu, Agent gọi:
  ```json
  {"name": "sys.ask_missing_info", "arguments": {"tool_name": "<app>.<tool>", "params": ["<param_name>"]}}
  ```
  *(Lưu ý: Có thể gọi sau khi đã thực hiện một số tool đọc trước đó).*
- **Kịch bản 3: Ngoài phạm vi (Out-of-scope Requests):** Nếu không có ứng dụng nào đã cài có thể xử lý được yêu cầu của người dùng $\rightarrow$ Trả về danh sách rỗng:
  ```json
  "tool_calls": []
  ```
  và `state_hash` là hash của trạng thái ban đầu (`env.get_state_hash()`).

---

## 2. CÔNG THỨC CHẤM ĐIỂM CHÍNH THỨC (EVALUATION METRICS & FORMULA)

Điểm số của từng sample được tính dựa trên 2 yếu tố: **State Match** (Trạng thái cuối của ứng dụng) và **Tool-call Match** (Độ chính xác của chuỗi lời gọi tool so với đáp án Gold).

### 2.1 Tiêu chí So khớp (Match Criteria)
1. **State Match:**
   - `state_hash` nộp phải **trùng khớp tuyệt đối 100%** với `gold state_hash`. Nếu sai khác $\rightarrow$ Sample nhận **0 điểm**.
   - Các trường văn bản tự do (`FREE_TEXT_KEYS`: `address`, `body`, `content`, `dropoff`, `expression`, `label`, `location`, `name`, `note`, `pickup`, `subject`, `title`) và các app tin tức/thời tiết (`LIVE_APPS`: `weather`, `gold_price`, `fuel_price`, `lottery`, `exchange_rate`) **không tham gia vào tính hash**.
   - `state_hash` bắt buộc tính từ `env.get_state_hash()`. BTC sẽ replay lại `tool_calls` bằng đúng bản `zaic_tools` để xác minh.
2. **Tool-call Match:**
   - Một call nộp được tính là khớp với một gold call khi:
     - Tên tool trùng khớp (`name`).
     - Mọi tham số không phải fuzzy (`non-fuzzy arguments`) phải giống hệt nhau.
     - Mọi tham số tự do (`fuzzy arguments` có đánh dấu `x-match: fuzzy`) phải đạt **token-F1 > 0.6** (sau khi lowercase và chuẩn hóa Unicode NFC).
   - Mỗi call nộp chỉ được ghép tối đa với 1 gold call; **thứ tự các call không bị ràng buộc**.
3. **Out-of-scope samples:**
   - Nếu đáp án gold là rỗng (`[]`): Sample đạt **1 điểm** nếu danh sách nộp cũng là `[]`; ngược lại nhận **0 điểm**.
4. **Missing Information (`sys.ask_missing_info`):** Được chấm điểm và so khớp như một tool thông thường.

### 2.2 Công thức Toán học Chi tiết (Từ Portal Evaluation)

$$
\text{Score}_i = \begin{cases}
0, & \text{nếu } \text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}} \text{ hoặc } M = 0 \\
\frac{2 \cdot P \cdot R}{P + R}, & \text{ngược lại (khi } \text{hash}_{\text{sys}} = \text{hash}_{\text{gold}} \text{ và } M > 0\text{)}
\end{cases}
$$

Trong đó:
- $M$: Số lượng tool calls khớp (Matched calls).
- $|G|$: Tổng số lượng tool calls trong đáp án Gold.
- $U$: Số lượng tool calls không khớp do hệ thống sinh ra (Unmatched submitted calls, **bao gồm cả các call bị từ chối do lỗi schema**).
- Precision: $P = \frac{M}{M + U}$
- Recall: $R = \frac{M}{|G|}$

**Điểm tổng kết trên Leaderboard (Thang điểm 100):**

$$
\text{Final Score} = \frac{100}{N} \sum_{i=1}^{N} \text{Score}_i
$$

*(Với $N$ là tổng số mẫu trong tập test).*

---

## 3. CÁC QUY ĐỊNH KỸ THUẬT & RÀNG BUỘC PHẦN CỨNG (TECHNICAL RULES & HARDWARE)

*(Trích xuất trực tiếp từ tab Rule của BTC — Rất quan trọng!)*

### 3.1 Giới hạn Kích thước Mô hình (Model Size Limit - P0 Bắt buộc)
- **Toàn bộ hệ thống (ASR + LLM + mọi thành phần khác):** Phải có **dưới 6 tỷ tham số (< 6B parameters in total)**.
- **Riêng mô hình LLM:** Phải có **dưới 5 tỷ tham số (< 5B parameters)** và context length tối đa **16,384 tokens (16K)**.
- ⚠️ **LƯU Ý CỰC KỲ QUAN TRỌNG:** Các model 7B (như `Qwen2.5-7B`) **sẽ bị loại vì vi phạm trần 5B**. Do đó, mô hình LLM bắt buộc phải là dòng **< 5B**, lựa chọn tối ưu nhất là **`Qwen/Qwen2.5-3B-Instruct`** (3.09B params) hoặc **`Qwen2.5-1.5B-Instruct`**.

### 3.2 Cấu hình Phần cứng Máy chủ Chấm thi (Scoring Hardware)
BTC sẽ chạy container Docker của các đội thi trên phần cứng:
- **GPU:** 1 x NVIDIA A30 (24 GB HBM2 memory).
- **CPU:** 16-core Intel Xeon Gold 6442Y CPU.
- **RAM:** 64 GB RAM.

### 3.3 Ràng buộc Thời gian & Môi trường Chạy (Runtime Constraints)
- **Thời gian xử lý tối đa:** **30 giây cho mỗi sample**. Sample nào chạy quá 30 giây hoặc bị crash/lỗi thực thi sẽ bị chấm **0 điểm**.
- **Chế độ Offline:** Hoàn toàn offline, **tuyệt đối không gọi External API/Internet** trong quá trình inference.
- **Tính tái lập (Reproducibility):** Phải cố định toàn bộ random seed (Python, NumPy, PyTorch), dùng deterministic decoding (greedy decoding hoặc temperature = 0). Nếu chạy lại có kết quả khác, BTC sẽ chạy 5 lần và lấy điểm trung bình.

### 3.4 Quy định Truy cập Dữ liệu & Môi trường
- **Nghiêm cấm đọc trực tiếp `state.json`:** Mọi thao tác đọc/ghi dữ liệu app phải đi qua thư viện `zaic_tools`.
- **Duy nhất 1 `ToolEnv` mỗi sample:** Không được tạo lại phiên để "thử lại" từ state ban đầu. Mọi call thực hiện đều phải có mặt trong kết quả nộp bài.

### 3.5 Luật Phân định Thắng Thua (Tie-Breaking)
Nếu hai đội bằng điểm nhau trên Leaderboard:
1. BTC sẽ tính lại điểm F1 trên `tool_calls` bỏ qua điều kiện `state_hash`.
2. Nếu vẫn hòa, đội có **thời gian xử lý trung bình mỗi sample thấp hơn** sẽ xếp trên.

---

## 4. QUY ĐỊNH NỘP BÀI (SUBMISSION GUIDELINES)

### 4.1 Vòng 1 — Public Test (26/10 – 16/11)
- Nộp file `submission.jsonl` trực tiếp trên portal.
- Giới hạn: **Tối đa 5 lần nộp/ngày** (hạn ngạch reset vào cuối ngày). Điểm số hiển thị trên Public Leaderboard.
- **Hạn nộp Final Solution (Code + Model + Data):** **23:59 ngày 16/11/2026**.
  - Nộp dưới dạng gói Docker qua form "Solution Submission".
  - Do file Docker thường rất lớn, đội thi cung cấp mã Checksum và link Google Drive công khai để BTC tải.
  - Cho phép nộp tối đa 2 lần (BTC chỉ chấm lần nộp cuối cùng).

### 4.2 Cấu trúc Chấm thi Docker của BTC
BTC sẽ mount thư mục test vào container và thu thập kết quả như sau:
- Thư mục test private được mount vào: `/data` (có cấu trúc y hệt `public_test`).
- File kết quả được ghi vào: `result/submission.jsonl`.

### 4.3 Vòng 2 — Private Test (20/11 – 23/11)
- 09:00 ngày 20/11: BTC phát hành tập `private_test`.
- Đội thi chạy solution đã đóng băng trên tập này và nộp file `submission.jsonl`.
- Hạn chót: **23:59 ngày 23/11/2026**.
- BTC chạy lại Docker image để xác minh kết quả. Công bố Top 2 vào ngày 02/12.

---

## 5. ĐỐI CHỨNG VÀ KẾT LUẬN THAY ĐỔI CHIẾN LƯỢC

Qua việc đối chứng trực tiếp với tab **Rule** và **Evaluation** trên portal, ta phát hiện **3 thay đổi mang tính sống còn**:

1. **Thay đổi Model LLM: Chuyển từ Qwen2.5-7B sang Qwen2.5-3B-Instruct**
   - *Lý do:* BTC giới hạn LLM < 5B và cả hệ thống < 6B. Dùng 7B sẽ bị loại ngay từ vòng duyệt Docker.
   - *Cấu hình mới:* `PhoWhisper` (~0.5B - 1.5B) + `Qwen2.5-3B-Instruct` (3.09B) = **~3.6B - 4.6B parameters < 6B limit** (Hoàn toàn hợp lệ!).
2. **Cập nhật Công thức Scorer nội bộ (`LocalScorer`):**
   - Phải cài đặt chính xác điều kiện: $\text{Score} = 0$ nếu $\text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}}$ hoặc $M = 0$.
   - Ngưỡng so khớp fuzzy arguments: `token-F1 > 0.6`.
3. **Chuẩn hóa Docker Output Path:**
   - Input: `/data` (thay vì path tương đối tuỳ ý).
   - Output: `result/submission.jsonl`.
