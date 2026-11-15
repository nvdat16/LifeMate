# Dữ liệu huấn luyện

Thư mục `sft/` chứa hai phiên bản của cùng hướng dữ liệu huấn luyện. Hãy chọn đúng một phiên bản khi fine-tune; không ghép hai phiên bản với nhau.

## Các file trong `sft/`

| File | Nội dung | Dùng khi nào |
|---|---|---|
| `train_chatml.jsonl` | 1.450 mẫu train, định dạng `messages` | Bộ SFT chuẩn; `training/llm/scripts/train_sft.py` đang đọc file này |
| `val_chatml.jsonl` | 76 mẫu validation cho bộ chuẩn | Validation tương ứng với bộ chuẩn |
| `train_sharegpt.jsonl` | Cùng nội dung với `train_chatml.jsonl`, đổi cấu trúc sang `conversations` | Chỉ cần cho công cụ huấn luyện nhận ShareGPT, như LLaMA-Factory |
| `val_sharegpt.jsonl` | Cùng nội dung với `val_chatml.jsonl`, đổi cấu trúc sang `conversations` | Validation định dạng ShareGPT |
| `harmonized_train_chatml.jsonl` | 1.450 mẫu của biến thể harmonized | Thử nghiệm riêng với dữ liệu đã chuẩn hóa |
| `harmonized_val_chatml.jsonl` | 76 mẫu validation của biến thể harmonized | Validation cho biến thể harmonized |
| `dataset_info.json` | Metadata và thống kê của bộ chuẩn | Tra cứu nguồn file và độ dài chuỗi |
| `harmonized_dataset_info.json` | Metadata của biến thể harmonized | Tra cứu thống kê và metric đã ghi cho biến thể này |

## Chọn phiên bản nào?

- Muốn chạy script huấn luyện hiện tại: dùng `train_chatml.jsonl` và `val_chatml.jsonl`.
- Muốn thử harmonization: dùng cặp `harmonized_*_chatml.jsonl`. Script huấn luyện hiện tại **không tự chuyển** sang cặp này.
- Dùng ChatML thì không cần các file ShareGPT; giữ chúng nếu cần tương thích với công cụ khác.

## Lưu ý khi so sánh kết quả

- Hai bộ train/validation đều không có ID giao nhau nội bộ; mỗi bộ gồm 1.450 train và 76 validation.
- Danh sách ID ở train/validation của bản chuẩn và bản harmonized **không giống nhau**. Muốn so sánh tác động của harmonization một cách công bằng, cần đánh giá cả hai trên cùng một tập ID validation.
- `harmonized_dataset_info.json` ghi metric tính trên toàn bộ 1.526 mẫu harmonized. Không xem các metric đó là kết quả validation độc lập.
- `dataset_info.json` ghi `recommended_max_seq_length: 10240`, trong khi thống kê của chính bộ này ghi mẫu dài nhất 10.506 tokens. Cấu hình 10.240 có thể cắt mẫu dài; hãy kiểm tra bằng tokenizer của model trước khi huấn luyện. Script hiện đặt mặc định `max_seq_length=11000`.
- Dataset chỉ có 1.526 mẫu đã chọn từ 1.545 train gốc. Muốn biết lý do 19 mẫu không có trong bộ SFT, truy lại dữ liệu trong `training/llm/data/distillation/` và script tạo dataset.

## Định dạng một dòng

ChatML lưu hội thoại trong `messages`, ví dụ `{ "id": "...", "messages": [...] }`. ShareGPT lưu cùng hội thoại trong `conversations`. Hai định dạng này là hai cách đóng gói, không phải hai tập dữ liệu độc lập.
