# Fine-tune ASR — PhoWhisper-large LoRA

[`scripts/main.py`](scripts/main.py) chuyển quy trình từ `finetune_phowhisper_large_lora.ipynb` thành một lệnh: kiểm tra audio → chia dữ liệu → train → chọn checkpoint theo validation WER → lưu/nạp lại adapter → chấm test trước–sau → đóng gói kết quả. Script chạy độc lập, không cần mở notebook.

## Tổ chức mã nguồn

File `scripts/main.py` chỉ là entry point. Logic nằm trong package `phowhisper/`, tách theo chức năng:

| Module | Chức năng |
|---|---|
| `phowhisper/config.py` | Tham số CLI, giá trị mặc định và kiểm tra cấu hình |
| `phowhisper/data.py` | Đọc JSONL/WAV, speaker metadata, lưu manifest; trả về `PreparedData` |
| `phowhisper/splitting.py` | Gom nhóm hội thoại/speaker/PCM, khử trùng, chia và audit split |
| `phowhisper/dataset.py` | Dataset xử lý audio theo mẫu và collator padding label |
| `phowhisper/metrics.py` | Chuẩn hóa tiếng Việt, WER/CER và callback metrics cho Trainer |
| `phowhisper/modeling.py` | Resolve commit base, processor, quantization, decoding và gắn LoRA |
| `phowhisper/trainer.py` | Kiểm tra gradient, cấu hình/resume Trainer, lưu best adapter |
| `phowhisper/evaluation.py` | Nạp lại adapter, inference trước–sau và xuất báo cáo |
| `phowhisper/artifacts.py` | Hash file/adapter, khôi phục run và đóng gói ZIP |
| `phowhisper/pipeline.py` | Điều phối các bước và quản lý thời điểm nạp thư viện GPU |

Các module trao đổi qua tham số, `PreparedData` và `ModelContext`; không phụ thuộc biến toàn cục của notebook. `--help` và `--prepare-only` không import các module GPU. Khi chuyển mã sang Kaggle/máy khác, giữ cả `scripts/` và `phowhisper/` trong thư mục `training/asr/`.

## Cài đặt

Dùng Python 3.10+ và một GPU NVIDIA/CUDA để train. Nên cài trong môi trường riêng với LLM/Unsloth; các phiên bản dưới đây giữ theo notebook. Từ thư mục project:

```bash
python -m pip install -r training/asr/requirements.txt
python training/asr/scripts/main.py --help
```

Lần đầu train cần Internet để tải model `vinai/PhoWhisper-large`. `--help` và `--prepare-only` chỉ dùng thư viện chuẩn Python, không cần cài stack GPU. Trên máy dùng tên lệnh `python3`, thay `python` bằng `python3`.

## Dữ liệu và chạy một lệnh

Theo README bộ dữ liệu, nguồn là `train/train.jsonl`: script lấy `audio`/`audio_text` của lượt hiện tại và các lượt `user` trong `history`. Đường dẫn audio được nối với thư mục chứa JSONL. Không dùng label tool hay `public_test` để train/chọn checkpoint.

Với dữ liệu hiện nằm trong `train+public_test/`, kiểm tra dữ liệu trước:

```bash
python training/asr/scripts/main.py --data-root "train+public_test" --prepare-only
```

Chạy thử toàn bộ pipeline với 3 optimizer steps, tối đa 32/8/8 clip train/validation/test:

```bash
python training/asr/scripts/main.py --data-root "train+public_test" --smoke-test
```

Train đầy đủ, tự chạy đánh giá và xuất adapter sau khi train xong:

```bash
python training/asr/scripts/main.py --data-root "train+public_test"
```

Có thể truyền `--data-root` là thư mục gói dữ liệu, thư mục `train/`, hoặc đường dẫn `train.jsonl`. Nếu bỏ tùy chọn này, script dùng `ZAIC_DATA_ROOT`, sau đó tự tìm dưới `/kaggle/input` khi có thư mục đó, hoặc dưới project root. Nếu tìm thấy nhiều nguồn, cần truyền đường dẫn cụ thể. Các manifest đã sinh trong output không được coi là nguồn train.

Ví dụ PowerShell khi dữ liệu ở nơi khác:

```powershell
$env:ZAIC_DATA_ROOT = 'D:\path\to\train+public_test'
python training/asr/scripts/main.py
```

Trên Kaggle, cài requirements một lần rồi chạy cùng script; có thể bỏ `--data-root` nếu chỉ có một nguồn `train.jsonl` trong Input. Output mặc định ở `/kaggle/working/phowhisper_lora/`; ngoài Kaggle là `training/asr/outputs/`.

## Cấu hình

| Tham số | Mặc định / chức năng |
|---|---|
| `--run-name` | `phowhisper-large-lora-v1`; smoke thêm hậu tố `-smoke` |
| `--output-root` | Thư mục chứa các run |
| `--base-model`, `--base-revision` | `vinai/PhoWhisper-large`, `main`; lưu commit đã resolve cho run |
| `--gpu` | `0`; nếu đã đặt `CUDA_VISIBLE_DEVICES` thì giữ biến môi trường đó |
| `--no-use-4bit` | Tắt NF4 khi GPU đủ bộ nhớ; mặc định bật 4-bit |
| `--batch-size`, `--gradient-accumulation-steps` | `1`, `16` |
| `--learning-rate`, `--epochs` | `1e-4`, `3` |
| `--lora-r`, `--lora-alpha`, `--lora-dropout` | `16`, `32`, `0.05`; target `q_proj`/`v_proj` encoder và decoder |
| `--seed`, `--split-ratios` | `42`, `0.8 0.1 0.1` |
| `--speaker-map` | CSV/JSONL chứa `speaker_id` và `clip_id` hoặc `audio` |
| `--audio-root` | Gốc để nối đường dẫn audio khi WAV nằm khác thư mục JSONL |
| `--no-deduplicate-audio` | Giữ clip trùng PCM, nhưng vẫn buộc chúng vào cùng split |
| `--max-new-tokens` | `440`; phạm vi `1..440` |
| `--zip-full-run` | Xuất thêm ZIP chứa optimizer/checkpoints để chuyển máy |

Audio phải WAV PCM mono 16 kHz/16-bit, dài trên 0 và không quá 30 giây. Script báo lỗi thay vì tự cắt audio hoặc transcript quá dài. Chia khoảng 80/10/10 theo nhóm liên thông hội thoại, speaker đã biết và PCM trùng; ưu tiên giữ nhóm hơn tỷ lệ chính xác. Không suy diễn speaker từ tên người dùng. Nếu thiếu speaker metadata, split chưa bảo đảm tách biệt người nói.

## Resume

Chạy lại cùng lệnh và cùng tên run để tự tìm checkpoint cuối trong `checkpoints/`. Split và cấu hình phải khớp run cũ; khi đổi dữ liệu hoặc hyperparameter, dùng `--run-name` mới. Có thể chỉ định `--resume-from-checkpoint` tới một checkpoint thuộc chính run đó.

Để chuyển máy/phiên Kaggle, giữ **cả thư mục run**, rồi copy bằng:

```bash
python training/asr/scripts/main.py --data-root "/path/to/data" --restore-run-dir "/path/to/old-run" --run-name phowhisper-restored
```

Đích phải chưa tồn tại; truyền lại các hyperparameter của run cũ nếu khác mặc định, gồm `--smoke-test` nếu khôi phục run smoke. Các đường dẫn best checkpoint trong trạng thái Trainer được đổi sang vị trí mới. Ở những lần chạy tiếp theo, bỏ `--restore-run-dir`. Adapter đơn lẻ không có optimizer state để resume training.

## Kết quả

Trong `<output-root>/<run-name>/`:

- `split_manifest.json`, `train.jsonl`, `validation.jsonl`, `test.jsonl`: split và audit dữ liệu.
- `training_config.json`, `training_result.json`: commit base, cấu hình, best validation WER và trạng thái train.
- `checkpoints/`: checkpoint Trainer; chọn best bằng validation WER, early stopping patience 2.
- `adapter/`: LoRA adapter và processor; cần base model đúng commit để inference.
- `phowhisper-large-before.jsonl`, `phowhisper-large-after.jsonl`: dự đoán test trước/sau LoRA.
- `before_after.json`, `before_after.csv`, `per_clip_comparison.csv`: WER/CER tổng thể, current/history và so sánh từng clip.

ZIP `<run-name>-adapter-results.zip` nằm cạnh thư mục run, không chứa trọng số base hoặc `checkpoints/`. `--zip-full-run` tạo thêm `<run-name>-resume.zip`. Output chứa transcript/metadata cuộc thi: giữ riêng, không commit hay phân phối công khai.

Baseline trước–sau dùng cùng base, precision, quantization và greedy decoding. WER/CER chuẩn hóa NFC, chữ thường, dấu câu thành khoảng trắng; giữ dấu tiếng Việt và chữ số. CER bỏ khoảng trắng; WER tính tổng lỗi / tổng số từ. Test chỉ chạy sau khi chọn adapter bằng validation. Theo notebook, bộ train đã từng được xem qua benchmark nên test tách ra là đánh giá nội bộ, không phải test hoàn toàn chưa từng xem. Điểm smoke chỉ dùng kiểm tra pipeline.

Notebook gốc được giữ làm tài liệu tham khảo. Adapter runtime vẫn ở `src/lifemate/adapters/asr/`; trainer không tự thay model đang dùng bởi agent.
