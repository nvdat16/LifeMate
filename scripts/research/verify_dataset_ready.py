import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.research.research_paths import PROJECT_ROOT, DATA_ROOT, DISTILLATION_DIR, SFT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training" / "data_prep"))

import sys
from pathlib import Path

WORKSPACE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(DATA_ROOT / "public_test"))

from zaic_tools import ToolEnv
from lifemate.adapters.storage.jsonl_dataset_reader import JsonlDatasetReader
from lifemate.adapters.sandbox.zaic_env_adapter import ZaicSandboxEnvFactory

def verify():
    print("=" * 60)
    print("VERIFYING DATASET STRUCTURE & COMPATIBILITY FOR KAGGLE")
    print("=" * 60)

    # 1. Train reader
    print("\n[1/5] Kiểm tra đọc tập train...")
    reader = JsonlDatasetReader(DATA_ROOT)
    train_samples = list(reader.read_samples("train", limit=5))
    assert len(train_samples) == 5, "Không đọc được 5 mẫu train"
    print(f"  -> OK! Đọc thành công {len(train_samples)} mẫu. Mẫu 1: id={train_samples[0].id}")

    # 2. Public test reader
    print("\n[2/5] Kiểm tra đọc tập public_test...")
    test_samples = list(reader.read_samples("public_test", limit=5))
    assert len(test_samples) == 5, "Không đọc được 5 mẫu public_test"
    print(f"  -> OK! Đọc thành công {len(test_samples)} mẫu. Mẫu 1: id={test_samples[0].id}")

    # 3. Sandbox for train
    print("\n[3/5] Kiểm tra Sandbox ToolEnv cho train...")
    factory = ZaicSandboxEnvFactory(DATA_ROOT)
    sess_train = factory.create_session("train", "train_0001")
    hash_train = sess_train.get_state_hash()
    assert len(hash_train) == 64, "state_hash không đúng chuẩn sha256"
    print(f"  -> OK! Tạo phiên môi trường train thành công. Hash: {hash_train}")

    # 4. Sandbox for public_test
    print("\n[4/5] Kiểm tra Sandbox ToolEnv cho public_test...")
    sess_test = factory.create_session("public_test", test_samples[0].id)
    hash_test = sess_test.get_state_hash()
    assert len(hash_test) == 64, "state_hash không đúng chuẩn sha256"
    print(f"  -> OK! Tạo phiên môi trường test thành công. Hash: {hash_test}")

    # 5. Audio files
    print("\n[5/5] Kiểm tra file âm thanh (Audio files)...")
    train_audio_count = len(list((DATA_ROOT / "train" / "audio").rglob("*.wav")))
    test_audio_count = len(list((DATA_ROOT / "public_test" / "audio").rglob("*.wav")))
    print(f"  -> train/audio: {train_audio_count} file .wav")
    print(f"  -> public_test/audio: {test_audio_count} file .wav")

    # 6. Check size & cleanup
    train_size_mb = sum(f.stat().st_size for f in (DATA_ROOT / "train").rglob("*") if f.is_file()) / (1024 * 1024)
    test_size_mb = sum(f.stat().st_size for f in (DATA_ROOT / "public_test").rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\nTổng dung lượng:")
    print(f"  - train/: {train_size_mb:.2f} MB")
    print(f"  - public_test/: {test_size_mb:.2f} MB")
    print(f"  - Tổng cộng 2 thư mục: {train_size_mb + test_size_mb:.2f} MB (~{(train_size_mb + test_size_mb)/1024:.2f} GB)")

    print("\n" + "=" * 60)
    print(">>> KẾT QUẢ: 2 THƯ MỤC HOÀN TOÀN CHUẨN XÁC ĐỂ ĐẨY LÊN KAGGLE! <<<")
    print("=" * 60)

if __name__ == "__main__":
    verify()
