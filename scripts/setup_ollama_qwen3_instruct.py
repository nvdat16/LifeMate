"""
Complete all-in-one setup script for Ollama with Qwen3-4B-Instruct-2507 on Kaggle / Linux.

Handles:
1. Installing zstd & Ollama if not present.
2. Starting Ollama daemon in background.
3. Downloading Qwen3-4B-Instruct-2507 GGUF from Unsloth HuggingFace.
4. Creating the Modelfile and registering model with Ollama.
5. Verifying model response.
"""

import os
import sys
import time
import shutil
import urllib.request
import subprocess
from pathlib import Path

def is_ollama_server_ready():
    try:
        req = urllib.request.urlopen("http://localhost:11434/", timeout=2)
        return req.getcode() == 200
    except Exception:
        return False

def ensure_ollama_installed():
    print("🔍 [1/5] Kiểm tra cài đặt Ollama...")
    if shutil.which("ollama") is None:
        print("  -> Ollama chưa được cài đặt. Đang cài đặt zstd và Ollama...")
        # Cài đặt zstd (bắt buộc trên Ubuntu 24.04 / Noble)
        subprocess.run(["apt-get", "update", "-y"], check=False)
        subprocess.run(["apt-get", "install", "-y", "zstd", "curl"], check=False)
        # Cài đặt Ollama
        install_cmd = "curl -fsSL https://ollama.com/install.sh | sh"
        subprocess.run(install_cmd, shell=True, check=True)
        print("  -> ✅ Đã cài đặt Ollama thành công!")
    else:
        print("  -> ✅ Ollama đã được cài đặt sẵn.")

def ensure_ollama_running():
    print("\n🚀 [2/5] Khởi động tiến trình Ollama daemon...")
    if is_ollama_server_ready():
        print("  -> ✅ Ollama daemon đã đang chạy sẵn trên http://localhost:11434")
        return

    # Khởi động ngầm
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    
    # Chờ tối đa 15 giây để server sẵn sàng
    for i in range(15):
        time.sleep(1)
        if is_ollama_server_ready():
            print(f"  -> ✅ Ollama daemon đã sẵn sàng sau {i+1}s!")
            return
    print("  -> ⚠️ Cảnh báo: Chưa kiểm tra được kết nối http://localhost:11434, tiếp tục thử nghiệm...")

def download_and_create_model():
    print("\n⏳ [3/5] Tải model Qwen3-4B-Instruct-2507 GGUF (~2.37 GB)...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  -> Cài đặt huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import hf_hub_download

    instruct_model_path = hf_hub_download(
        repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        filename="Qwen3-4B-Instruct-2507-UD-Q4_K_XL.gguf"
    )
    print(f"  -> Đã tải xong model về: {instruct_model_path}")

    print("\n📝 [4/5] Tạo Modelfile và đăng ký model vào Ollama...")
    modelfile_content = f"""FROM {instruct_model_path}
PARAMETER temperature 0.0
PARAMETER num_ctx 16384
PARAMETER stop "<|im_end|>"
"""
    with open("Modelfile_instruct", "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    res = subprocess.run(["ollama", "create", "qwen3-4b-instruct", "-f", "Modelfile_instruct"], check=True)
    if res.returncode == 0:
        print("  -> ✅ Đã đăng ký thành công model 'qwen3-4b-instruct' vào Ollama!")

def verify_model():
    print("\n🧪 [5/5] Kiểm tra thử nghiệm model qwen3-4b-instruct...")
    res = subprocess.run(
        ["ollama", "run", "qwen3-4b-instruct", "Xin chào, hãy trả lời trong 5 từ."],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    print("  -> Output thử nghiệm:")
    print(f"     \"{res.stdout.strip()}\"")
    print("\n🎉 HOÀN TẤT CÀI ĐẶT! BẠN ĐÃ CÓ THỂ CHẠY INFERENCE NGAY!")

if __name__ == "__main__":
    ensure_ollama_installed()
    ensure_ollama_running()
    download_and_create_model()
    verify_model()

