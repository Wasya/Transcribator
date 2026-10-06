"""Диагностика окружения для WhisperX: python check_env.py (результат можно целиком присылать в баг-репорт)."""
import importlib.metadata as md
import platform
import sys


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "НЕ УСТАНОВЛЕН"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Python {sys.version.split()[0]} / {platform.platform()}")
    for pkg in ("deepgram-sdk", "PyAudioWPatch", "python-dotenv", "whisperx", "faster-whisper",
                "lameenc", "ctranslate2", "torch", "torchaudio", "pyannote.audio", "numpy"):
        print(f"{pkg:16} {version(pkg)}")
    try:
        import torch

        print(f"CUDA доступна: {torch.cuda.is_available()}  (torch собран для CUDA {torch.version.cuda})")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 2**20} МБ")
    except ImportError:
        print("torch не установлен")
    try:
        import ctranslate2

        print(f"ctranslate2 видит CUDA-устройств: {ctranslate2.get_cuda_device_count()}")
    except Exception as exc:  # noqa: BLE001
        print(f"ctranslate2: {exc}")
    from dotenv import load_dotenv
    import os

    load_dotenv()
    for name in ("DEEPGRAM_API_KEY", "HF_TOKEN"):
        print(f"{name}: {'задан' if os.environ.get(name, '').strip() else 'не задан'}")


if __name__ == "__main__":
    main()
