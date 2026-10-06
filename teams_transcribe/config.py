import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    deepgram_api_key: Optional[str]
    hf_token: Optional[str]
    whisper_model: Optional[str]  # None = pick by device
    whisper_device: str  # "auto" | "cuda" | "cpu"
    whisper_compute_type: Optional[str]  # None = pick by device
    whisper_use_context: bool = False  # pass previous text to Whisper as a prompt (experimental)


def load_settings() -> Settings:
    def env(name: str) -> Optional[str]:
        value = os.environ.get(name, "").strip()
        return value or None

    return Settings(
        deepgram_api_key=env("DEEPGRAM_API_KEY"),
        hf_token=env("HF_TOKEN"),
        whisper_model=env("WHISPER_MODEL"),
        whisper_device=(env("WHISPER_DEVICE") or "auto").lower(),
        whisper_compute_type=env("WHISPER_COMPUTE_TYPE"),
        whisper_use_context=(env("WHISPER_USE_CONTEXT") or "").lower() in ("1", "true", "yes", "on"),
    )


def resolve_whisper_runtime(settings: Settings, model_override: Optional[str] = None) -> tuple[str, str, str]:
    """Return (model, device, compute_type). 'auto' device means CUDA when available."""
    device = settings.whisper_device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    model = model_override or settings.whisper_model or ("large-v3-turbo" if device == "cuda" else "small")
    compute_type = settings.whisper_compute_type or ("float16" if device == "cuda" else "int8")
    return model, device, compute_type
