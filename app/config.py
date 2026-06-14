import logging
import os
import sys
import warnings
from enum import StrEnum
from pathlib import Path

import ctranslate2
import torch  # Import torch to check for CUDA availability
from pydantic import BaseModel, ConfigDict, Field

# Suppress Starlette/Gradio deprecation warnings regarding HTTP status codes immediately
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")


class ModelSize(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"

    @classmethod
    def values(cls) -> list[str]:
        return [cls.SMALL, cls.MEDIUM, cls.LARGE_V3]


class GlobalConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_model_size: ModelSize = ModelSize.MEDIUM
    # Optimal for CPU multi-core utilization
    diarization_batch_size: int = Field(32, gt=0)
    encoder_std_model: str = Field(default="speechbrain/spkrec-ecapa-voxceleb")
    encoder_fast_model: str = Field(default="speechbrain/spkrec-xvect-voxceleb")
    sample_rate: int = Field(16000, gt=0)
    paragraph_pause: float = Field(1.5, ge=0.1)
    transcription_timeout: int = Field(3600, gt=0)
    cache_base_dir: Path = Field(default=Path(os.getenv("CACHE_DIR", "cache")))
    # Models can be large (~5GB+ total), so allowing an independent mount point is recommended
    models_dir: Path = Field(default=Path(os.getenv("MODELS_DIR", "cache/models")))
    default_cache_size_mb: int = 1000
    session_expiry_hours: int = 24
    whisper_model_sizes_mb: dict[str, int] = Field(default={"small": 600, "medium": 1600, "large-v3": 3200})
    device: str = Field(default="cpu")
    compute_type: str = Field(default="int8_float32")


settings = GlobalConfig()

EMBEDDING_CACHE_DIR = settings.cache_base_dir / "embeddings"
TRANSCRIPT_CACHE_DIR = settings.cache_base_dir / "transcriptions"


def update_settings(model_size: ModelSize, sample_rate: int, paragraph_pause: float, transcription_timeout: int):
    """Updates global settings at runtime based on command-line arguments."""
    settings.default_model_size = model_size
    settings.sample_rate = sample_rate
    settings.paragraph_pause = paragraph_pause
    settings.transcription_timeout = transcription_timeout


SUPPORTED_LANGUAGES = [
    ("Türkçe", "tr"),
    ("English", "en"),
    ("Deutsch", "de"),
    ("Français", "fr"),
    ("Español", "es"),
    ("Italiano", "it"),
    ("Otomatik Algıla", None),
]


class WatchdogConfig(BaseModel):
    # No-progress grace period before declaring a genuine native hang. Must sit
    # comfortably above worst-case first-segment latency (VAD pass + first decode
    # at beam_size=5 across all temperatures on CPU/large-v3), which can run for
    # minutes on long files. The old 120s tripped on healthy long jobs.
    check_interval: int = Field(30, gt=0)
    timeout: int = Field(600, gt=0)
    exit_code: int = 42


watchdog_settings = WatchdogConfig()


class RetryConfig(BaseModel):
    max_attempts: int = 3
    initial_delay: float = 2.0


retry_settings = RetryConfig()


def setup_logging():
    """Configures the global logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    # Reduce logging verbosity for httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)


def is_cuda_available() -> bool:
    """
    Verifies if CUDA is actually usable by checking device count and
    attempting a minimal CTranslate2 operation to ensure libraries like
    libcublas and libcudnn are correctly linked.
    """
    if settings.device != "cuda":
        return False
    try:
        # We check both CTranslate2 and Torch because transcription and diarization
        # use different backends. Both must see the GPU to ensure a consistent pipeline.
        return ctranslate2.get_cuda_device_count() > 0 and torch.cuda.is_available()
    except Exception:
        return False


settings.device = "cuda" if is_cuda_available() else "cpu"
settings.compute_type = "float16" if settings.device == "cuda" else "int8_float32"
