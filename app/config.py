import logging
import os
import sys
import warnings
from enum import StrEnum
from pathlib import Path

import ctranslate2
import torch  # Import torch to check for CUDA availability

# Suppress Starlette/Gradio deprecation warnings regarding HTTP status codes immediately
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")


class ModelSize(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"

    @classmethod
    def values(cls) -> list[str]:
        return [cls.SMALL, cls.MEDIUM, cls.LARGE_V3]


DEFAULT_MODEL_SIZE = ModelSize.MEDIUM
SAMPLE_RATE = 16000
PARAGRAPH_PAUSE = 1.5
TRANSCRIPTION_TIMEOUT = 3600  # 1 hour limit for a single file

SUPPORTED_LANGUAGES = [
    ("Türkçe", "tr"),
    ("English", "en"),
    ("Deutsch", "de"),
    ("Français", "fr"),
    ("Español", "es"),
    ("Italiano", "it"),
    ("Otomatik Algıla", None),
]

# Watchdog and Retry Settings
WATCHDOG_CHECK_INTERVAL = 30  # Seconds
# No-progress grace period before declaring a genuine native hang. Must sit
# comfortably above worst-case first-segment latency (VAD pass + first decode
# at beam_size=5 across all temperatures on CPU/large-v3), which can run for
# minutes on long files. The old 120s tripped on healthy long jobs.
WATCHDOG_TIMEOUT = 600  # Seconds with zero heartbeats before process restart
WATCHDOG_EXIT_CODE = 42  # Non-zero exit so the supervisor restarts the process
RETRY_MAX_ATTEMPTS = 3
RETRY_INITIAL_DELAY = 2.0  # Base delay for backoff
DIARIZATION_BATCH_SIZE = 32  # Optimal for CPU multi-core utilization
ENCODER_STD = "speechbrain/spkrec-ecapa-voxceleb"
ENCODER_FAST = "speechbrain/spkrec-xvect-voxceleb"

# Allow overriding the cache directory via environment variable for easier deployment
CACHE_BASE_DIR = Path(os.getenv("CACHE_DIR", "cache"))
EMBEDDING_CACHE_DIR = CACHE_BASE_DIR / "embeddings"
TRANSCRIPT_CACHE_DIR = CACHE_BASE_DIR / "transcriptions"
# Models can be large (~5GB+ total), so allowing an independent mount point is recommended
MODELS_DIR = Path(os.getenv("MODELS_DIR", CACHE_BASE_DIR / "models"))
TEMP_EXPORT_DIR = CACHE_BASE_DIR / "temp_exports"
DEFAULT_CACHE_SIZE_MB = 1000


def setup_logging():
    """Configures the global logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    # Reduce logging verbosity for httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)


try:
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 and torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"
compute_type = "float16" if device == "cuda" else "int8_float32"
