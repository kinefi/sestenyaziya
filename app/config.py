import logging
import sys
from pathlib import Path
import ctranslate2
import torch # Import torch to check for CUDA availability
from enum import StrEnum


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

CACHE_BASE_DIR = Path("cache")
EMBEDDING_CACHE_DIR = CACHE_BASE_DIR / "embeddings"
TRANSCRIPT_CACHE_DIR = CACHE_BASE_DIR / "transcriptions"
MODELS_DIR = Path("models")
TEMP_EXPORT_DIR = CACHE_BASE_DIR / "temp_exports"
DEFAULT_CACHE_SIZE_MB = 1000

def setup_logging():
    """Configures the global logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )
    # Reduce logging verbosity for httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)


try:
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 and torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"
compute_type = "float16" if device == "cuda" else "int8"
