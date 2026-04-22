import ctranslate2
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

try:
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
except Exception:
    device = "cpu"
compute_type = "float16" if device == "cuda" else "int8"
