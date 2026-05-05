import logging
import threading

from faster_whisper import WhisperModel
import ctranslate2 # Import ctranslate2 to check for CUDA availability
from resemblyzer import VoiceEncoder

from .config import device, compute_type, MODELS_DIR

logger = logging.getLogger(__name__)

model: WhisperModel | None = None
current_model_size: str | None = None
voice_encoder: VoiceEncoder | None = None
_model_lock = threading.Lock()
_encoder_lock = threading.Lock()
pause_event = threading.Event()
stop_event = threading.Event()


def load_model(model_size: str) -> None:
    global model, current_model_size
    if current_model_size == model_size and model is not None:
        return
    with _model_lock:
        if current_model_size == model_size and model is not None:
            return
        try:
            logger.info(f"Model yükleniyor: {model_size} (Cihaz: {device}). İlk indirme ise birkaç dakika sürebilir...")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Determine the actual device and compute type based on system capabilities.
            # This is crucial to avoid RuntimeError if CUDA is not truly available,
            # especially when a CPU-only PyTorch is installed as indicated by uv.lock.
            actual_device = device
            actual_compute_type = compute_type

            try:
                model = WhisperModel(
                    f"Systran/faster-whisper-{model_size}",
                    device=actual_device,
                    compute_type=actual_compute_type,
                    download_root=str(MODELS_DIR),
                )
            except RuntimeError as e:
                # Handle cases where CUDA hardware is present but libraries like libcublas are missing
                if actual_device == "cuda":
                    logger.warning(f"CUDA hatası alındı: {e}. CPU üzerinden devam ediliyor.")
                    actual_device = "cpu"
                    actual_compute_type = "int8"
                    model = WhisperModel(
                        f"Systran/faster-whisper-{model_size}",
                        device=actual_device,
                        compute_type=actual_compute_type,
                        download_root=str(MODELS_DIR),
                    )
                else:
                    raise

            current_model_size = model_size
            logger.info("Model başarıyla yüklendi.")
        except Exception as e:
            logger.exception(f"Model yüklenirken hata oluştu: {model_size}")
            model = None
            current_model_size = None


def get_voice_encoder() -> VoiceEncoder:
    global voice_encoder
    if voice_encoder is None:
        with _encoder_lock:
            if voice_encoder is None:
                logger.info("Ses encoder (Resemblyzer) yükleniyor...")
                voice_encoder = VoiceEncoder()
                logger.info("Ses encoder başarıyla yüklendi.")
    return voice_encoder
