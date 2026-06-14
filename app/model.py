import logging
import threading

from faster_whisper import WhisperModel

from .cache_utils import get_free_disk_mb
from .config import settings
from .utils import retry

logger = logging.getLogger(__name__)

model: WhisperModel | None = None
current_model_size: str | None = None
_model_lock = threading.Lock()


@retry()
def _create_whisper_instance(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """
    Helper factory to create a WhisperModel instance.
    """
    return WhisperModel(
        f"Systran/faster-whisper-{model_size}",
        device=device,
        compute_type=compute_type,
        download_root=str(settings.models_dir),
    )


def load_model(model_size: str) -> None:
    global model, current_model_size
    if current_model_size == model_size and model is not None:
        return
    with _model_lock:
        if current_model_size == model_size and model is not None:
            return
        try:
            logger.info(
                f"Model yükleniyor: {model_size} (Cihaz: {settings.device}). "
                "İlk indirme ise birkaç dakika sürebilir..."
            )
            settings.models_dir.mkdir(parents=True, exist_ok=True)

            # Check for disk space if it's likely a fresh download
            req_mb = settings.whisper_model_sizes_mb.get(model_size, 1600)
            free_mb = get_free_disk_mb()

            if free_mb < req_mb:
                logger.error(
                    f"⚠️ Yetersiz disk alanı! Gerekli: ~{req_mb}MB, "
                    f"Mevcut: {free_mb:.1f}MB. İndirme başarısız olabilir."
                )

            try:
                model = _create_whisper_instance(model_size, settings.device, settings.compute_type)
            except RuntimeError as e:
                # Final fallback in case health check missed something
                if settings.device == "cuda":
                    logger.warning(f"CUDA hatası alındı: {e}. CPU üzerinden devam ediliyor.")
                    model = _create_whisper_instance(model_size, "cpu", "int8_float32")
                else:
                    raise

            current_model_size = model_size
            logger.info("Model başarıyla yüklendi.")
        except Exception:
            logger.exception(f"Model yüklenirken hata oluştu: {model_size}")
            model = None
            current_model_size = None


def get_model_status():
    return current_model_size if current_model_size else "Yüklü Değil"
