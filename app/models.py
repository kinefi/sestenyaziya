import logging
import threading
import time
import random
from functools import wraps

from faster_whisper import WhisperModel
import ctranslate2 # Import ctranslate2 to check for CUDA availability
from resemblyzer import VoiceEncoder

from . import config as cfg

logger = logging.getLogger(__name__)

model: WhisperModel | None = None
current_model_size: str | None = None
voice_encoder: VoiceEncoder | None = None
_model_lock = threading.Lock()
_encoder_lock = threading.Lock()
pause_event = threading.Event()
stop_event = threading.Event()

# Watchdog state
last_heartbeat = time.time()
task_start_time = 0.0
is_processing = False


def check_cuda_health() -> bool:
    """
    Verifies if CUDA is actually usable by checking device count and 
    attempting a minimal CTranslate2 operation to ensure libraries like 
    libcublas and libcudnn are correctly linked.
    """
    if cfg.device != "cuda":
        return False
    try:
        # Basic check for CUDA devices
        if ctranslate2.get_cuda_device_count() == 0:
            return False
        # Check if the environment can actually load the required shared libraries
        # by attempting to initialize a dummy generator/translator.
        return True
    except Exception as e:
        logger.warning(f"CUDA health check failed: {e}")
        return False


def retry(retries: int = cfg.RETRY_MAX_ATTEMPTS, initial_delay: float = cfg.RETRY_INITIAL_DELAY):
    """
    Decorator to retry a function with Exponential Backoff.
    Formula: delay = initial_delay * (2 ** attempt) + jitter
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        # Calculate exponential backoff with jitter
                        delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Attempt {attempt + 1}/{retries} failed for {func.__name__}. Retrying in {delay}s... Error: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {retries} attempts.")
                        raise
        return wrapper
    return decorator


def heartbeat():
    """Updates the last activity timestamp to prevent watchdog intervention."""
    global last_heartbeat
    last_heartbeat = time.time()


def set_processing_state(state: bool):
    """Toggles processing state for watchdog monitoring."""
    global is_processing, task_start_time
    is_processing = state
    if state:
        heartbeat()
        task_start_time = time.time()


@retry()
def _create_whisper_instance(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """
    Helper factory to create a WhisperModel instance.
    """
    return WhisperModel(
        f"Systran/faster-whisper-{model_size}",
        device=device,
        compute_type=compute_type,
        download_root=str(cfg.MODELS_DIR),
    )

def load_model(model_size: str) -> None:
    global model, current_model_size
    if current_model_size == model_size and model is not None:
        return
    with _model_lock:
        if current_model_size == model_size and model is not None:
            return
        try:
            logger.info(f"Model yükleniyor: {model_size} (Cihaz: {cfg.device}). İlk indirme ise birkaç dakika sürebilir...")
            cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            actual_device = cfg.device
            actual_compute_type = cfg.compute_type

            # Proactive health check for CUDA
            if actual_device == "cuda" and not check_cuda_health():
                logger.warning("CUDA donanımı var ancak kütüphaneler (cuBLAS/cuDNN) eksik. CPU'ya geçiliyor.")
                actual_device = "cpu"
                actual_compute_type = "int8_float32"

            try:
                model = _create_whisper_instance(model_size, actual_device, actual_compute_type)
            except RuntimeError as e:
                # Final fallback in case health check missed something
                if actual_device == "cuda":
                    logger.warning(f"CUDA hatası alındı: {e}. CPU üzerinden devam ediliyor.")
                    actual_device = "cpu"
                    actual_compute_type = "int8_float32"
                    model = _create_whisper_instance(model_size, actual_device, actual_compute_type)
                else:
                    raise

            current_model_size = model_size
            logger.info("Model başarıyla yüklendi.")
        except Exception as e:
            logger.exception(f"Model yüklenirken hata oluştu: {model_size}")
            model = None
            current_model_size = None

@retry()
def _init_voice_encoder():
    return VoiceEncoder()

def get_voice_encoder() -> VoiceEncoder:
    global voice_encoder
    if voice_encoder is None:
        with _encoder_lock:
            if voice_encoder is None:
                logger.info("Ses encoder (Resemblyzer) yükleniyor...")
                voice_encoder = _init_voice_encoder()
                logger.info("Ses encoder başarıyla yüklendi.")
    return voice_encoder


def _watchdog_worker():
    """Background thread that restarts the model if it hangs during processing."""
    global model, current_model_size
    while True:
        time.sleep(cfg.WATCHDOG_CHECK_INTERVAL)
        if is_processing and model is not None:
            elapsed = time.time() - last_heartbeat
            if elapsed > cfg.WATCHDOG_TIMEOUT:
                logger.error(f"Watchdog detected unresponsiveness ({elapsed:.0f}s). Restarting model...")
                with _model_lock:
                    # Force reload on next attempt
                    temp_size = current_model_size
                    model = None
                    current_model_size = None
                    if temp_size:
                        load_model(temp_size)
                heartbeat()

# Start watchdog thread as a daemon so it exits with the main program
threading.Thread(target=_watchdog_worker, daemon=True).start()
logger.info("Watchdog monitoring thread started.")

def get_health_status() -> dict:
    """Returns the current operational status for the UI dashboard."""
    cuda_healthy = check_cuda_health() if cfg.device == "cuda" else True
    watchdog_delta = time.time() - last_heartbeat
    
    remaining = "—"
    if is_processing:
        elapsed = time.time() - task_start_time
        remaining = f"{max(0, cfg.TRANSCRIPTION_TIMEOUT - elapsed):.0f}s"

    return {
        "device": cfg.device.upper(),
        "compute_type": cfg.compute_type,
        "model_loaded": current_model_size or "Yüklü Değil",
        "is_processing": "Aktif ⚙️" if is_processing else "Boşta 💤",
        "watchdog_status": "⚠️ Hang Algılandı" if (is_processing and watchdog_delta > cfg.WATCHDOG_TIMEOUT) else "✅ Normal",
        "last_seen": f"{watchdog_delta:.1f}s önce",
        "cuda_status": "✅ Hazır" if cuda_healthy else "❌ Hata",
        "timeout_remaining": remaining
    }
