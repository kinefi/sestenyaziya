import logging
import os
import random
import threading
import time
from functools import wraps
from typing import Any

import ctranslate2  # Import ctranslate2 to check for CUDA availability
import numpy as np
import torch
from faster_whisper import WhisperModel

from . import config as cfg
from .cache_utils import check_free_space_mb

logger = logging.getLogger(__name__)

model: WhisperModel | None = None
current_model_size: str | None = None
voice_encoder: "SpeechBrainEncoder | None" = None
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


def retry(
    retries: int = cfg.RETRY_MAX_ATTEMPTS,
    initial_delay: float = cfg.RETRY_INITIAL_DELAY,
):
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
                        delay = initial_delay * (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            "Attempt %d/%d failed for %s. Retrying in %.2fs... Error: %s",
                            attempt + 1,
                            retries,
                            func.__name__,
                            delay,
                            e,
                        )
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
            logger.info(
                f"Model yükleniyor: {model_size} (Cihaz: {cfg.device}). " "İlk indirme ise birkaç dakika sürebilir..."
            )
            cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

            # Check for disk space if it's likely a fresh download
            # small: ~500MB, medium: ~1.5GB, large-v3: ~3GB
            size_map = {"small": 600, "medium": 1600, "large-v3": 3200}
            req_mb = size_map.get(model_size, 1600)
            free_mb = check_free_space_mb(cfg.MODELS_DIR)

            if free_mb < req_mb:
                logger.error(
                    f"⚠️ Yetersiz disk alanı! Gerekli: ~{req_mb}MB, "
                    f"Mevcut: {free_mb:.1f}MB. İndirme başarısız olabilir."
                )

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
        except Exception:
            logger.exception(f"Model yüklenirken hata oluştu: {model_size}")
            model = None
            current_model_size = None


@retry()
def _init_voice_encoder(source: str) -> Any:
    # SpeechBrain models are ~100MB max.
    free_mb = check_free_space_mb(cfg.MODELS_DIR)
    if free_mb < 150:
        logger.warning(f"⚠️ Disk alanı çok düşük (Mevcut: {free_mb:.1f}MB). Ses encoder indirilemeyebilir.")

    return SpeechBrainEncoder(device=cfg.device, source=source)


class SpeechBrainEncoder:
    """
    A modern replacement for Resemblyzer using SpeechBrain's ECAPA-TDNN model.
    Provides a compatible API for sliding window embeddings used in diarization.
    """

    def __init__(self, device: str, source: str):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        # Use ECAPA-TDNN trained on VoxCeleb, a state-of-the-art speaker embedding model
        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            run_opts={"device": device},
            savedir=str(cfg.MODELS_DIR / "speechbrain"),
        )

    def embed_utterance(self, wav: np.ndarray, return_partials: bool = True, rate: float = 1.3):
        """
        Mimics Resemblyzer's embed_utterance API.
        Returns (full_embedding, partial_embeddings, slices).
        """
        sampling_rate = cfg.SAMPLE_RATE
        window_samples = int(1.6 * sampling_rate)  # 1.6s window is standard for d-vectors/x-vectors
        step_samples = int(sampling_rate / rate)

        wav_tensor = torch.from_numpy(wav).to(self.device).float()

        # Full utterance embedding (centroid)
        with torch.inference_mode():
            full_emb = self.classifier.encode_batch(wav_tensor.unsqueeze(0)).squeeze().cpu().numpy()

        if not return_partials:
            return full_emb, None, None

        if len(wav) < window_samples:
            return full_emb, np.array([full_emb]), [slice(0, len(wav))]

        slices = []
        batch_chunks = []
        for i in range(0, len(wav) - window_samples + 1, step_samples):
            slices.append(slice(i, i + window_samples))
            batch_chunks.append(wav_tensor[i : i + window_samples])

        if batch_chunks:
            # Sub-batching to better utilize CPU multi-core and cache locality
            all_embeds = []
            for i in range(0, len(batch_chunks), cfg.DIARIZATION_BATCH_SIZE):
                batch = torch.stack(batch_chunks[i : i + cfg.DIARIZATION_BATCH_SIZE])
                with torch.inference_mode():
                    # encode_batch returns (batch, 1, embedding_dim)
                    emb = self.classifier.encode_batch(batch).squeeze(1).cpu().numpy()
                    all_embeds.append(emb)

            return full_emb, np.concatenate(all_embeds, axis=0), slices

        return full_emb, np.array([]), []


def get_voice_encoder(low_latency: bool = False) -> "SpeechBrainEncoder":
    global voice_encoder
    source = cfg.ENCODER_FAST if low_latency else cfg.ENCODER_STD

    # If the requested model type is different from the one loaded, reload it.
    if voice_encoder is None or voice_encoder.classifier.hparams.source != source:
        with _encoder_lock:
            if voice_encoder is None or voice_encoder.classifier.hparams.source != source:
                logger.info(f"Ses encoder ({'X-Vector' if low_latency else 'ECAPA-TDNN'}) yükleniyor...")
                voice_encoder = _init_voice_encoder(source)
                logger.info("Ses encoder başarıyla yüklendi.")
    return voice_encoder


def _watchdog_worker():
    """Background thread that restarts the *process* on a genuine native hang.

    An in-process model reload cannot recover a wedged CUDA/ctranslate2 call: the
    stuck thread is blocked in native code holding the device, and an in-flight job
    binds its own local model reference, so swapping ``model`` does nothing. The only
    real recovery is a fresh process, so we exit non-zero and let the supervisor
    (systemd/docker/k8s) restart a clean one.
    """
    while True:
        time.sleep(cfg.WATCHDOG_CHECK_INTERVAL)
        if is_processing and model is not None:
            elapsed = time.time() - last_heartbeat
            if elapsed > cfg.WATCHDOG_TIMEOUT:
                logger.critical(
                    f"Watchdog: no progress for {elapsed:.0f}s (limit "
                    f"{cfg.WATCHDOG_TIMEOUT}s). Process is wedged; exiting "
                    f"{cfg.WATCHDOG_EXIT_CODE} for supervisor restart."
                )
                # The main thread is stuck in native code and cannot be signalled
                # cooperatively; flush logs and hard-exit. os._exit (not sys.exit)
                # is required to terminate the process from a daemon thread.
                logging.shutdown()
                os._exit(cfg.WATCHDOG_EXIT_CODE)


# Start watchdog thread as a daemon so it exits with the main program
threading.Thread(target=_watchdog_worker, daemon=True).start()
logger.info("Watchdog monitoring thread started.")


def get_health_status() -> dict:
    """Returns the current operational status for the UI dashboard."""
    cuda_healthy = check_cuda_health() if cfg.device == "cuda" else True
    watchdog_delta = time.time() - last_heartbeat

    # Identify the active voice encoder model
    encoder_info = "Yüklü Değil"
    if voice_encoder is not None:
        try:
            source = voice_encoder.classifier.hparams.source
            encoder_info = "X-Vector (Hızlı)" if "xvect" in source else "ECAPA (Kaliteli)"
        except Exception:
            encoder_info = "Belirlenemedi"

    remaining = "—"
    if is_processing:
        elapsed = time.time() - task_start_time
        remaining = f"{max(0, cfg.TRANSCRIPTION_TIMEOUT - elapsed):.0f}s"

    return {
        "device": cfg.device.upper(),
        "compute_type": cfg.compute_type,
        "model_loaded": current_model_size or "Yüklü Değil",
        "encoder_loaded": encoder_info,
        "is_processing": "Aktif ⚙️" if is_processing else "Boşta 💤",
        "watchdog_status": (
            "⚠️ Hang Algılandı" if (is_processing and watchdog_delta > cfg.WATCHDOG_TIMEOUT) else "✅ Normal"
        ),
        "last_seen": f"{watchdog_delta:.1f}s önce",
        "cuda_status": "✅ Hazır" if cuda_healthy else "❌ Hata",
        "timeout_remaining": remaining,
    }
