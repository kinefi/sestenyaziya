import logging
import os
import threading
import time

from . import model as model_module
from .config import is_cuda_available, settings, watchdog_settings
from .encoder import get_encoder_status
from .model import get_model_status

logger = logging.getLogger(__name__)


# Watchdog state
last_heartbeat: float = time.time()
task_start_time = 0.0
active_timeout = settings.transcription_timeout
is_processing = False

pause_event = threading.Event()
stop_event = threading.Event()


def heartbeat():
    """Updates the last activity timestamp to prevent watchdog intervention."""
    global last_heartbeat
    last_heartbeat = time.time()


def set_processing_state(state: bool, timeout: int = None):
    """Toggles processing state for watchdog monitoring."""
    global is_processing, task_start_time, active_timeout
    is_processing = state
    if state:
        heartbeat()
        task_start_time = time.time()
        active_timeout = timeout if timeout is not None else settings.transcription_timeout


def _watchdog_worker():
    """Background thread that restarts the *process* on a genuine native hang.

    An in-process model reload cannot recover a wedged CUDA/ctranslate2 call: the
    stuck thread is blocked in native code holding the device, and an in-flight job
    binds its own local model reference, so swapping ``model`` does nothing. The only
    real recovery is a fresh process, so we exit non-zero and let the supervisor
    (systemd/docker/k8s) restart a clean one.
    """
    while True:
        time.sleep(watchdog_settings.check_interval)
        if is_processing and model_module.model is not None:
            elapsed = time.time() - last_heartbeat
            if elapsed > watchdog_settings.timeout:
                logger.critical(
                    f"Watchdog: no progress for {elapsed:.0f}s (limit "
                    f"{watchdog_settings.timeout}s). Process is wedged; exiting "
                    f"{watchdog_settings.exit_code} for supervisor restart."
                )
                # The main thread is stuck in native code and cannot be signalled
                # cooperatively; flush logs and hard-exit. os._exit (not sys.exit)
                # is required to terminate the process from a daemon thread.
                logging.shutdown()
                os._exit(watchdog_settings.exit_code)


# Start watchdog thread as a daemon so it exits with the main program
threading.Thread(target=_watchdog_worker, daemon=True).start()
logger.info("Watchdog monitoring thread started.")


def get_health_status() -> dict:
    """Returns the current operational status for the UI dashboard."""
    cuda_healthy = is_cuda_available() if settings.device == "cuda" else True
    watchdog_delta = time.time() - last_heartbeat

    remaining = "—"
    remaining_raw = None
    if is_processing:
        elapsed = time.time() - task_start_time
        remaining_raw = max(0, active_timeout - elapsed)
        remaining = f"{remaining_raw:.0f}s"

    return {
        "device": settings.device.upper(),
        "compute_type": settings.compute_type,
        "model_loaded": get_model_status(),
        "encoder_loaded": get_encoder_status(),
        "is_processing": "Aktif ⚙️" if is_processing else "Boşta 💤",
        "watchdog_status": (
            "⚠️ Hang Algılandı" if (is_processing and watchdog_delta > watchdog_settings.timeout) else "✅ Normal"
        ),
        "last_seen": f"{watchdog_delta:.1f}s önce",
        "cuda_status": "✅ Hazır" if cuda_healthy else "❌ Hata",
        "timeout_remaining": remaining,
        "remaining_raw": remaining_raw,
        "active_timeout": active_timeout,
    }
