import threading
import traceback

from faster_whisper import WhisperModel
from resemblyzer import VoiceEncoder

from .config import device, compute_type

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
            print(f"🔄 Model yükleniyor ({model_size}, {device})...")
            model = WhisperModel(
                f"Systran/faster-whisper-{model_size}",
                device=device,
                compute_type=compute_type,
            )
            current_model_size = model_size
            print("✅ Model yüklendi!")
        except Exception:
            traceback.print_exc()
            model = None
            current_model_size = None


def get_voice_encoder() -> VoiceEncoder:
    global voice_encoder
    if voice_encoder is None:
        with _encoder_lock:
            if voice_encoder is None:
                print("🔄 Ses encoder yükleniyor...")
                voice_encoder = VoiceEncoder()
                print("✅ Ses encoder yüklendi!")
    return voice_encoder
