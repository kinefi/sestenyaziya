import logging
import threading
from typing import Any

import numpy as np
import torch

from .cache_utils import get_free_disk_mb
from .config import settings
from .utils import retry

logger = logging.getLogger(__name__)


class SpeechBrainEncoder:
    """
    A modern replacement for Resemblyzer using SpeechBrain's ECAPA-TDNN model.
    Provides a compatible API for sliding window embeddings used in diarization.
    """

    def __init__(self, device: str, source: str):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        self.source = source
        # Use ECAPA-TDNN trained on VoxCeleb, a state-of-the-art speaker embedding model
        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            run_opts={"device": device},
            savedir=str(settings.models_dir / "speechbrain"),
        )

    def embed_utterance(self, wav: np.ndarray, return_partials: bool = True, rate: float = 1.3):
        """
        Mimics Resemblyzer's embed_utterance API.
        Returns (full_embedding, partial_embeddings, slices).
        """
        sampling_rate = settings.sample_rate
        # 1.6s window is standard for d-vectors/x-vectors
        window_samples = int(1.6 * sampling_rate)
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
            for i in range(0, len(batch_chunks), settings.diarization_batch_size):
                batch = torch.stack(batch_chunks[i : i + settings.diarization_batch_size])
                with torch.inference_mode():
                    # encode_batch returns (batch, 1, embedding_dim)
                    emb = self.classifier.encode_batch(batch).squeeze(1).cpu().numpy()
                    all_embeds.append(emb)

            return full_emb, np.concatenate(all_embeds, axis=0), slices

        return full_emb, np.array([]), []


voice_encoder: SpeechBrainEncoder | None = None
_encoder_lock = threading.Lock()


@retry()
def _init_voice_encoder(source: str) -> Any:
    # SpeechBrain models are ~100MB max.
    free_mb = get_free_disk_mb()
    if free_mb < 150:
        logger.warning(f"⚠️ Disk alanı çok düşük (Mevcut: {free_mb:.1f}MB). Ses encoder indirilemeyebilir.")

    return SpeechBrainEncoder(device=settings.device, source=source)


def get_voice_encoder(low_latency: bool = False) -> SpeechBrainEncoder:
    global voice_encoder
    source = settings.encoder_fast_model if low_latency else settings.encoder_std_model

    logger.info(f"Ses encoder modeli: {source}")

    # If the requested model type is different from the one loaded, reload it.
    # Use isinstance to guard against stale/corrupted objects (e.g. SimpleNamespace from SpeechBrain internals).
    if not isinstance(voice_encoder, SpeechBrainEncoder) or voice_encoder.source != source:
        with _encoder_lock:
            if not isinstance(voice_encoder, SpeechBrainEncoder) or voice_encoder.source != source:
                logger.info(f"Ses encoder ({'X-Vector' if low_latency else 'ECAPA-TDNN'}) yükleniyor...")
                voice_encoder = _init_voice_encoder(source)
                logger.info("Ses encoder başarıyla yüklendi.")
    return voice_encoder


def get_encoder_status():
    encoder_info = "Yüklü Değil"
    if voice_encoder is not None:
        try:
            # Safely check for 'source' attribute on the wrapper object.
            # This handles both stale instances and potential reload issues.
            source = getattr(voice_encoder, "source", None)
            if source:
                source_str = str(source).lower()
                encoder_info = "X-Vector (Hızlı)" if "xvect" in source_str else "ECAPA (Kaliteli)"
            else:
                encoder_info = "Yüklendi"
        except Exception as e:
            logger.error(f"Encoder durum bilgisi alınırken hata oluştu: {e}")
            encoder_info = "Belirlenemedi"
    return encoder_info
