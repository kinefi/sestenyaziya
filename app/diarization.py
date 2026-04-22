import av
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import models
from .config import SAMPLE_RATE


def _load_wav(audio_path: str) -> np.ndarray:
    """Load audio as mono float32 at SAMPLE_RATE Hz via PyAV (handles m4a, mp3, wav, …)."""
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    with av.open(audio_path) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray()[0])
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray()[0])
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    wav = np.concatenate(chunks).astype(np.float32)
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav


def _auto_detect_k(embeddings: np.ndarray) -> tuple[int, np.ndarray]:
    """Pick the k with the best silhouette score in range [2, 5]."""
    n = len(embeddings)
    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, min(6, n)):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
        if len(set(labels)) > 1:
            score = float(silhouette_score(embeddings, labels))
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
    if best_labels is None:
        best_labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(embeddings)
    return best_k, best_labels


def diarize(audio_path: str, num_speakers: int) -> list[tuple[float, float, str]]:
    """Returns (start_sec, end_sec, speaker_label) segments covering the full audio."""
    encoder = models.get_voice_encoder()
    wav = _load_wav(audio_path)
    duration_sec = len(wav) / SAMPLE_RATE

    if duration_sec < 2.0:
        return [(0.0, duration_sec, "Konuşmacı 1")]

    # rate=2 → one embedding per 500 ms; sufficient for clustering, ~8× faster than rate=16
    _, partial_embeds, wav_splits = encoder.embed_utterance(
        wav, return_partials=True, rate=2
    )

    n = len(partial_embeds)
    if n < 2:
        return [(0.0, duration_sec, "Konuşmacı 1")]

    if num_speakers <= 1:
        num_speakers, best_labels = _auto_detect_k(partial_embeds)
    else:
        num_speakers = min(num_speakers, n)
        best_labels = KMeans(n_clusters=num_speakers, random_state=42, n_init=10).fit_predict(partial_embeds)

    seen: dict[int, int] = {}

    def speaking_order(raw: int) -> int:
        if raw not in seen:
            seen[raw] = len(seen) + 1
        return seen[raw]

    return [
        (split.start / SAMPLE_RATE, split.stop / SAMPLE_RATE, f"Konuşmacı {speaking_order(int(lbl))}")
        for split, lbl in zip(wav_splits, best_labels)
    ]


def dominant_speaker(start: float, end: float, timeline: list[tuple[float, float, str]]) -> str:
    """Returns the speaker with the greatest time overlap in [start, end]."""
    votes: dict[str, float] = {}
    for t0, t1, speaker in timeline:
        overlap = min(end, t1) - max(start, t0)
        if overlap > 0:
            votes[speaker] = votes.get(speaker, 0.0) + overlap
    if votes:
        return max(votes, key=lambda k: votes[k])
    mid = (start + end) / 2.0
    return min(timeline, key=lambda s: abs((s[0] + s[1]) / 2.0 - mid))[2]
