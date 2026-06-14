import bisect
import logging
import os
import tempfile
from collections.abc import Generator

import av
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.config import EMBEDDING_CACHE_DIR, settings

from .cache_utils import get_embedding_hash
from .encoder import get_voice_encoder

logger = logging.getLogger(__name__)
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _stream_audio(audio_path: str) -> Generator[np.ndarray]:
    """Yields audio chunks from the file to avoid loading everything at once."""
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=settings.sample_rate)
    try:
        with av.open(audio_path) as container:
            if not container.streams.audio:
                return
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    yield out.to_ndarray()[0]
            for out in resampler.resample(None):
                yield out.to_ndarray()[0]
    except (av.AVError, UnicodeDecodeError) as e:
        logger.error(f"Failed to stream audio from {audio_path}: {e}")
        return


def _auto_detect_k(embeddings: np.ndarray) -> tuple[int, np.ndarray]:
    """Fast auto-detection of speaker count using KMeans."""
    n = len(embeddings)
    best_k, best_score, best_labels = 2, -1.0, None

    # Sample embeddings if there are too many to speed up silhouette calculation
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n, min(n, 1000), replace=False) if n > 1000 else np.arange(n)
    eval_embeddings = embeddings[sample_indices]

    # Search up to 10 speakers or the number of samples available
    for k in range(2, min(11, n)):
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(embeddings)
        if len(set(labels)) > 1:
            # Calculate silhouette on sample for speed
            score = float(silhouette_score(eval_embeddings, labels[sample_indices]))
            if score > best_score:
                best_score, best_k, best_labels = score, k, labels
    if best_labels is None:
        model = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
        best_labels = model.fit_predict(embeddings)
    return best_k, best_labels


def diarize(
    audio_path: str,
    num_speakers: int,
    session_id: str = "",
    low_latency: bool = False,
    progress=None,
) -> tuple[list[tuple[float, float, str]], bool]:
    """Returns (start_sec, end_sec, speaker_label) segments covering the full audio."""
    encoder = get_voice_encoder(low_latency=low_latency)

    if progress:
        progress(0.1, desc="Ses dosyası yükleniyor...")

    emb_hash = get_embedding_hash(audio_path, session_id)
    cache_file = EMBEDDING_CACHE_DIR / f"{emb_hash}.npz"
    is_cached = False
    wav_splits = []
    duration_sec = 0.0

    if cache_file.exists():
        if progress:
            progress(0.3, desc="İmzalar önbellekten yükleniyor...")
        is_cached = True
        with np.load(cache_file, allow_pickle=True) as data:
            partial_embeds = data["embeds"]
            wav_splits = [slice(start, stop) for start, stop in zip(data["starts"], data["stops"], strict=False)]
            if wav_splits:
                duration_sec = wav_splits[-1].stop / settings.sample_rate
    else:
        if progress:
            progress(0.3, desc="Konuşmacı imzaları çıkarılıyor...")

        all_partial_embeds = []
        current_samples = []
        current_count = 0
        total_processed = 0
        chunk_limit = 5 * 60 * settings.sample_rate

        def process_chunk():
            nonlocal total_processed, current_samples, current_count
            if not current_samples:
                return
            segment = np.concatenate(current_samples).astype(np.float32)
            m = np.abs(segment).max()
            if m > 0:
                segment /= m / 0.9
            # Lowering rate to 1.0 significantly speeds up extraction by reducing
            # forward passes by 50% compared to rate=2.0.
            _, embeds, slices = encoder.embed_utterance(segment, return_partials=True, rate=1.0)
            all_partial_embeds.append(embeds)
            for s in slices:
                wav_splits.append(slice(s.start + total_processed, s.stop + total_processed))
            total_processed += current_count
            current_samples = []
            current_count = 0

        for audio_chunk in _stream_audio(audio_path):
            current_samples.append(audio_chunk)
            current_count += len(audio_chunk)
            if current_count >= chunk_limit:
                process_chunk()

        process_chunk()

        if not all_partial_embeds:
            return ([(0.0, 0.0, "Sistem: Ses yüklenemedi")], False)

        partial_embeds = np.concatenate(all_partial_embeds, axis=0)
        duration_sec = total_processed / settings.sample_rate

        # Atomic save for numpy files
        with tempfile.NamedTemporaryFile(dir=cache_file.parent, delete=False, suffix=".npz") as tmp:
            np.savez(
                tmp,
                embeds=partial_embeds,
                starts=[s.start for s in wav_splits],
                stops=[s.stop for s in wav_splits],
            )
            os.replace(tmp.name, cache_file)

    if progress:
        progress(0.8, desc="Konuşmacılar gruplandırılıyor...")

    # Unit normalize embeddings for more accurate clustering and silhouette scores
    norms = np.linalg.norm(partial_embeds, axis=1, keepdims=True)
    partial_embeds = np.divide(
        partial_embeds,
        norms,
        out=np.zeros_like(partial_embeds),
        where=norms > 1e-6,
    )

    n = len(partial_embeds)
    if duration_sec < 2.0 or n < 2 or num_speakers == 1:
        return ([(0.0, duration_sec, "Konuşmacı 1")], is_cached)

    if num_speakers <= 0:
        num_speakers, best_labels = _auto_detect_k(partial_embeds)
        if num_speakers < 2:
            return ([(0.0, duration_sec, "Konuşmacı 1")], False)
    else:
        num_speakers = min(num_speakers, n)
        model = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
        best_labels = model.fit_predict(partial_embeds)

    seen: dict[int, int] = {}

    def speaking_order(raw: int) -> int:
        if raw not in seen:
            seen[raw] = len(seen) + 1
        return seen[raw]

    segments = [
        (
            split.start / settings.sample_rate,
            split.stop / settings.sample_rate,
            f"Konuşmacı {speaking_order(int(lbl))}",
        )
        for split, lbl in zip(wav_splits, best_labels, strict=False)
    ]
    segments.sort(key=lambda x: x[0])
    return segments, is_cached


def dominant_speaker(start: float, end: float, timeline: list[tuple[float, float, str]]) -> str:
    """Returns the speaker with the greatest time overlap in [start, end].

    Requires timeline to be sorted by start time (guaranteed by diarize()).
    """
    if not timeline:
        return "Bilinmeyen"

    # Use binary search to find segments starting before the current window ends
    # bisect_left returns the first index where timeline[i][0] >= end
    idx_end = bisect.bisect_left(timeline, (end,))

    # Find approximate start index and look back one segment to catch overlaps
    # that started before the 'start' timestamp.
    idx_start = bisect.bisect_left(timeline, (start,))
    search_start = max(0, idx_start - 1)

    votes: dict[str, float] = {}
    for i in range(search_start, idx_end):
        t0, t1, speaker = timeline[i]
        overlap = min(end, t1) - max(start, t0)
        if overlap > 0:
            votes[speaker] = votes.get(speaker, 0.0) + overlap

    if votes:
        return max(votes, key=lambda k: votes[k])

    mid = (start + end) / 2.0
    # Fallback to the closest segment using the sliced window
    search_range = timeline[search_start : idx_end + 1] or timeline
    return min(search_range, key=lambda s: abs((s[0] + s[1]) / 2.0 - mid))[2]
