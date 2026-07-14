import logging
import time
from collections.abc import Generator
from dataclasses import astuple, dataclass
from pathlib import Path
import tempfile
import os
from types import SimpleNamespace

from pydub import AudioSegment

from . import model
from .cache_utils import atomic_write_text, clean_cache_directories, get_transcription_hash
from .config import TRANSCRIPT_CACHE_DIR, settings
from .diarization import diarize, dominant_speaker
from .watchdog import heartbeat, pause_event, set_processing_state, stop_event

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    result: str
    txt_path: str | None
    srt_path: str | None
    vtt_path: str | None
    status: str
    speaker_info: str
    progress: float = 0.0


TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fmt(seconds: float, is_sub: bool = False, ms_sep: str = ",") -> str:
    if not is_sub:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // (1000 * 60)) % 60
    h = total_ms // (1000 * 60 * 60)
    return f"{h:02d}:{m:02d}:{s:02d}{ms_sep}{ms:03d}"


def _generate_srt_vtt(segments: list, is_vtt: bool = False, speaker_timeline=None) -> str:
    output = "WEBVTT\n\n" if is_vtt else ""
    sep = "." if is_vtt else ","

    for i, seg in enumerate(segments, 1):
        start_str = _fmt(seg.start, True, sep)
        end_str = _fmt(seg.end, True, sep)

        speaker_prefix = ""
        if speaker_timeline:
            speaker = dominant_speaker(seg.start, seg.end, speaker_timeline)
            speaker_prefix = f"{speaker}: "

        if is_vtt:
            output += f"{start_str} --> {end_str}\n{speaker_prefix}{seg.text.strip()}\n\n"
        else:
            output += f"{i}\n{start_str} --> {end_str}\n{speaker_prefix}{seg.text.strip()}\n\n"
    return output


def _save_transcription_artifacts(text: str, segments: list, speaker_timeline: list | None, paths: tuple):
    """Helper to write all output formats to disk atomically."""
    txt_p, srt_p, vtt_p = paths
    if text.strip():
        atomic_write_text(txt_p, text.strip())
    if segments:
        srt_content = _generate_srt_vtt(segments, is_vtt=False, speaker_timeline=speaker_timeline)
        atomic_write_text(srt_p, srt_content)
        vtt_content = _generate_srt_vtt(segments, is_vtt=True, speaker_timeline=speaker_timeline)
        atomic_write_text(vtt_p, vtt_content)


def transcribe(
    audio_path: str,
    model_size: str,
    enable_diarization: bool,
    num_speakers: int,
    session_id: str,
    timeout: int,
    language: str | None = "tr",
    low_latency: bool = False,
    # runtime overrides from UI
    chunk_size_seconds: int | None = None,
    chunk_overlap_seconds: int | None = None,
    use_chunking: bool = True,
    use_ffmpeg_piping: bool = False,
) -> Generator[tuple]:
    """Ses dosyasını Türkçe metne dönüştürür; akış olarak sonuç verir."""
    pause_event.clear()
    stop_event.clear()
    set_processing_state(True, timeout=timeout)

    speaker_info = ""  # Initialize early for error/early exit paths
    try:
        # Validate inputs before doing any expensive work.
        if audio_path is None:
            yield astuple(
                TranscriptionResult(
                    result="⚠️ Lütfen bir ses dosyası yükleyin.",
                    txt_path=None,
                    srt_path=None,
                    vtt_path=None,
                    status="",
                    speaker_info="",
                    progress=0.0,
                )
            )
            return

        # ⚡ Check for cached transcription
        t_hash = get_transcription_hash(
            audio_path,
            model_size,
            enable_diarization,
            num_speakers,
            session_id,
            language or "auto",
        )
        txt_cache = TRANSCRIPT_CACHE_DIR / f"{t_hash}.txt"
        srt_cache = TRANSCRIPT_CACHE_DIR / f"{t_hash}.srt"
        vtt_cache = TRANSCRIPT_CACHE_DIR / f"{t_hash}.vtt"

        if txt_cache.exists() and srt_cache.exists() and vtt_cache.exists():
            logger.info(f"Loading transcription from cache: {t_hash}")
            cached_text = txt_cache.read_text(encoding="utf-8")

            yield astuple(
                TranscriptionResult(
                    result=cached_text,
                    txt_path=str(txt_cache),
                    srt_path=str(srt_cache),
                    vtt_path=str(vtt_cache),
                    status="✅ Önbellekten yüklendi!",
                    speaker_info="⚡ İşlem atlandı",
                    progress=100.0,
                )
            )
            return

        if model.current_model_size != model_size or model.model is None:
            yield astuple(
                TranscriptionResult(
                    result="",
                    txt_path=None,
                    srt_path=None,
                    vtt_path=None,
                    status="🔄 Model yükleniyor, lütfen bekleyin...",
                    speaker_info="",
                    progress=0.0,
                )
            )
            model.load_model(model_size)

        # Bind a stable local reference
        if model.model is None:
            yield astuple(
                TranscriptionResult(
                    result="❌ Hata: Model yüklenemedi. Logları kontrol edin.",
                    txt_path=None,
                    srt_path=None,
                    vtt_path=None,
                    status="",
                    speaker_info="",
                    progress=0.0,
                )
            )
            return

        result = ""
        start_time = time.time()

        file_size_mb = 0.0
        try:
            file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
        except OSError:
            pass

        speed_optimized = low_latency or file_size_mb >= 40.0
        if speed_optimized and not low_latency:
            logger.info(
                "Uzun ses dosyası algılandı (%.1f MB). Hız için transkripsiyon ayarları optimize ediliyor...",
                file_size_mb,
            )

        speaker_timeline: list[tuple[float, float, str]] | None = None
        if enable_diarization:
            try:
                yield astuple(
                    TranscriptionResult(
                        result="",
                        txt_path=None,
                        srt_path=None,
                        vtt_path=None,
                        status="🔍 Konuşmacılar analiz ediliyor...",
                        speaker_info="",
                        progress=5.0,
                    )
                )

                def progress_callback(pct, desc=""):
                    heartbeat()
                    logger.info(f"Diarization: {pct * 100:.0f}% - {desc}")

                speaker_timeline, used_cache = diarize(
                    audio_path,
                    int(num_speakers),
                    session_id=session_id,
                    low_latency=low_latency,
                    progress=progress_callback,
                )
                n = len(set(lbl for _, _, lbl in speaker_timeline))
                cache_tag = " · ⚡ Önbellek" if used_cache else ""
                speaker_info = f"👥 {n} konuşmacı algılandı{cache_tag}"
            except Exception as e:
                logger.exception("Konuşmacı ayrıştırma sırasında hata oluştu")
                yield astuple(
                    TranscriptionResult(
                        result="",
                        txt_path=None,
                        srt_path=None,
                        vtt_path=None,
                        status=(
                            f"⚠️ Konuşmacı ayrıştırma başarısız ({type(e).__name__}: {e}), "
                            "düz metin devam ediliyor..."
                        ),
                        speaker_info="",
                        progress=10.0,
                    )
                )

        yield astuple(
            TranscriptionResult(
                result="",
                txt_path=None,
                srt_path=None,
                vtt_path=None,
                status="⏳ Transkripsiyon başlatılıyor...",
                speaker_info=speaker_info,
                progress=10.0,
            )
        )

        current_beam_size = 1 if speed_optimized else 5
        temperature = [0.0] if speed_optimized else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        vad_parameters = {"min_silence_duration_ms": 800 if speed_optimized else 500}

        logger.info(
            "Transcription starting: model=%s, lang=%s, beam=%s, temp=%s, vad_ms=%s",
            model_size,
            language or "auto",
            current_beam_size,
            temperature,
            vad_parameters["min_silence_duration_ms"],
        )

        # Passing an explicit language improves accuracy by preventing "language hallucination"
        # and speeds up the process by skipping the first 30-second detection phase.
        target_lang = language if language != "auto" else None
        # For very long files, transcribe in chunks to improve responsiveness
        initial_prompt = (
            "Türkçe konuşma kaydı. Noktalama ve büyük harf kullanımı." if language == "tr" else None
        )

        def _transcribe_in_chunks(
            audio_path: str,
            model_obj,
            target_lang,
            beam_size,
            temperature,
            vad_parameters,
            chunk_seconds: int,
            overlap_seconds: int,
            initial_prompt,
            use_ffmpeg: bool = False,
        ):
            """Split `audio_path` into chunks with `chunk_seconds` length and `overlap_seconds` overlap,
            transcribe each chunk with `model_obj.transcribe`, then stitch segments back together
            while adjusting timestamps to the original audio timeline.
            Returns: (segments_list, info_like)
            """
            chunk_ms = int(chunk_seconds * 1000)
            overlap_ms = int(overlap_seconds * 1000)
            step_ms = chunk_ms - overlap_ms if chunk_ms > overlap_ms else chunk_ms

            combined_segments: list[SimpleNamespace] = []
            last_end_global = 0.0

            # Use ffmpeg piping to decode chunks directly into numpy if requested
            for start_ms in range(0, 10**9, step_ms):
                # compute byte ranges for chunking using ffprobe/fallback via pydub
                # We'll load the chunk using pydub for duration convenience, but decode via ffmpeg when piping
                audio = AudioSegment.from_file(audio_path)
                total_duration = len(audio) / 1000.0
                if start_ms >= len(audio):
                    break
                end_ms = min(start_ms + chunk_ms, len(audio))

                offset = start_ms / 1000.0

                if use_ffmpeg:
                    # Decode chunk via ffmpeg to float32 PCM on stdout
                    cmd = [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{start_ms/1000}",
                        "-to",
                        f"{end_ms/1000}",
                        "-i",
                        str(audio_path),
                        "-f",
                        "f32le",
                        "-acodec",
                        "pcm_f32le",
                        "-ac",
                        "1",
                        "-ar",
                        str(settings.sample_rate),
                        "-",
                    ]
                    try:
                        import subprocess

                        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        out, err = p.communicate()
                        if p.returncode != 0:
                            raise RuntimeError(err.decode(errors="ignore"))
                        import numpy as _np

                        audio_array = _np.frombuffer(out, dtype=_np.float32)
                        try:
                            segs, info = model_obj.transcribe(
                                audio_array,
                                language=target_lang,
                                beam_size=beam_size,
                                temperature=temperature,
                                condition_on_previous_text=False,
                                compression_ratio_threshold=2.4,
                                log_prob_threshold=-1.0,
                                no_speech_threshold=0.6,
                                vad_filter=True,
                                vad_parameters=vad_parameters,
                                word_timestamps=False,
                                initial_prompt=initial_prompt,
                            )
                        except Exception:
                            # Fallback: write chunk to temp WAV via pydub
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                                tmp_path = tf.name
                            try:
                                chunk = AudioSegment.from_file(audio_path)[start_ms:end_ms]
                                chunk = chunk.set_frame_rate(settings.sample_rate).set_channels(1)
                                chunk.export(tmp_path, format="wav")
                                segs, info = model_obj.transcribe(
                                    tmp_path,
                                    language=target_lang,
                                    beam_size=beam_size,
                                    temperature=temperature,
                                    condition_on_previous_text=False,
                                    compression_ratio_threshold=2.4,
                                    log_prob_threshold=-1.0,
                                    no_speech_threshold=0.6,
                                    vad_filter=True,
                                    vad_parameters=vad_parameters,
                                    word_timestamps=False,
                                    initial_prompt=initial_prompt,
                                )
                            finally:
                                try:
                                    os.unlink(tmp_path)
                                except Exception:
                                    pass
                    except Exception:
                        # On any ffmpeg error fallback to pydub temp file approach
                        chunk = AudioSegment.from_file(audio_path)[start_ms:end_ms]
                        chunk = chunk.set_frame_rate(settings.sample_rate).set_channels(1)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                            tmp_path = tf.name
                        try:
                            chunk.export(tmp_path, format="wav")
                            segs, info = model_obj.transcribe(
                                tmp_path,
                                language=target_lang,
                                beam_size=beam_size,
                                temperature=temperature,
                                condition_on_previous_text=False,
                                compression_ratio_threshold=2.4,
                                log_prob_threshold=-1.0,
                                no_speech_threshold=0.6,
                                vad_filter=True,
                                vad_parameters=vad_parameters,
                                word_timestamps=False,
                                initial_prompt=initial_prompt,
                            )
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                else:
                    chunk = AudioSegment.from_file(audio_path)[start_ms:end_ms]
                    chunk = chunk.set_frame_rate(settings.sample_rate).set_channels(1)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tmp_path = tf.name
                    try:
                        chunk.export(tmp_path, format="wav")
                        segs, info = model_obj.transcribe(
                            tmp_path,
                            language=target_lang,
                            beam_size=beam_size,
                            temperature=temperature,
                            condition_on_previous_text=False,
                            compression_ratio_threshold=2.4,
                            log_prob_threshold=-1.0,
                            no_speech_threshold=0.6,
                            vad_filter=True,
                            vad_parameters=vad_parameters,
                            word_timestamps=False,
                            initial_prompt=initial_prompt,
                        )
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                for s in segs:
                    s_start = s.start + offset
                    s_end = s.end + offset
                    if s_end <= last_end_global + 0.01:
                        continue
                    if s_start < last_end_global:
                        s_start = last_end_global
                    combined_segments.append(SimpleNamespace(start=s_start, end=s_end, text=s.text))
                    last_end_global = s_end

            return combined_segments, SimpleNamespace(duration=total_duration)

        # Apply runtime overrides from UI if provided
        if chunk_size_seconds is not None:
            settings.chunk_size_seconds = int(chunk_size_seconds)
        if chunk_overlap_seconds is not None:
            settings.chunk_overlap_seconds = int(chunk_overlap_seconds)

        # Prepare accumulators for streamed processing
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0
        paragraphs: list[list[str]] = [[]]
        diarized_parts: list[str] = []
        last_speaker: str | None = None
        last_end = 0.0
        all_segments = []

        if speed_optimized and use_chunking:
            chunk_ms = int(settings.chunk_size_seconds * 1000)
            overlap_ms = int(settings.chunk_overlap_seconds * 1000)
            step_ms = chunk_ms - overlap_ms if chunk_ms > overlap_ms else chunk_ms

            for start_ms in range(0, len(audio), step_ms):
                heartbeat()
                end_ms = min(start_ms + chunk_ms, len(audio))
                offset = start_ms / 1000.0

                # Respect timeout between chunks
                elapsed_now = time.time() - start_time
                remaining_time = max(0, timeout - elapsed_now)
                if elapsed_now > timeout:
                    logger.error(f"Transcription timed out after {timeout} seconds.")
                    _save_transcription_artifacts(result, all_segments, speaker_timeline, (txt_cache, srt_cache, vtt_cache))
                    yield astuple(
                        TranscriptionResult(
                            result=result,
                            txt_path=str(txt_cache),
                            srt_path=str(srt_cache),
                            vtt_path=str(vtt_cache),
                            status="❌ Hata: İşlem zaman aşımına uğradı.",
                            speaker_info=speaker_info,
                        )
                    )
                    return

                try:
                    if use_ffmpeg_piping:
                        # decode chunk via ffmpeg to numpy
                        cmd = [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-ss",
                            f"{start_ms/1000}",
                            "-to",
                            f"{end_ms/1000}",
                            "-i",
                            str(audio_path),
                            "-f",
                            "f32le",
                            "-acodec",
                            "pcm_f32le",
                            "-ac",
                            "1",
                            "-ar",
                            str(settings.sample_rate),
                            "-",
                        ]
                        import subprocess
                        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        out, err = p.communicate()
                        if p.returncode != 0:
                            raise RuntimeError(err.decode(errors="ignore"))
                        import numpy as _np

                        audio_array = _np.frombuffer(out, dtype=_np.float32)
                        segs, info = model.model.transcribe(
                            audio_array,
                            language=target_lang,
                            beam_size=current_beam_size,
                            temperature=temperature,
                            condition_on_previous_text=False,
                            compression_ratio_threshold=2.4,
                            log_prob_threshold=-1.0,
                            no_speech_threshold=0.6,
                            vad_filter=True,
                            vad_parameters=vad_parameters,
                            word_timestamps=False,
                            initial_prompt=initial_prompt,
                        )
                    else:
                        chunk = audio[start_ms:end_ms]
                        chunk = chunk.set_frame_rate(settings.sample_rate).set_channels(1)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                            tmp_path = tf.name
                        try:
                            chunk.export(tmp_path, format="wav")
                            segs, info = model.model.transcribe(
                                tmp_path,
                                language=target_lang,
                                beam_size=current_beam_size,
                                temperature=temperature,
                                condition_on_previous_text=False,
                                compression_ratio_threshold=2.4,
                                log_prob_threshold=-1.0,
                                no_speech_threshold=0.6,
                                vad_filter=True,
                                vad_parameters=vad_parameters,
                                word_timestamps=False,
                                initial_prompt=initial_prompt,
                            )
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                except Exception as e:
                    logger.exception("Chunk transcription failed, aborting chunked flow")
                    yield astuple(TranscriptionResult(result=f"❌ Bir hata oluştu: {e}", txt_path=None, srt_path=None, vtt_path=None, status="", speaker_info="", progress=0.0))
                    return

                # Process segments from this chunk
                for seg in segs:
                    heartbeat()
                    # timeout check inside per-segment processing as well
                    elapsed_now = time.time() - start_time
                    remaining_time = max(0, timeout - elapsed_now)
                    if elapsed_now > timeout:
                        logger.error(f"Transcription timed out after {timeout} seconds.")
                        _save_transcription_artifacts(result, all_segments, speaker_timeline, (txt_cache, srt_cache, vtt_cache))
                        yield astuple(
                            TranscriptionResult(
                                result=result,
                                txt_path=str(txt_cache),
                                srt_path=str(srt_cache),
                                vtt_path=str(vtt_cache),
                                status="❌ Hata: İşlem zaman aşımına uğradı.",
                                speaker_info=speaker_info,
                            )
                        )
                        return

                    s_start = seg.start + offset
                    s_end = seg.end + offset
                    if s_end <= last_end + 0.01:
                        continue
                    if s_start < last_end:
                        s_start = last_end
                    # create a normalized segment object
                    s_obj = SimpleNamespace(start=s_start, end=s_end, text=seg.text)
                    all_segments.append(s_obj)

                    if pause_event.is_set():
                        yield astuple(
                            TranscriptionResult(
                                result=result,
                                txt_path=None,
                                srt_path=None,
                                vtt_path=None,
                                status="⏸️ Duraklatıldı...",
                                speaker_info=speaker_info,
                            )
                        )
                        while pause_event.is_set() and not stop_event.is_set():
                            time.sleep(0.1)

                    gap = s_obj.start - last_end if s_obj.start > last_end else 0.0
                    last_end = s_obj.end

                    if speaker_timeline:
                        speaker = dominant_speaker(s_obj.start, s_obj.end, speaker_timeline)
                        text = s_obj.text.strip()
                        if speaker != last_speaker:
                            if diarized_parts:
                                diarized_parts.append("\n\n")
                            diarized_parts.append(f"[{speaker}]\n")
                            last_speaker = speaker
                        else:
                            if gap > settings.paragraph_pause:
                                diarized_parts.append("\n\n")
                            else:
                                diarized_parts.append(" ")
                        diarized_parts.append(text)
                        result = "".join(diarized_parts)
                    else:
                        if gap > settings.paragraph_pause:
                            paragraphs.append([])
                        paragraphs[-1].append(s_obj.text.strip())
                        result = "\n\n".join(" ".join(p) for p in paragraphs if p)

                    status = (
                        f"⏳ Çözümleniyor... {_fmt(s_obj.end)} / {_fmt(duration)} (Kalan süre: {remaining_time:.0f}s)"
                        if duration > 0
                        else f"⏳ Çözümleniyor... (Kalan süre: {remaining_time:.0f}s)"
                    )
                    progress_pct = (s_obj.end / duration * 100) if duration > 0 else 0

                    if stop_event.is_set():
                        current_text = result.strip()
                        _save_transcription_artifacts(current_text, all_segments, speaker_timeline, (txt_cache, srt_cache, vtt_cache))
                        yield astuple(
                            TranscriptionResult(
                                result=current_text,
                                txt_path=str(txt_cache),
                                srt_path=str(srt_cache),
                                vtt_path=str(vtt_cache),
                                status="⏹️ Durduruldu.",
                                speaker_info=speaker_info,
                                progress=progress_pct,
                            )
                        )
                        return

                    yield astuple(
                        TranscriptionResult(
                            result=result,
                            txt_path=None,
                            srt_path=None,
                            vtt_path=None,
                            status=status,
                            speaker_info=speaker_info,
                            progress=progress_pct,
                        )
                    )
            # end chunk loop
        else:
            # Non-chunked (original single-pass) flow
            segments, info = model.model.transcribe(
                str(audio_path),
                language=target_lang,
                beam_size=current_beam_size,
                temperature=temperature,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                vad_filter=True,
                vad_parameters=vad_parameters,
                word_timestamps=False,
                initial_prompt=initial_prompt,
            )

            # process segments from full-file transcription
            for seg in segments:
                heartbeat()
                elapsed_now = time.time() - start_time
                remaining_time = max(0, timeout - elapsed_now)
                if elapsed_now > timeout:
                    logger.error(f"Transcription timed out after {timeout} seconds.")
                    _save_transcription_artifacts(result, all_segments, speaker_timeline, (txt_cache, srt_cache, vtt_cache))
                    yield astuple(
                        TranscriptionResult(
                            result=result,
                            txt_path=str(txt_cache),
                            srt_path=str(srt_cache),
                            vtt_path=str(vtt_cache),
                            status="❌ Hata: İşlem zaman aşımına uğradı.",
                            speaker_info=speaker_info,
                        )
                    )
                    return

                all_segments.append(seg)
                if pause_event.is_set():
                    yield astuple(
                        TranscriptionResult(
                            result=result,
                            txt_path=None,
                            srt_path=None,
                            vtt_path=None,
                            status="⏸️ Duraklatıldı...",
                            speaker_info=speaker_info,
                        )
                    )
                    while pause_event.is_set() and not stop_event.is_set():
                        time.sleep(0.1)

                gap = seg.start - last_end if seg.start > last_end else 0.0
                last_end = seg.end

                if speaker_timeline:
                    speaker = dominant_speaker(seg.start, seg.end, speaker_timeline)
                    text = seg.text.strip()
                    if speaker != last_speaker:
                        if diarized_parts:
                            diarized_parts.append("\n\n")
                        diarized_parts.append(f"[{speaker}]\n")
                        last_speaker = speaker
                    else:
                        if gap > settings.paragraph_pause:
                            diarized_parts.append("\n\n")
                        else:
                            diarized_parts.append(" ")
                    diarized_parts.append(text)
                    result = "".join(diarized_parts)
                else:
                    if gap > settings.paragraph_pause:
                        paragraphs.append([])
                    paragraphs[-1].append(seg.text.strip())
                    result = "\n\n".join(" ".join(p) for p in paragraphs if p)

                status = (
                    f"⏳ Çözümleniyor... {_fmt(seg.end)} / {_fmt(duration)} (Kalan süre: {remaining_time:.0f}s)"
                    if duration > 0
                    else f"⏳ Çözümleniyor... (Kalan süre: {remaining_time:.0f}s)"
                )
                progress_pct = (seg.end / duration * 100) if duration > 0 else 0

                if stop_event.is_set():
                    current_text = result.strip()
                    _save_transcription_artifacts(
                        current_text, all_segments, speaker_timeline, (txt_cache, srt_cache, vtt_cache)
                    )
                    yield astuple(
                        TranscriptionResult(
                            result=current_text,
                            txt_path=str(txt_cache),
                            srt_path=str(srt_cache),
                            vtt_path=str(vtt_cache),
                            status="⏹️ Durduruldu.",
                            speaker_info=speaker_info,
                            progress=progress_pct,
                        )
                    )
                    return

                yield astuple(
                    TranscriptionResult(
                        result=result,
                        txt_path=None,
                        srt_path=None,
                        vtt_path=None,
                        status=status,
                        speaker_info=speaker_info,
                        progress=progress_pct,
                    )
                )

        elapsed = time.time() - start_time
        final_result = (
            "".join(diarized_parts) if speaker_timeline else "\n\n".join(" ".join(p) for p in paragraphs if p)
        )

        if not final_result:
            yield astuple(
                TranscriptionResult(
                    result="⚠️ Ses dosyasında konuşma algılanamadı.",
                    txt_path=None,
                    srt_path=None,
                    vtt_path=None,
                    status="",
                    speaker_info=speaker_info,
                    progress=100.0,
                )
            )
            return

        yield astuple(
            TranscriptionResult(
                result=final_result,
                txt_path=None,
                srt_path=None,
                vtt_path=None,
                status="⏳ Dosya kaydediliyor...",
                speaker_info=speaker_info,
                progress=95.0,
            )
        )

        _save_transcription_artifacts(final_result, all_segments, speaker_timeline, (txt_cache, srt_cache, vtt_cache))

        speed = f"{duration / elapsed:.1f}x" if elapsed > 0 else "—"
        stats = (
            f"✅ Tamamlandı!\n\n"
            f"**📊 İstatistikler**\n"
            f"- Model: {model.current_model_size} ({settings.device.upper()})\n"
            f"- Ses süresi: {duration:.1f} sn\n"
            f"- İşlem süresi: {elapsed:.1f} sn\n"
            f"- Hız: {speed}"
        )

        yield astuple(
            TranscriptionResult(
                result=final_result,
                txt_path=str(txt_cache),
                srt_path=str(srt_cache),
                vtt_path=str(vtt_cache),
                status=stats,
                speaker_info=speaker_info,
                progress=100.0,
            )
        )

    except Exception as e:
        logger.exception("Transkripsiyon işlemi sırasında beklenmedik hata")
        yield astuple(
            TranscriptionResult(
                result=f"❌ Bir hata oluştu: {str(e)}",
                txt_path=None,
                srt_path=None,
                vtt_path=None,
                status="",
                speaker_info="",
                progress=0.0,
            )
        )
    finally:
        set_processing_state(False)
        # Proactively clean up generated caches to protect ephemeral storage limits
        try:
            clean_cache_directories()
        except Exception:
            logger.exception("İşlem sonrası önbellek temizliği başarısız oldu")
