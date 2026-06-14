import logging
import time
from collections.abc import Generator
from dataclasses import astuple, dataclass

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

        current_beam_size = 1 if low_latency else 5

        logger.info(f"Transcription starting: model={model_size}, lang={language or 'auto'}, beam={current_beam_size}")

        # Passing an explicit language improves accuracy by preventing "language hallucination"
        # and speeds up the process by skipping the first 30-second detection phase.
        target_lang = language if language != "auto" else None

        segments, info = model.model.transcribe(
            str(audio_path),
            language=target_lang,
            beam_size=current_beam_size,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
            initial_prompt=("Türkçe konuşma kaydı. Noktalama ve büyük harf kullanımı." if language == "tr" else None),
        )

        duration = info.duration
        paragraphs: list[list[str]] = [[]]
        diarized_parts: list[str] = []
        last_speaker: str | None = None
        last_end = 0.0
        all_segments = []

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
                # Save whatever we have so far before exiting
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
