import logging
import tempfile
import time
from pathlib import Path
import gradio as gr
from dataclasses import dataclass, astuple
from typing import Generator

from . import config as cfg
from . import models
from .config import device, PARAGRAPH_PAUSE
from .diarization import diarize, dominant_speaker
from .cache_utils import get_transcription_hash

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    result: str
    txt_path: str | None
    srt_path: str | None
    vtt_path: str | None
    status: str
    speaker_info: str
    preview_srt: str = ""

cfg.TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
cfg.TEMP_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def _fmt(seconds: float, is_sub: bool = False, ms_sep: str = ",") -> str:
    if not is_sub:
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"
    
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // (1000 * 60)) % 60
    h = (total_ms // (1000 * 60 * 60))
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


def transcribe(
    audio_path: str,
    model_size: str,
    enable_diarization: bool,
    num_speakers: int,
    progress=gr.Progress(track_tqdm=True),
) -> Generator[tuple[str, str | None, str | None, str | None, str, str, str], None, None]:
    """Ses dosyasını Türkçe metne dönüştürür; akış olarak sonuç verir."""
    models.pause_event.clear()
    models.stop_event.clear()

    # Validate inputs before doing any expensive work.
    if audio_path is None:
        yield astuple(TranscriptionResult(
            result="⚠️ Lütfen bir ses dosyası yükleyin.", 
            txt_path=None, srt_path=None, vtt_path=None,
            status="", speaker_info=""
        ))
        return

    # ⚡ Check for cached transcription
    t_hash = get_transcription_hash(audio_path, model_size, enable_diarization, num_speakers)
    cache_file = cfg.TRANSCRIPT_CACHE_DIR / f"{t_hash}.txt"
    
    if cache_file.exists():
        logger.info(f"Loading transcription from cache: {cache_file}")
        cached_text = cache_file.read_text(encoding="utf-8")
        
        yield astuple(TranscriptionResult(
            result=cached_text,
            txt_path=str(cache_file),
            srt_path=None,
            vtt_path=None,
            status="✅ Önbellekten yüklendi!",
            speaker_info="⚡ İşlem atlandı"
        ))
        return

    if models.current_model_size != model_size or models.model is None:
        yield astuple(TranscriptionResult(
            result="", txt_path=None, srt_path=None, vtt_path=None,
            status="🔄 Model yükleniyor, lütfen bekleyin...", speaker_info=""
        ))
        models.load_model(model_size)

    if models.model is None:
        yield astuple(TranscriptionResult(
            result="❌ Hata: Model yüklenemedi. Logları kontrol edin.", 
            txt_path=None, srt_path=None, vtt_path=None,
            status="", speaker_info=""
        ))
        return

    result = ""
    try:
        start_time = time.time()

        diar_weight = 0.0
        speaker_timeline: list[tuple[float, float, str]] | None = None
        speaker_info = ""
        if enable_diarization:
            try:
                # Dynamic progress weight based on file size (MB)
                file_size_mb = Path(audio_path).stat().st_size / (1024 * 1024)
                diar_weight = max(0.1, min(0.4, file_size_mb / 50.0))

                yield astuple(TranscriptionResult(
                    result="", txt_path=None, srt_path=None, vtt_path=None,
                    status="🔍 Konuşmacılar analiz ediliyor...", speaker_info=""
                ))
                def dp(v, desc=""):
                    progress(v * diar_weight, desc=desc)
                speaker_timeline, used_cache = diarize(audio_path, int(num_speakers), progress=dp)
                n = len(set(lbl for _, _, lbl in speaker_timeline))
                cache_tag = " · ⚡ Önbellek" if used_cache else ""
                speaker_info = f"👥 {n} konuşmacı algılandı{cache_tag}"
            except Exception as e:
                logger.exception("Konuşmacı ayrıştırma sırasında hata oluştu")
                yield astuple(TranscriptionResult(
                    result="",
                    txt_path=None, srt_path=None, vtt_path=None,
                    status=f"⚠️ Konuşmacı ayrıştırma başarısız ({type(e).__name__}: {e}), düz metin devam ediliyor...",
                    speaker_info=""
                ))

        yield astuple(TranscriptionResult(
            result="", 
            txt_path=None, srt_path=None, vtt_path=None,
            status="⏳ Transkripsiyon başlatılıyor...", 
            speaker_info=speaker_info
        ))

        segments, info = models.model.transcribe(
            str(audio_path),
            language="tr",
            beam_size=1,
            vad_filter=True,
            word_timestamps=False,
        )

        duration = info.duration
        # paragraphs[i] holds the stripped segment texts for the i-th paragraph.
        # Using a list-of-lists avoids O(n²) string concatenation while streaming.
        paragraphs: list[list[str]] = [[]]
        # Parts for diarized result: list of (speaker, text) or just strings
        diarized_parts: list[str] = []
        last_speaker: str | None = None
        last_end = 0.0
        all_segments = []

        for seg in segments:
            all_segments.append(seg)
            if models.stop_event.is_set():
                yield astuple(TranscriptionResult(
                    result=result, txt_path=None, srt_path=None, vtt_path=None, 
                    status="⏹️ Durduruldu.", speaker_info=speaker_info))
                return

            if models.pause_event.is_set():
                yield astuple(TranscriptionResult(
                    result=result, txt_path=None, srt_path=None, vtt_path=None, 
                    status="⏸️ Duraklatıldı...", speaker_info=speaker_info))
                while models.pause_event.is_set() and not models.stop_event.is_set():
                    time.sleep(0.1)
            
            gap = seg.start - last_end if seg.start > last_end else 0.0
            last_end = seg.end

            if speaker_timeline:
                speaker = dominant_speaker(seg.start, seg.end, speaker_timeline)
                text = seg.text.strip()
                if speaker != last_speaker:
                    if diarized_parts:
                        diarized_parts.append("\n\n")
                    diarized_parts.append(f"{speaker}:\n")
                    last_speaker = speaker
                else:
                    if gap > PARAGRAPH_PAUSE:
                        diarized_parts.append("\n\n")
                    else:
                        diarized_parts.append(" ")
                diarized_parts.append(text)
                result = "".join(diarized_parts)
            else:
                if gap > PARAGRAPH_PAUSE:
                    paragraphs.append([])
                paragraphs[-1].append(seg.text.strip())
                result = "\n\n".join(" ".join(p) for p in paragraphs if p)

            status = (
                f"⏳ Çevriliyor... {_fmt(seg.end)} / {_fmt(duration)}"
                if duration > 0
                else "⏳ Çevriliyor..."
            )
            
            if duration > 0:
                p_val = diar_weight + (seg.end / duration) * (0.9 - diar_weight)
                progress(p_val, desc=f"Metne dönüştürülüyor... {_fmt(seg.end)} / {_fmt(duration)}")
                
            yield astuple(TranscriptionResult(
                result=result, 
                txt_path=None, srt_path=None, vtt_path=None,
                status=status, speaker_info=speaker_info))

        elapsed = time.time() - start_time
        final_result = (
            "".join(diarized_parts)
            if speaker_timeline
            else "\n\n".join(" ".join(p) for p in paragraphs if p)
        )

        if not final_result:
            yield astuple(TranscriptionResult(
                result="⚠️ Ses dosyasında konuşma algılanamadı.", 
                txt_path=None, srt_path=None, vtt_path=None,
                status="", speaker_info=speaker_info
            ))
            return

        progress(0.9, desc="⏳ Dosyalar kaydediliyor...")

        yield astuple(TranscriptionResult(
            result=final_result, 
            txt_path=None, srt_path=None, vtt_path=None,
            status="⏳ Dosya kaydediliyor...", speaker_info=speaker_info))

        def save_temp(content, suffix):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8", dir=cfg.TEMP_EXPORT_DIR
            ) as f:
                f.write(content)
                return f.name

        txt_path = save_temp(final_result, ".txt")
        srt_content = _generate_srt_vtt(all_segments, is_vtt=False, speaker_timeline=speaker_timeline)
        srt_path = save_temp(srt_content, ".srt")
        vtt_content = _generate_srt_vtt(all_segments, is_vtt=True, speaker_timeline=speaker_timeline)
        vtt_path = save_temp(vtt_content, ".vtt")

        speed = f"{duration / elapsed:.1f}x" if elapsed > 0 else "—"
        stats = (
            f"✅ Tamamlandı!\n\n"
            f"**📊 İstatistikler**\n"
            f"- Model: {models.current_model_size} ({device.upper()})\n"
            f"- Ses süresi: {duration:.1f} sn\n"
            f"- İşlem süresi: {elapsed:.1f} sn\n"
            f"- Hız: {speed}"
        )
        
        # ⚡ Save to cache
        try:
            cache_file.write_text(final_result, encoding="utf-8")
        except Exception:
            logger.exception("Failed to write transcript cache")
            
        progress(1.0, desc="✅ Tamamlandı")

        yield astuple(TranscriptionResult(
            result=final_result, 
            txt_path=txt_path,
            srt_path=srt_path,
            vtt_path=vtt_path,
            status=stats, 
            speaker_info=speaker_info,
            preview_srt=srt_content
        ))

    except Exception as e:
        logger.exception("Transkripsiyon işlemi sırasında beklenmedik hata")
        yield astuple(TranscriptionResult(
            result=f"❌ Bir hata oluştu: {str(e)}", 
            txt_path=None, srt_path=None, vtt_path=None,
            status="", speaker_info=""
        ))
