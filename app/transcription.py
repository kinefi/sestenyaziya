import tempfile
import time
import traceback
from typing import Generator

from . import models
from .config import device, PARAGRAPH_PAUSE
from .diarization import diarize, dominant_speaker

_YieldType = tuple[str, str | None, str, str]


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def transcribe(
    audio_path: str,
    model_size: str,
    enable_diarization: bool,
    num_speakers: int,
) -> Generator[_YieldType, None, None]:
    """Ses dosyasını Türkçe metne dönüştürür; akış olarak sonuç verir."""
    models.pause_event.clear()
    models.stop_event.clear()

    # Validate inputs before doing any expensive work.
    if audio_path is None:
        yield "⚠️ Lütfen bir ses dosyası yükleyin.", None, "", ""
        return

    if models.current_model_size != model_size or models.model is None:
        yield "", None, "🔄 Model yükleniyor, lütfen bekleyin...", ""
        models.load_model(model_size)

    if models.model is None:
        yield "❌ Hata: Model yüklenemedi. Logları kontrol edin.", None, "", ""
        return

    try:
        start_time = time.time()

        speaker_timeline: list[tuple[float, float, str]] | None = None
        speaker_info = ""
        if enable_diarization:
            yield "", None, "🔍 Konuşmacılar analiz ediliyor...", ""
            try:
                speaker_timeline = diarize(audio_path, int(num_speakers))
                n = len(set(lbl for _, _, lbl in speaker_timeline))
                speaker_info = f"👥 {n} konuşmacı algılandı"
            except Exception as e:
                traceback.print_exc()
                yield "", None, (
                    f"⚠️ Konuşmacı ayrıştırma başarısız ({type(e).__name__}: {e}), "
                    "düz metin olarak devam ediliyor..."
                ), ""

        yield "", None, "⏳ Transkripsiyon başlatılıyor...", speaker_info

        segments, info = models.model.transcribe(
            audio_path,
            language="tr",
            beam_size=1,
            vad_filter=True,
            word_timestamps=False,
        )

        duration = info.duration
        # paragraphs[i] holds the stripped segment texts for the i-th paragraph.
        # Using a list-of-lists avoids O(n²) string concatenation while streaming.
        paragraphs: list[list[str]] = [[]]
        stream_result = ""
        last_speaker: str | None = None
        last_end = 0.0
        result = ""

        for seg in segments:
            if models.stop_event.is_set():
                yield result, None, "⏹️ Durduruldu.", speaker_info
                return

            if models.pause_event.is_set():
                yield result, None, "⏸️ Duraklatıldı...", speaker_info
                while models.pause_event.is_set() and not models.stop_event.is_set():
                    time.sleep(0.1)

            if models.stop_event.is_set():
                yield result, None, "⏹️ Durduruldu.", speaker_info
                return

            gap = seg.start - last_end
            last_end = seg.end

            if speaker_timeline:
                speaker = dominant_speaker(seg.start, seg.end, speaker_timeline)
                text = seg.text.strip()
                if speaker == last_speaker:
                    if gap > PARAGRAPH_PAUSE:
                        stream_result = stream_result.rstrip() + "\n\n" + text
                    else:
                        stream_result = stream_result.rstrip() + " " + text
                else:
                    stream_result = (
                        (stream_result + "\n\n") if stream_result else ""
                    ) + f"{speaker}:\n{text}"
                    last_speaker = speaker
                result = stream_result
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
            yield result, None, status, speaker_info

        elapsed = time.time() - start_time
        final_result = (
            stream_result
            if speaker_timeline
            else "\n\n".join(" ".join(p) for p in paragraphs if p)
        )

        if not final_result:
            yield "⚠️ Ses dosyasında konuşma algılanamadı.", None, "", speaker_info
            return

        yield final_result, None, "⏳ Dosya kaydediliyor...", speaker_info

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(final_result)

        speed = f"{duration / elapsed:.1f}x" if elapsed > 0 else "—"
        stats = (
            f"✅ Tamamlandı!\n\n"
            f"**📊 İstatistikler**\n"
            f"- Model: {models.current_model_size} ({device.upper()})\n"
            f"- Ses süresi: {duration:.1f} sn\n"
            f"- İşlem süresi: {elapsed:.1f} sn\n"
            f"- Hız: {speed}"
        )
        yield final_result, f.name, stats, speaker_info

    except Exception as e:
        traceback.print_exc()
        yield f"❌ Bir hata oluştu: {str(e)}", None, "", ""
