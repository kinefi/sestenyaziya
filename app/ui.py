import logging
import os

import gradio as gr

from . import models
from .cache_utils import (
    get_cache_size_mb,
)
from .config import (
    DEFAULT_MODEL_SIZE,
    EMBEDDING_CACHE_DIR,
    MODELS_DIR,
    SUPPORTED_LANGUAGES,
    TRANSCRIPT_CACHE_DIR,
    TRANSCRIPTION_TIMEOUT,
    ModelSize,
    device,
)
from .transcription import transcribe

logger = logging.getLogger(__name__)


def toggle_pause():
    if models.pause_event.is_set():
        models.pause_event.clear()
        return gr.update(value="⏸️ Duraklat")
    else:
        models.pause_event.set()
        return gr.update(value="▶️ Devam Et")


def handle_audio_change(new_path, old_path):
    """Yeni bir dosya yüklendiğinde veya temizlendiğinde eskisini diskten siler."""
    if old_path and old_path != new_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except Exception:
            pass
    return new_path


def clear_outputs():
    """Yeni bir dosya yüklendiğinde veya temizlendiğinde eski sonuç alanlarını sıfırlar."""
    return (
        "",  # output_text
        "",  # status_text
        "",  # detected_speakers
        0,  # progress_bar
        gr.update(interactive=False),  # session_clear_btn
        gr.update(value=None, interactive=False),  # download_txt
        gr.update(value=None, interactive=False),  # download_srt
        gr.update(value=None, interactive=False),  # download_vtt
    )


def on_start():
    return (
        gr.update(interactive=False),  # submit_btn
        gr.update(interactive=True),  # pause_btn
        gr.update(interactive=True),  # stop_btn
        "",  # detected_speakers
        gr.update(interactive=False),  # enable_diarization
        gr.update(interactive=False),  # num_speakers_slider
        gr.update(visible=False),  # copy_btn
        gr.update(interactive=False),  # session_clear_btn
        gr.update(interactive=False),  # low_latency_chk
        gr.update(interactive=False),  # timeout_slider
        gr.update(interactive=False),  # audio_input
        gr.update(interactive=False),  # model_selector
        gr.update(interactive=False),  # language_selector
        gr.update(interactive=False),  # download_txt
        gr.update(interactive=False),  # download_srt
        gr.update(interactive=False),  # download_vtt
        0,  # progress_bar reset
    )


def on_diarization_change(enabled):
    return gr.update(visible=enabled)


def on_finish():
    return (
        gr.update(interactive=True),
        gr.update(interactive=False, value="⏸️ Duraklat"),
        gr.update(interactive=False),
        gr.update(interactive=True),  # enable_diarization
        gr.update(interactive=True),  # num_speakers_slider
        gr.update(interactive=True),  # low_latency_chk
        gr.update(interactive=True),  # timeout_slider
        gr.update(visible=True),  # copy_btn
        gr.update(interactive=True),  # session_clear_btn
        gr.update(interactive=True),  # audio_input
        gr.update(interactive=True),  # model_selector
        gr.update(interactive=True),  # language_selector
        gr.update(interactive=True),  # download_txt
        gr.update(interactive=True),  # download_srt
        gr.update(interactive=True),  # download_vtt
    )


def on_stop():
    models.stop_event.set()
    return on_finish()


def _format_size(size_mb: float) -> str:
    """Formats size in MB to either MB or GB string."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    return f"{size_mb:.1f} MB"


def handle_session_cleanup(audio_path, txt_path, srt_path, vtt_path):
    """Mevcut oturum dosyasını ve üretilen çıktıları siler, UI'ı sıfırlar."""
    # If all paths are None (user cancelled the JS prompt), skip the update
    if not any([audio_path, txt_path, srt_path, vtt_path]):
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "",
        )

    files_to_delete = [audio_path, txt_path, srt_path, vtt_path]
    for p in files_to_delete:
        if p and isinstance(p, str) and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    logger.info(f"Oturum temizlendi. Silinen dosya sayısı: {len(files_to_delete)}")
    return (
        None,  # audio_input
        "",  # output_text
        "",  # status_text
        "",  # detected_speakers
        0,  # progress_bar
        gr.update(interactive=False),  # session_clear_btn
        gr.update(value=None, interactive=False),  # download_txt
        gr.update(value=None, interactive=False),  # download_srt
        gr.update(value=None, interactive=False),  # download_vtt
        "🗑️ Oturum verileri ve geçici dosyalar temizlendi.",
    )


def update_health_dashboard():
    stats = models.get_health_status()
    transcription_cache = get_cache_size_mb([EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR])
    model_cache = get_cache_size_mb([MODELS_DIR])

    time_display = stats["timeout_remaining"]
    # Apply red highlighting if time is below 60 seconds
    if stats.get("remaining_raw") is not None and stats["remaining_raw"] < 60:
        time_display = f"<span style='color: #ff4b4b; font-weight: bold;'>{time_display} ⚠️</span>"

    return f"""
    ### 🏥 Sistem Sağlığı
    | Parametre | Durum |
    | :--- | :--- |
    | **İşlem / Watchdog** | {stats['is_processing']} / {stats['watchdog_status']} |
    | **Whisper / Encoder** | `{stats['model_loaded']}` / `{stats['encoder_loaded']}` |
    | **Cihaz / CUDA** | `{stats['device']}` / {stats['cuda_status']} |
    | **Son Sinyal** | {stats['last_seen']} |
    | **Süre Sınırı / Kalan** | {stats['active_timeout']}s / {time_display} |
    | **Veri Önbelleği** | {_format_size(transcription_cache)} |
    | **Model Önbelleği** | {_format_size(model_cache)} |
    """


def update_char_count(text: str) -> str:
    """Calculates and returns the character count of the given text."""
    count = len(text) if text else 0
    return f"📝 Karakter sayısı: **{count}**"


def sync_copy_button_on_load(text: str):
    """Sayfa yüklendiğinde çıktı kutusunda metin varsa kopyalama butonunu görünür yapar."""
    return gr.update(visible=bool(text and text.strip()))


UI_CSS = """
    .gradio-container { max-width: 95% !important; margin: auto !important; }

    /* ── Remove block margins ────────────────────────────────── */
    .gradio-container .block {
        margin: 0 !important;
        padding: 5px !important;
    }

    /* ── Collapse flex gaps between components ───────────────── */
    .gradio-container .flex-col,
    .gradio-container .flex.flex-col { 
        gap: 2px !important; 
    }

    /* ── Remove row margins to bring buttons up ──────────────── */
    .gradio-container .row, 
    .gradio-container .flex.row { 
        margin-top: 0 !important; 
    }

    /* Tailwind gap utilities used by Gradio */
    .gradio-container .gap-2 { gap: 2px !important; }
    .gradio-container .gap-3 { gap: 4px !important; }
    .gradio-container .gap-4 { gap: 4px !important; }

    /* ── Shrink label bottom spacing ─────────────────────────── */
    .gradio-container .label-wrap { padding-bottom: 1px !important; }
"""

with gr.Blocks(title="Sesten Yazıya") as demo:
    last_audio_path = gr.State("")
    # Persistent Session ID via Browser localStorage
    session_id = gr.Textbox(visible=False, elem_id="session_id_storage")

    # Logic to load session_id from localStorage or create a new one
    demo.load(
        fn=None,
        js="""
        () => {
            let sid = localStorage.getItem('sestenyaziya_session_id');
            if (!sid) {
                sid = 'sess_' + (window.crypto && window.crypto.randomUUID 
                    ? window.crypto.randomUUID() 
                    : Math.random().toString(36).substring(2, 15));
                localStorage.setItem('sestenyaziya_session_id', sid);
            }
            return sid;
        }
        """,
        outputs=[session_id],
    )

    with gr.Row():
        gr.Markdown(f"""
                    # 🎙️ Sesten Yazıya
                    Yapay zeka ile Türkçe ses kayıtlarını metne dönüştürün &nbsp;•&nbsp; {device.upper()}.
                    Sonuçlar anlık ekrana düşer. Model değiştirilirse ilk çalıştırmada yeniden yüklenir.
                    [Kaynak kodu inceleyebilirsiniz.](https://github.com/kinefi/sestenyaziya)
                    """)

    with gr.Row(equal_height=True):
        # 1) Audio file, model size, settings
        with gr.Column(scale=1, min_width=200):
            audio_input = gr.Audio(
                label="Ses Dosyası (Yeni dosya yüklemek önceki sonuçları siler)",
                type="filepath",
                sources=["upload"],
                elem_id="audio_input",
            )
            model_selector = gr.Dropdown(
                choices=ModelSize.values(),
                value=DEFAULT_MODEL_SIZE,
                label=("Model Seçimi (" "small: ~480MB · " "medium: ~1.5GB · " "large-v3: ~3.0GB)"),
                elem_id="model_selector",
            )
            language_selector = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="tr",
                label="Dil Seçimi",
                elem_id="language_selector",
            )

            enable_diarization = gr.Checkbox(
                label="Konuşmacıları ayırt et",
                value=False,
                elem_id="chk_enable_diarization",
            )
            low_latency_chk = gr.Checkbox(
                label="🚀 Turbo Mod (Düşük Gecikmeli Transkripsiyon)",
                value=False,
                elem_id="chk_low_latency",
            )
            timeout_slider = gr.Slider(
                minimum=300,
                maximum=7200,
                step=300,
                value=TRANSCRIPTION_TIMEOUT,
                label="Maksimum İşlem Süresi (Saniye)",
                elem_id="sld_timeout",
            )
            with gr.Row(visible=False, elem_id="row_diarization_settings") as diarization_row:
                num_speakers_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=0,
                    label="Konuşmacı sayısı (0 = otomatik algıla)",
                    elem_id="sld_num_speakers",
                )

            detected_speakers = gr.Markdown("", elem_id="md_speakers")
            status_text = gr.Markdown("", elem_id="md_status")

        # 2) Transcription result
        with gr.Column(scale=1, min_width=200):
            output_text = gr.Textbox(
                label="Transkripsiyon Sonucu",
                placeholder="Sonuçlar konuşma algılandıkça buraya akacak...",
                lines=20,
                interactive=False,
                elem_id="txt_output",
            )
            char_counter = gr.Markdown("📝 Karakter sayısı: **0**", elem_id="char_counter")
            progress_bar = gr.Slider(
                label="İşlem İlerlemesi (%)",
                minimum=0,
                maximum=100,
                value=0,
                interactive=False,
                elem_id="progress_bar",
            )
            with gr.Row():
                copy_btn = gr.Button("📋 Metni Kopyala", visible=False, elem_id="btn_copy")
            with gr.Row():
                download_txt = gr.DownloadButton("📥 TXT", interactive=False, elem_id="dl_txt")
                download_srt = gr.DownloadButton("📥 SRT İndir", interactive=False, elem_id="dl_srt")
                download_vtt = gr.DownloadButton("📥 VTT İndir", interactive=False, elem_id="dl_vtt")

        # 3) Health and Cache Settings (Rightmost Column)
        with gr.Column(scale=1, min_width=200):
            health_dashboard = gr.Markdown(update_health_dashboard())
            gr.Timer(5).tick(update_health_dashboard, outputs=health_dashboard)

            session_clear_btn = gr.Button("🗑️ Oturumu Temizle", interactive=False, variant="secondary")

            cache_mgmt_status = gr.Label(value="", label="İşlem Durumu", visible=False)

    with gr.Row():
        submit_btn = gr.Button("✨ Başlat", variant="primary", scale=2, elem_id="btn_submit")
        pause_btn = gr.Button("⏸️ Duraklat", interactive=False, scale=1, elem_id="btn_pause")
        stop_btn = gr.Button(
            "⏹️ Durdur",
            variant="stop",
            interactive=False,
            scale=1,
            elem_id="btn_stop",
        )

    enable_diarization.change(
        fn=on_diarization_change,
        inputs=[enable_diarization],
        outputs=[diarization_row],
    )

    # Event for loading new audio or clearing current one
    audio_input.change(handle_audio_change, [audio_input, last_audio_path], [last_audio_path])
    audio_input.change(
        fn=lambda: (gr.update(interactive=True), gr.update(interactive=True), *clear_outputs()),
        outputs=[
            enable_diarization,
            num_speakers_slider,
            output_text,
            status_text,
            detected_speakers,
            progress_bar,
            session_clear_btn,
            download_txt,
            download_srt,
            download_vtt,
        ],
        queue=False,
    )

    btn_outputs = [
        submit_btn,
        pause_btn,
        stop_btn,
        enable_diarization,
        num_speakers_slider,
        low_latency_chk,
        timeout_slider,
        copy_btn,
        session_clear_btn,
        audio_input,
        model_selector,
        language_selector,
        download_txt,
        download_srt,
        download_vtt,
    ]

    # 7 Outputs to match the new TranscriptionResult dataclass
    transcribe_outputs = [
        output_text,
        download_txt,
        download_srt,
        download_vtt,
        status_text,
        detected_speakers,
        progress_bar,
    ]

    (
        submit_btn.click(
            fn=on_start,
            outputs=[
                submit_btn,
                pause_btn,
                stop_btn,
                detected_speakers,
                enable_diarization,
                num_speakers_slider,
                copy_btn,
                session_clear_btn,
                low_latency_chk,
                timeout_slider,
                audio_input,
                model_selector,
                language_selector,
                download_txt,
                download_srt,
                download_vtt,
                progress_bar,
            ],
            queue=True,
        )
        .then(
            fn=transcribe,
            inputs=[
                audio_input,
                model_selector,
                enable_diarization,
                num_speakers_slider,
                session_id,
                timeout_slider,
                language_selector,
                low_latency_chk,
            ],
            outputs=transcribe_outputs,
            show_progress="full",
        )
        .then(fn=on_finish, outputs=btn_outputs, queue=False)
        .then(
            fn=None,
            js="""
            (txt_path) => {
                if (txt_path && typeof txt_path === 'string' && 
                    confirm("Transkripsiyon tamamlandı. TXT dosyasını şimdi indirmek ister misiniz?")) {
                        const link = document.createElement('a');
                        link.href = window.location.origin + '/file=' + txt_path;
                        link.download = txt_path.split('/').pop();
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                }
            }
            """,
            inputs=[download_txt],
        )
    )
    stop_btn.click(fn=on_stop, outputs=btn_outputs, queue=False)

    session_clear_btn.click(
        fn=handle_session_cleanup,
        js="""
        (audio, txt, srt, vtt) => {
            const msg = "Oturum verilerini ve üretilen tüm dosyaları kalıcı olarak silmek istediğinize emin misiniz?";
            if (!confirm(msg)) {
                return [null, null, null, null];
            }
            navigator.clipboard.writeText("");
            return [audio, txt, srt, vtt];
        }
        """,
        inputs=[last_audio_path, download_txt, download_srt, download_vtt],
        outputs=[
            audio_input,
            output_text,
            status_text,
            detected_speakers,
            progress_bar,
            session_clear_btn,
            download_txt,
            download_srt,
            download_vtt,
            cache_mgmt_status,
        ],
    )

    pause_btn.click(fn=toggle_pause, outputs=[pause_btn])

    # Copy to clipboard logic via Browser JS
    copy_btn.click(
        fn=None,
        js="(text) => { navigator.clipboard.writeText(text); }",
        inputs=[output_text],
        queue=False,
    )

    # Live character counter
    output_text.change(
        fn=update_char_count,
        inputs=[output_text],
        outputs=[char_counter],
        queue=False,
    )

    # Automatic cleanup when the user closes the browser tab or the session expires
    demo.unload(
        fn=lambda: logger.info("Oturum bağlantısı kesildi (Tarayıcı sekmesi kapatıldı)."),
    )

    # Sayfa yenilendiğinde sonuç kutusunda metin varsa kopyalama butonunu göster
    demo.load(
        fn=sync_copy_button_on_load,
        inputs=[output_text],
        outputs=[copy_btn],
        queue=False,
    )
