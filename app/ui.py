import gradio as gr

from . import models
from . import config as cfg
from .config import (ModelSize, DEFAULT_MODEL_SIZE, device,
                     EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR, CACHE_BASE_DIR, DEFAULT_CACHE_SIZE_MB)
from .transcription import transcribe
from .cache_utils import clear_all_cache, clean_embedding_cache, get_cache_size_mb
import logging

def toggle_pause():
    if models.pause_event.is_set():
        models.pause_event.clear()
        return gr.update(value="⏸️ Duraklat")
    else:
        models.pause_event.set()
        return gr.update(value="▶️ Devam Et")


def on_start():
    return (
        gr.update(interactive=False),  # submit_btn
        gr.update(interactive=True),   # pause_btn
        gr.update(interactive=True),   # stop_btn
        "",                            # detected_speakers — clear from previous run
        gr.update(interactive=False),  # download_txt
        gr.update(interactive=False),  # download_srt
        gr.update(interactive=False),  # download_vtt
        "",                            # preview_text — clear from previous run
    )


def on_finish():
    return (
        gr.update(interactive=True),
        gr.update(interactive=False, value="⏸️ Duraklat"),
        gr.update(interactive=False),
    )


def on_stop():
    models.stop_event.set()
    return on_finish()


def get_cache_status():
    size = get_cache_size_mb([EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR])
    return f"📊 Mevcut Önbellek: **{size:.2f} MB**"

def handle_clear_cache():
    clear_all_cache(CACHE_BASE_DIR)
    return "🗑️ Önbellek temizlendi.", get_cache_status()

def handle_cache_cleanup(limit):
    clean_embedding_cache([EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR], max_size_mb=limit)
    return f"🧹 Temizlik yapıldı (Sınır: {limit} MB)", get_cache_status()


_CSS = """
    .gradio-container { max-width: 1024px !important; margin: auto !important; }

    /* ── Remove block margins ────────────────────────────────── */
    .gradio-container .block {
        margin: 0 !important;
        padding: 5px !important;
        min-width: 0 !important;
    }

    /* ── Collapse flex gaps between components ───────────────── */
    .gradio-container .flex-col,
    .gradio-container .flex.flex-col { 
        gap: 2px !important; 
        min-width: 0 !important;
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

with gr.Blocks(title="Sesten Yazıya", css=_CSS) as demo:

    with gr.Row():
        gr.Markdown(f"""
                    # 🎙️ Sesten Yazıya
                    Yapay zeka ile Türkçe ses kayıtlarını metne dönüştürün &nbsp;•&nbsp; {device.upper()}.
                    Sonuçlar anlık ekrana düşer. Model değiştirilirse ilk çalıştırmada yeniden yüklenir.
                    [Kaynak kodu inceleyebilirsiniz.](https://github.com/tekrei/sestenyaziya)
                    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1):

            audio_input = gr.Audio(
                label="Ses Dosyası",
                type="filepath",
                sources=["upload"],
            )
            model_selector = gr.Dropdown(
                choices=ModelSize.values(),
                value=DEFAULT_MODEL_SIZE,
                label="Model  (small: hızlı · medium: dengeli · large-v3: kaliteli)",
            )

            enable_diarization = gr.Checkbox(
                label="Konuşmacıları ayırt et",
                value=False,
            )
            with gr.Row(visible=False) as diarization_row:
                num_speakers_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=1,
                    label="Konuşmacı sayısı (1 = otomatik algıla)",
                )

            with gr.Group():
                gr.Markdown("### ⚙️ Önbellek Yönetimi")
                cache_info_display = gr.Markdown(get_cache_status())
                cache_limit_slider = gr.Slider(
                    minimum=100, maximum=5000, step=100, 
                    value=DEFAULT_CACHE_SIZE_MB, 
                    label="Boyut Sınırı (MB)"
                )
                with gr.Row():
                    cleanup_btn = gr.Button("🧹 Temizle", scale=1)
                    clear_btn = gr.Button("🗑️ Tümünü Sil", variant="stop", scale=1)
                cache_mgmt_status = gr.Label(value="", label="Durum")

            cleanup_btn.click(
                fn=handle_cache_cleanup, 
                inputs=[cache_limit_slider], 
                outputs=[cache_mgmt_status, cache_info_display]
            )
            clear_btn.click(
                fn=handle_clear_cache, 
                outputs=[cache_mgmt_status, cache_info_display]
            )

            detected_speakers = gr.Markdown("")
            status_text = gr.Markdown("")

        with gr.Column(scale=1):

            output_text = gr.Textbox(
                label="Transkripsiyon Sonucu",
                placeholder="Sonuçlar konuşma algılandıkça buraya akacak...",
                lines=20,
                interactive=False,
            )
            preview_text = gr.Textbox(
                label="Dosya Önizleme (SRT/VTT)",
                lines=5,
                interactive=False,
            )

    with gr.Row():
        submit_btn = gr.Button("✨ Başlat", variant="primary", scale=2)
        pause_btn = gr.Button("⏸️ Duraklat", interactive=False, scale=1)
        stop_btn = gr.Button("⏹️ Durdur", variant="stop",
                             interactive=False, scale=1)

    with gr.Row():
        download_txt = gr.DownloadButton("📥 TXT İndir", interactive=False)
        download_srt = gr.DownloadButton("📥 SRT İndir", interactive=False)
        download_vtt = gr.DownloadButton("📥 VTT İndir", interactive=False)

    enable_diarization.change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[enable_diarization],
        outputs=[diarization_row],
    )

    btn_outputs = [submit_btn, pause_btn, stop_btn]

    (
        submit_btn.click(
            fn=on_start,
            outputs=[submit_btn, pause_btn, stop_btn,
                     detected_speakers, download_txt, download_srt, download_vtt, preview_text],
            queue=False,
        )
        .then(
            fn=transcribe,
            inputs=[audio_input, model_selector,
                    enable_diarization, num_speakers_slider],
            outputs=[output_text, download_txt, download_srt, download_vtt,
                     status_text, detected_speakers, preview_text],
        )
        .then(fn=on_finish, outputs=btn_outputs, queue=False)
        .then(fn=lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)), 
              outputs=[download_txt, download_srt, download_vtt], queue=False)
    )
    stop_btn.click(fn=on_stop, outputs=btn_outputs, queue=False)
    pause_btn.click(fn=toggle_pause, outputs=[pause_btn])
