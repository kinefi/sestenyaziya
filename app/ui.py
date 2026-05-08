import gradio as gr

from . import models
from .config import (ModelSize, DEFAULT_MODEL_SIZE, device,
                     EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR, CACHE_BASE_DIR, DEFAULT_CACHE_SIZE_MB)
from .config import TRANSCRIPTION_TIMEOUT
from .transcription import transcribe
from .cache_utils import clear_all_cache, clean_embedding_cache, get_cache_size_mb

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
        "",                            # Clear detected speakers
        gr.update(interactive=False),  # enable_diarization
        gr.update(interactive=False),  # num_speakers_slider
    )

def on_diarization_change(enabled):
    return gr.update(visible=enabled)


def on_finish():
    return (
        gr.update(interactive=True),
        gr.update(interactive=False, value="⏸️ Duraklat"),
        gr.update(interactive=False),
        gr.update(interactive=True),   # enable_diarization
        gr.update(interactive=True),   # num_speakers_slider
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

def update_health_dashboard():
    stats = models.get_health_status()
    return f"""
    ### 🏥 Sistem Sağlığı
    | Parametre | Durum |
    | :--- | :--- |
    | **İşlem / Watchdog** | {stats['is_processing']} / {stats['watchdog_status']} |
    | **Model / Cihaz** | `{stats['model_loaded']}` / {stats['device']} |
    | **CUDA / Son Sinyal** | {stats['cuda_status']} / {stats['last_seen']} |
    | **Süre Sınırı / Kalan** | {TRANSCRIPTION_TIMEOUT}s / {stats['timeout_remaining']} |
    """

def handle_cache_cleanup(limit):
    clean_embedding_cache([EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR], max_size_mb=limit)
    return f"🧹 Temizlik yapıldı (Sınır: {limit} MB)", get_cache_status()


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

    with gr.Row():
        gr.Markdown(f"""
                    # 🎙️ Sesten Yazıya
                    Yapay zeka ile Türkçe ses kayıtlarını metne dönüştürün &nbsp;•&nbsp; {device.upper()}.
                    Sonuçlar anlık ekrana düşer. Model değiştirilirse ilk çalıştırmada yeniden yüklenir.
                    [Kaynak kodu inceleyebilirsiniz.](https://github.com/tekrei/sestenyaziya)
                    """)

    with gr.Row(equal_height=True):

        # 1) Audio file, model size, settings
        with gr.Column(scale=1, min_width=200):

            audio_input = gr.Audio(
                label="Ses Dosyası",
                type="filepath",
                sources=["upload"],
                elem_id="audio_input"
            )
            model_selector = gr.Dropdown(
                choices=ModelSize.values(),
                value=DEFAULT_MODEL_SIZE,
                label="Model  (small: hızlı · medium: dengeli · large-v3: kaliteli)",
                elem_id="model_selector"
            )

            enable_diarization = gr.Checkbox(
                label="Konuşmacıları ayırt et",
                value=False,
                elem_id="chk_enable_diarization"
            )
            with gr.Row(visible=False, elem_id="row_diarization_settings") as diarization_row:
                num_speakers_slider = gr.Slider(
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=0,
                    label="Konuşmacı sayısı (0 = otomatik algıla)",
                    elem_id="sld_num_speakers"
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
                elem_id="txt_output"
            )
            download_txt = gr.DownloadButton("📥 TXT İndir", interactive=False, elem_id="dl_txt")

        # 3) SRT preview
        with gr.Column(scale=1, min_width=200):
            preview_text = gr.Textbox(
                label="Dosya Önizleme (SRT/VTT)",
                placeholder="İşlem tamamlandığında SRT/VTT önizlemesi burada görünecek...",
                lines=20,
                interactive=False,
                elem_id="txt_preview"
            )
            with gr.Row():
                download_srt = gr.DownloadButton("📥 SRT İndir", interactive=False, elem_id="dl_srt")
                download_vtt = gr.DownloadButton("📥 VTT İndir", interactive=False, elem_id="dl_vtt")

        # 4) Health and Cache Settings (Rightmost Column)
        with gr.Column(scale=1, min_width=200):
            health_dashboard = gr.Markdown(update_health_dashboard())
            gr.Timer(5).tick(update_health_dashboard, outputs=health_dashboard)

            # --- Cache Management Pane ---
            with gr.Group():
                gr.Markdown("### ⚙️ Önbellek Yönetimi")
                cache_info_display = gr.Markdown(get_cache_status())
                cache_limit_slider = gr.Slider(
                    minimum=100, maximum=5000, step=100,
                    value=DEFAULT_CACHE_SIZE_MB,
                    label="Boyut Sınırı (MB)"
                )
                with gr.Row():
                    cleanup_btn = gr.Button("🧹 Temizle", scale=1, elem_id="btn_cleanup")
                    clear_btn = gr.Button("🗑️ Tümünü Sil", variant="stop", scale=1, elem_id="btn_clear")
                cache_mgmt_status = gr.Label(value="", label="Durum")

    with gr.Row():
        submit_btn = gr.Button("✨ Başlat", variant="primary", scale=2, elem_id="btn_submit")
        pause_btn = gr.Button("⏸️ Duraklat", interactive=False, scale=1, elem_id="btn_pause")
        stop_btn = gr.Button(
            "⏹️ Durdur", variant="stop", 
            interactive=False, scale=1,
            elem_id="btn_stop"
        )

    cleanup_btn.click(
        fn=handle_cache_cleanup, 
        inputs=[cache_limit_slider], 
        outputs=[cache_mgmt_status, cache_info_display]
    )
    clear_btn.click(
        fn=handle_clear_cache, 
        outputs=[cache_mgmt_status, cache_info_display]
    )

    enable_diarization.change(
        fn=on_diarization_change,
        inputs=[enable_diarization],
        outputs=[diarization_row],
    )

    # Event for loading new audio or clearing current one
    audio_input.change(
        fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
        outputs=[enable_diarization, num_speakers_slider],
        queue=False
    )

    btn_outputs = [submit_btn, pause_btn, stop_btn, enable_diarization, num_speakers_slider]

    (
        submit_btn.click(
            fn=on_start,
            outputs=[submit_btn, pause_btn, stop_btn, detected_speakers, enable_diarization, num_speakers_slider],
            queue=True,
        )
        .then(
            fn=transcribe,
            inputs=[audio_input, model_selector,
                    enable_diarization, num_speakers_slider],
            outputs=[output_text, download_txt, download_srt, download_vtt,
                     status_text, detected_speakers, preview_text],
            show_progress="hidden",
        )
        .then(fn=on_finish, outputs=btn_outputs, queue=False)
        .then(fn=lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)), 
              outputs=[download_txt, download_srt, download_vtt], queue=False)
    )
    stop_btn.click(fn=on_stop, outputs=btn_outputs, queue=False)
    pause_btn.click(fn=toggle_pause, outputs=[pause_btn])
