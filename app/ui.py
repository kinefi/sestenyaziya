import gradio as gr

from . import models
from .config import ModelSize, DEFAULT_MODEL_SIZE, device
from .transcription import transcribe


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


_CSS = """
    footer { display: none !important; }
    .gradio-container { max-width: 1100px !important; margin: auto !important; }

    /* ── Remove block margins ────────────────────────────────── */
    .gradio-container .block {
        margin: 0 !important;
        padding: 4px 8px !important;
    }

    /* ── Collapse flex gaps between components ───────────────── */
    .gradio-container .flex-col,
    .gradio-container .flex.flex-col { gap: 2px !important; }

    /* Tailwind gap utilities used by Gradio */
    .gradio-container .gap-2 { gap: 2px !important; }
    .gradio-container .gap-3 { gap: 4px !important; }
    .gradio-container .gap-4 { gap: 4px !important; }

    /* ── Shrink label bottom spacing ─────────────────────────── */
    .gradio-container .label-wrap { padding-bottom: 1px !important; }
"""

with gr.Blocks(title="Sesten Yazıya") as demo:

    gr.HTML(f"""
        <style>{_CSS}</style>
        <div style="display:flex; align-items:center; justify-content:center; gap:10px;
                    padding: 8px 16px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; margin-bottom: 6px; color: white;">
            <span style="font-size:1.5rem;">🎙️</span>
            <div>
                <h1 style="font-size:1.15rem; font-weight:700; margin:0; line-height:1.3;">
                    Sesten Yazıya
                </h1>
                <p style="font-size:0.75rem; opacity:0.85; margin:0;">
                    Yapay zeka ile Türkçe ses kayıtlarını metne dönüştürün &nbsp;•&nbsp; {device.upper()}
                </p>
            </div>
        </div>
    """)

    with gr.Row(equal_height=True):

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

            gr.HTML("<hr style='margin:4px 0; border:none; border-top:1px solid #e5e7eb;'>")

            enable_diarization = gr.Checkbox(
                label="Konuşmacıları ayırt et",
                value=False,
            )
            num_speakers_slider = gr.Slider(
                minimum=1,
                maximum=20,
                step=1,
                value=1,
                label="Konuşmacı sayısı (1 = otomatik algıla)",
                visible=False,
            )
            detected_speakers = gr.Markdown("")
            status_text = gr.Markdown("")

            gr.HTML("""
                <div style="background:#f0fdf4; border:1px solid #86efac;
                            border-radius:6px; padding:5px 10px;">
                    <p style="margin:0; color:#166534; font-size:11px;">
                        💡 Sonuçlar anlık ekrana düşer.
                        Model değiştirilirse ilk çalıştırmada yeniden yüklenir.
                    </p>
                </div>
            """)

            with gr.Row():
                submit_btn = gr.Button("✨ Başlat", variant="primary", scale=2)
                pause_btn  = gr.Button("⏸️ Duraklat", interactive=False, scale=1)
                stop_btn   = gr.Button("⏹️ Durdur", variant="stop", interactive=False, scale=1)

        with gr.Column(scale=1):

            output_text = gr.Textbox(
                label="Transkripsiyon Sonucu",
                placeholder="Sonuçlar konuşma algılandıkça buraya akacak...",
                lines=10,
                interactive=False,
            )
            with gr.Row():
                download_file = gr.File(label="📥 İndir (.txt)", scale=1, min_width=0)

    enable_diarization.change(
        fn=lambda enabled: gr.update(visible=enabled),
        inputs=[enable_diarization],
        outputs=[num_speakers_slider],
    )

    btn_outputs = [submit_btn, pause_btn, stop_btn]

    (
        submit_btn.click(
            fn=on_start,
            outputs=[submit_btn, pause_btn, stop_btn, detected_speakers],
            queue=False,
        )
        .then(
            fn=transcribe,
            inputs=[audio_input, model_selector, enable_diarization, num_speakers_slider],
            outputs=[output_text, download_file, status_text, detected_speakers],
        )
        .then(fn=on_finish, outputs=btn_outputs, queue=False)
    )
    stop_btn.click(fn=on_stop, outputs=btn_outputs, queue=False)
    pause_btn.click(fn=toggle_pause, outputs=[pause_btn])
