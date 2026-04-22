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
        gr.update(interactive=False),  # download_file
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
    .gradio-container { max-width: 1024px !important; margin: auto !important; }

    /* ── Remove block margins ────────────────────────────────── */
    .gradio-container .block {
        margin: 0 !important;
        padding: 5px !important;
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

with gr.Blocks(title="Sesten Yazıya", css=_CSS) as demo:

    with gr.Row():
        gr.Markdown(f"""
                    # 🎙️ Sesten Yazıya
                    Yapay zeka ile Türkçe ses kayıtlarını metne dönüştürün &nbsp;•&nbsp; {device.upper()}.
                    Sonuçlar anlık ekrana düşer. Model değiştirilirse ilk çalıştırmada yeniden yüklenir.
                    [Kaynak kodu inceleyebilirsiniz.](https://github.com/tekrei/sestenyaziya)
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
            detected_speakers = gr.Markdown("")
            status_text = gr.Markdown("")

        with gr.Column(scale=1):

            output_text = gr.Textbox(
                label="Transkripsiyon Sonucu",
                placeholder="Sonuçlar konuşma algılandıkça buraya akacak...",
                lines=20,
                interactive=False,
            )

    with gr.Row():
        submit_btn = gr.Button("✨ Başlat", variant="primary", scale=2)
        pause_btn = gr.Button("⏸️ Duraklat", interactive=False, scale=1)
        stop_btn = gr.Button("⏹️ Durdur", variant="stop",
                             interactive=False, scale=1)
        download_file = gr.DownloadButton(
            label="📥 İndir (.txt)", interactive=False)

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
                     detected_speakers, download_file],
            queue=False,
        )
        .then(
            fn=transcribe,
            inputs=[audio_input, model_selector,
                    enable_diarization, num_speakers_slider],
            outputs=[output_text, download_file,
                     status_text, detected_speakers],
        )
        .then(fn=on_finish, outputs=btn_outputs, queue=False)
        .then(fn=lambda: gr.update(interactive=True), outputs=[download_file], queue=False)
    )
    stop_btn.click(fn=on_stop, outputs=btn_outputs, queue=False)
    pause_btn.click(fn=toggle_pause, outputs=[pause_btn])
