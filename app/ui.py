import logging
import os

import gradio as gr

from .cache_utils import (
    delete_session_cache,
    get_cache_size_mb,
    get_models_size_mb,
)
from .config import (
    SUPPORTED_LANGUAGES,
    ModelSize,
    settings,
)
from .transcription import transcribe
from .watchdog import get_health_status, pause_event, stop_event

logger = logging.getLogger(__name__)


def _safe_remove(path: str | None):
    """Helper to safely delete a file if it exists."""
    if path and isinstance(path, str) and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            logger.warning(f"Dosya silinirken hata oluştu ({path}): {e}")


def _format_size(size_mb: float) -> str:
    """Formats size in MB to either MB or GB string."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    return f"{size_mb:.1f} MB"


def update_health_dashboard():
    stats = get_health_status()
    total_cache_size = get_cache_size_mb()
    model_cache_size = get_models_size_mb()

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
    | **Toplam Önbellek** | {_format_size(total_cache_size)} |
    | **Model Klasörü** | {_format_size(model_cache_size)} |
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
                    Yapay zeka ile Türkçe ses kayıtlarını metne dönüştürün &nbsp;•&nbsp; {settings.device.upper()}.
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
                value=settings.default_model_size,
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
                value=settings.transcription_timeout,
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
            progress_bar = gr.Number(visible=False, value=0)
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

    # --- UI Logic Functions (Dictionary Returns) ---

    def on_diarization_change(enabled):
        return {diarization_row: gr.update(visible=enabled)}

    def handle_audio_update(new_path, last_path):
        if last_path and last_path != new_path:
            _safe_remove(last_path)
        has_audio = new_path is not None
        return {
            last_audio_path: new_path,
            submit_btn: gr.update(interactive=has_audio),
            enable_diarization: gr.update(interactive=has_audio),
            num_speakers_slider: gr.update(interactive=has_audio),
            output_text: "",
            status_text: "",
            detected_speakers: "",
            progress_bar: 0,
            session_clear_btn: gr.update(interactive=False),
            download_txt: gr.update(value=None, interactive=False),
            download_srt: gr.update(value=None, interactive=False),
            download_vtt: gr.update(value=None, interactive=False),
        }

    def on_start():
        return {
            submit_btn: gr.update(interactive=False),
            pause_btn: gr.update(interactive=True, value="⏸️ Duraklat"),
            stop_btn: gr.update(interactive=True),
            detected_speakers: "",
            enable_diarization: gr.update(interactive=False),
            num_speakers_slider: gr.update(interactive=False),
            copy_btn: gr.update(visible=False),
            session_clear_btn: gr.update(interactive=False),
            low_latency_chk: gr.update(interactive=False),
            timeout_slider: gr.update(interactive=False),
            audio_input: gr.update(interactive=False),
            model_selector: gr.update(interactive=False),
            language_selector: gr.update(interactive=False),
            download_txt: gr.update(interactive=False),
            download_srt: gr.update(interactive=False),
            download_vtt: gr.update(interactive=False),
            progress_bar: 0,
        }

    def on_finish():
        return {
            submit_btn: gr.update(interactive=True),
            pause_btn: gr.update(interactive=False, value="⏸️ Duraklat"),
            stop_btn: gr.update(interactive=False),
            enable_diarization: gr.update(interactive=True),
            num_speakers_slider: gr.update(interactive=True),
            low_latency_chk: gr.update(interactive=True),
            timeout_slider: gr.update(interactive=True),
            copy_btn: gr.update(visible=True),
            session_clear_btn: gr.update(interactive=True),
            audio_input: gr.update(interactive=True),
            model_selector: gr.update(interactive=True),
            language_selector: gr.update(interactive=True),
            download_txt: gr.update(interactive=True),
            download_srt: gr.update(interactive=True),
            download_vtt: gr.update(interactive=True),
        }

    def on_stop():
        stop_event.set()
        return on_finish()

    def toggle_pause():
        if pause_event.is_set():
            pause_event.clear()
            return {pause_btn: gr.update(value="⏸️ Duraklat")}
        else:
            pause_event.set()
            return {pause_btn: gr.update(value="▶️ Devam Et")}

    def handle_session_cleanup(session_id, audio_path, txt_path, srt_path, vtt_path):
        if session_id:
            delete_session_cache(session_id)

        if not any([session_id, audio_path, txt_path, srt_path, vtt_path]):
            return {}
        for p in [audio_path, txt_path, srt_path, vtt_path]:
            _safe_remove(p)
        logger.info(f"Oturum verileri ve geçici dosyalar temizlendi: {session_id}")
        return {
            audio_input: None,
            output_text: "",
            status_text: "",
            detected_speakers: "",
            progress_bar: 0,
            session_clear_btn: gr.update(interactive=False),
            download_txt: gr.update(value=None, interactive=False),
            download_srt: gr.update(value=None, interactive=False),
            download_vtt: gr.update(value=None, interactive=False),
            cache_mgmt_status: "🗑️ Oturum verileri ve geçici dosyalar temizlendi.",
        }

    # --- Event Handlers ---

    # Shared output lists for dictionary returns to ensure UI synchronization
    audio_update_outputs = [
        last_audio_path,
        submit_btn,
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
    ]

    ui_state_outputs = [
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
        progress_bar,
        detected_speakers,
    ]

    cleanup_outputs = [
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
    ]

    enable_diarization.change(
        fn=on_diarization_change,
        inputs=[enable_diarization],
        outputs=[diarization_row],
    )

    audio_input.change(
        fn=handle_audio_update,
        inputs=[audio_input, last_audio_path],
        outputs=audio_update_outputs,
    )

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
            outputs=ui_state_outputs,
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
        .then(fn=on_finish, outputs=ui_state_outputs, queue=False)
        .then(
            fn=None,
            js="""
            (txt_path, progress) => {
                if (txt_path && typeof txt_path === 'string' && progress >= 100 &&
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
            inputs=[download_txt, progress_bar],
        )
    )
    stop_btn.click(fn=on_stop, outputs=ui_state_outputs, queue=False)

    session_clear_btn.click(
        fn=handle_session_cleanup,
        js="""
        (sid, audio, txt, srt, vtt) => {
            const msg = "Oturum verilerini ve üretilen tüm dosyaları kalıcı olarak silmek istediğinize emin misiniz?";
            if (!confirm(msg)) {
                return [null, null, null, null, null];
            }
            navigator.clipboard.writeText("");
            return [sid, audio, txt, srt, vtt];
        }
        """,
        inputs=[session_id, last_audio_path, download_txt, download_srt, download_vtt],
        outputs=cleanup_outputs,
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

    # Clean up logging for session disconnects
    def on_unload():
        logger.info("Oturum sonlandırıldı.")
        # Note: We don't automatically delete files here to allow for page refreshes

    demo.unload(fn=on_unload)

    # Consolidated load event for better sync
    demo.load(
        fn=sync_copy_button_on_load,
        inputs=[output_text],
        outputs=[copy_btn],
    )
