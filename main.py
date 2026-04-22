import argparse

import app.config as cfg

parser = argparse.ArgumentParser(description="Ses'ten Yazıya — Türkçe konuşmayı metne dönüştürür")
parser.add_argument(
    "--model",
    default=cfg.DEFAULT_MODEL_SIZE,
    choices=cfg.ModelSize.values(),
    metavar="SIZE",
    help="Whisper model boyutu: small | medium | large-v3 (varsayılan: %(default)s)",
)
parser.add_argument(
    "--sample-rate",
    type=int,
    default=cfg.SAMPLE_RATE,
    metavar="HZ",
    help="Ses örnekleme hızı (varsayılan: %(default)s)",
)
parser.add_argument(
    "--paragraph-pause",
    type=float,
    default=cfg.PARAGRAPH_PAUSE,
    metavar="SEC",
    help="Paragraf sınırı için sessizlik eşiği saniye (varsayılan: %(default)s)",
)
parser.add_argument(
    "--port",
    type=int,
    default=7860,
    metavar="PORT",
    help="Gradio sunucu portu (varsayılan: %(default)s)",
)
parser.add_argument(
    "--share",
    action="store_true",
    help="Herkese açık Gradio paylaşım bağlantısı oluştur",
)

# parse_known_args ignores Gradio's own argv when hot-reloading via `gradio main.py`
args, _ = parser.parse_known_args()

# Mutate config before any other app module is imported so all
# downstream `from .config import X` bindings pick up the CLI values.
cfg.DEFAULT_MODEL_SIZE = args.model
cfg.SAMPLE_RATE = args.sample_rate
cfg.PARAGRAPH_PAUSE = args.paragraph_pause

print(f"🖥️  Cihaz: {cfg.device.upper()} | Hesaplama tipi: {cfg.compute_type}")

from app.ui import demo  # noqa: E402

demo.queue()

if __name__ == "__main__":
    demo.launch(share=args.share, show_error=True, server_port=args.port)
