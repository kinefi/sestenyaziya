#!/usr/bin/env python3

import argparse
import logging
import os

from app.cache_utils import clean_cache_directories
from app.config import ModelSize, settings, setup_logging, update_settings

logger = logging.getLogger(__name__)


def setup_config():
    """Parses CLI arguments and updates global configuration."""
    parser = argparse.ArgumentParser(description="Ses'ten Yazıya — Türkçe konuşmayı metne dönüştürür")
    parser.add_argument(
        "--model",
        default=settings.default_model_size,
        choices=ModelSize.values(),
        metavar="SIZE",
        help="Whisper model boyutu: small | medium | large-v3 (varsayılan: %(default)s)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=settings.sample_rate,
        metavar="HZ",
        help="Ses örnekleme hızı (varsayılan: %(default)s)",
    )
    parser.add_argument(
        "--paragraph-pause",
        type=float,
        default=settings.paragraph_pause,
        metavar="SEC",
        help="Paragraf sınırı için sessizlik eşiği saniye (varsayılan: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=settings.transcription_timeout,
        metavar="SEC",
        help="Tek bir dosya için işlem zaman aşımı (varsayılan: %(default)s)",
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
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face Hub erişim tokenı (rate limit uyarılarını önlemek için)",
    )

    # parse_known_args ignores Gradio's own argv when hot-reloading
    args, _ = parser.parse_known_args()

    # Update config values globally
    update_settings(args.model, args.sample_rate, args.paragraph_pause, args.timeout)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    return args


args = setup_config()
setup_logging()

# Clean up embedding cache on startup
clean_cache_directories()

# Import UI components at module level so Gradio CLI can detect the 'demo' object
from app.ui import UI_CSS, demo  # noqa: E402

demo.queue()

if __name__ == "__main__":
    demo.launch(
        share=args.share,
        show_error=True,
        server_port=args.port,
        css=UI_CSS,
    )
