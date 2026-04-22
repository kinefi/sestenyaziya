import argparse

import app.config as cfg
from app.cache_utils import clean_embedding_cache

def setup_config():
    """Parses CLI arguments and updates global configuration."""
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

    # parse_known_args ignores Gradio's own argv when hot-reloading
    args, _ = parser.parse_known_args()

    # Update config values globally
    cfg.DEFAULT_MODEL_SIZE = args.model
    cfg.SAMPLE_RATE = args.sample_rate
    cfg.PARAGRAPH_PAUSE = args.paragraph_pause
    
    return args


if __name__ == "__main__":
    args = setup_config()
    cfg.setup_logging()
    
    # Clean up embedding cache on startup (limit to 1GB)
    clean_embedding_cache([cfg.EMBEDDING_CACHE_DIR, cfg.TRANSCRIPT_CACHE_DIR, cfg.TEMP_EXPORT_DIR], 
                          max_size_mb=cfg.DEFAULT_CACHE_SIZE_MB)
    
    print(f"🖥️  Cihaz: {cfg.device.upper()} | Hesaplama tipi: {cfg.compute_type}")

    # Deferred import: UI components must load AFTER config is mutated
    from app.ui import demo

    demo.queue()
    demo.launch(share=args.share, show_error=True, server_port=args.port)
