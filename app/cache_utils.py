import logging
import hashlib
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def get_file_hash(path: str) -> str:
    """Computes SHA-256 hash of the file content for caching."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_transcription_hash(audio_path: str, model_size: str, diarization: bool, num_speakers: int) -> str:
    """Generates a hash for a specific transcription configuration."""
    file_hash = get_file_hash(audio_path)
    key = f"{file_hash}_{model_size}_{int(diarization)}_{num_speakers}"
    return hashlib.sha256(key.encode()).hexdigest()

def get_cache_size_mb(directories: list[Path]) -> float:
    """Calculates total size of files in given directories in MB."""
    total_size = 0
    for d in directories:
        if d.exists():
            for f in d.glob("**/*"):
                if f.is_file():
                    total_size += f.stat().st_size
    return total_size / (1024 * 1024)

def clear_all_cache(cache_base_dir: Path):
    """Deletes all cached files."""
    if cache_base_dir.exists():
        shutil.rmtree(cache_base_dir)
    cache_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("All cache files cleared.")

def clean_embedding_cache(directories: list[Path], max_size_mb: int = 1000):
    """Deletes oldest cache files until the total size is within the limit."""
    files = []
    for d in directories:
        if d.exists():
            files.extend(list(d.glob("*.*")))
            
    if not files:
        return

    # Sort by modification time (oldest first)
    files.sort(key=lambda x: x.stat().st_mtime)
    
    total_size = sum(f.stat().st_size for f in files)
    max_bytes = max_size_mb * 1024 * 1024
    
    deleted_count = 0
    while total_size > max_bytes and files:
        f = files.pop(0)
        size = f.stat().st_size
        try:
            f.unlink()
            total_size -= size
            deleted_count += 1
        except Exception:
            logger.exception(f"Failed to delete cache file: {f}")
            
    if deleted_count > 0:
        logger.info(f"Cache cleaned: {deleted_count} files deleted. Current size: {total_size / (1024*1024):.1f} MB")