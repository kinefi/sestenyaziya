import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

from .config import EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR, settings

logger = logging.getLogger(__name__)


def get_file_hash(path: str) -> str:
    """Computes SHA-256 hash of the file content for caching."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_transcription_hash(
    audio_path: str,
    model_size: str,
    diarization: bool,
    num_speakers: int,
    session_id: str = "",
    language: str = "",
) -> str:
    """Generates a hash for a specific transcription configuration."""
    file_hash = get_file_hash(audio_path)
    key = f"{session_id}_{file_hash}_{model_size}_{int(diarization)}_{num_speakers}_{language}"
    return hashlib.sha256(key.encode()).hexdigest()


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8"):
    """
    Writes text content to a temporary file then atomically renames it to
    prevent file corruption during crashes.
    """
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def get_cache_size_mb() -> float:
    """Calculates total size of files in given directories in MB."""
    total_size = 0
    for d in [EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR]:
        if d.exists():
            for f in d.glob("**/*"):
                if f.is_file():
                    total_size += f.stat().st_size
    return total_size / (1024 * 1024)


def clean_cache_directories():
    """Deletes oldest cache files until the total size is within the limit."""
    directories = [EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR]
    max_size_mb = settings.default_cache_size_mb
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
        logger.info(f"Cache cleaned: {deleted_count} files deleted. Current size: {total_size / (1024 * 1024):.1f} MB")


def get_free_disk_mb() -> float:
    """Returns free disk space on the models directory mount point in MB."""
    usage = shutil.disk_usage(settings.models_dir.absolute())
    return usage.free / (1024 * 1024)
