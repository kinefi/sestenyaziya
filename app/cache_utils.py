import hashlib
import logging
import os
import shutil
import tempfile
import time
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
    # Prefixing the hash with session_id makes it easy to find and delete session files later
    h = hashlib.sha256(key.encode()).hexdigest()
    return f"{session_id}_{h}" if session_id else h


def get_embedding_hash(audio_path: str, session_id: str = "") -> str:
    """Generates a hash for embedding cache, optionally prefixed by session_id."""
    file_hash = get_file_hash(audio_path)
    return f"{session_id}_{file_hash}" if session_id else file_hash


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


def cleanup_expired_cache():
    """Deletes files that haven't been accessed for more than the configured expiry time."""
    expiry_seconds = settings.session_expiry_hours * 3600
    now = time.time()
    deleted_count = 0

    for d in [EMBEDDING_CACHE_DIR, TRANSCRIPT_CACHE_DIR]:
        if not d.exists():
            continue
        for f in d.glob("*"):
            if f.is_file():
                if (now - f.stat().st_mtime) > expiry_seconds:
                    try:
                        f.unlink()
                        deleted_count += 1
                    except Exception:
                        logger.exception(f"Failed to delete expired cache file: {f}")
    if deleted_count > 0:
        logger.info(f"Expired cache cleaned: {deleted_count} files removed (TTL: {settings.session_expiry_hours}h)")


def clean_cache_directories():
    """Deletes oldest cache files until the total size is within the limit."""
    cleanup_expired_cache()
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


def get_models_size_mb() -> float:
    """Calculates total size of files in the models directory in MB."""
    total_size = 0
    if settings.models_dir.exists():
        for f in settings.models_dir.glob("**/*"):
            if f.is_file():
                total_size += f.stat().st_size
    return total_size / (1024 * 1024)


def get_free_disk_mb() -> float:
    """Returns free disk space on the models directory mount point in MB."""
    path = settings.models_dir.absolute()
    # Walk up to the first existing parent directory to check disk usage
    # if the target directory doesn't exist yet.
    while not path.exists() and path.parent != path:
        path = path.parent
    usage = shutil.disk_usage(path)
    return usage.free / (1024 * 1024)


def delete_session_cache(session_id: str):
    """Deletes all cache files associated with a specific session_id."""
    if not session_id:
        return

    prefix = f"{session_id}_"
    for d in [TRANSCRIPT_CACHE_DIR, EMBEDDING_CACHE_DIR]:
        for f in d.glob(f"{prefix}*"):
            try:
                f.unlink()
            except Exception:
                logger.exception(f"Failed to delete session file: {f}")
