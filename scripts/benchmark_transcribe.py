import time
import argparse
import sys
import os
# Ensure repository root is on sys.path so `app` imports resolve under `uv run`
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from app import model
from app.transcription import transcribe


def run_benchmark(audio_path, model_size, use_chunking, chunk_size, overlap, ffmpeg, timeout=7200):
    # Ensure model loaded
    model.load_model(model_size)

    gen = transcribe(
        audio_path,
        model_size,
        enable_diarization=False,
        num_speakers=0,
        session_id="bench",
        timeout=timeout,
        language=None,
        low_latency=False,
        chunk_size_seconds=chunk_size,
        chunk_overlap_seconds=overlap,
        use_chunking=use_chunking,
        use_ffmpeg_piping=ffmpeg,
    )

    start = time.time()
    last_progress = 0
    for out in gen:
        # out is a tuple (dataclass as tuple)
        # TranscriptionResult astuple ordering: result, txt_path, srt_path, vtt_path, status, speaker_info, progress
        try:
            progress = out[6]
        except Exception:
            progress = 0
        if int(progress) != int(last_progress):
            print(f"Progress: {progress:.1f}% - {out[4]}")
            last_progress = progress
    elapsed = time.time() - start
    print(f"Benchmark finished in {elapsed:.1f}s")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('audio', help='Path to audio file')
    p.add_argument('--model', default='medium')
    p.add_argument('--no-chunk', dest='use_chunking', action='store_false')
    p.add_argument('--chunk-size', type=int, default=30)
    p.add_argument('--overlap', type=int, default=2)
    p.add_argument('--ffmpeg', action='store_true')
    args = p.parse_args()

    run_benchmark(args.audio, args.model, args.use_chunking, args.chunk_size, args.overlap, args.ffmpeg)
