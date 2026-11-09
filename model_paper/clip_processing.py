"""
CLIP zero-shot: embed the MSVD test videos.

Model: openai/clip-vit-base-patch32, no video training. 12 uniform frames,
each encoded by the CLIP image tower, mean-pooled into one 512-D vector.

Input : MSVD_data/msvd_test.json, MSVD_data/Video/test/*.avi
Output: msvd_evaluate/features/clip_video.npz  (video_ids, video)

Example:
    python msvd_evaluate/clip_processing.py
    python msvd_evaluate/clip_processing.py --limit 20
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import clip_encoder
from msvd_common import (
    FEATURE_DIR, JSON_FILE, VIDEO_DIR, load_records, read_frames, setup_stdout, video_path,
)

MODEL_NAME = "openai/clip-vit-base-patch32"
NUM_FRAMES = 12
OUTPUT_FILE = FEATURE_DIR / "clip_video.npz"


def main():
    setup_stdout()

    parser = argparse.ArgumentParser(description="Embed MSVD test videos with CLIP.")
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    records = load_records(args.json)[:args.limit]
    processor, model = clip_encoder.load_vision(MODEL_NAME)

    video_ids, vectors, failed = [], [], []

    for r in tqdm(records, desc="CLIP video"):
        try:
            frames = read_frames(video_path(r["video_id"], args.video_dir), args.frames)
        except RuntimeError as e:
            failed.append(r["video_id"])
            print(f"\n{e}")
            continue
        video_ids.append(r["video_id"])
        vectors.append(clip_encoder.encode_video(processor, model, frames))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        video_ids=np.array(video_ids),
        video=np.stack(vectors).astype(np.float32),
        model=np.array(MODEL_NAME),
        frames=np.array(args.frames),
    )

    print(f"\nEncoded {len(video_ids)} videos ({len(failed)} failed) -> {args.output}")


if __name__ == "__main__":
    main()
