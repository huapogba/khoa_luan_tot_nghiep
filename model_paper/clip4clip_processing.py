"""
CLIP4Clip: embed the MSVD test videos.

Default model: Searchium-ai/clip4clip-webvid150k, CLIP4Clip (meanP, ViT-B/32)
trained on WebVid-150k, not on MSVD (zero-shot transfer).
--checkpoint: a model fine-tuned on MSVD train by clip4clip_finetune.py.
12 uniform frames, each encoded by the image tower, mean-pooled into one
512-D vector.

Input : MSVD_data/msvd_test.json, MSVD_data/Video/test/*.avi
Output: msvd_evaluate/features/clip4clip_video.npz     (video_ids, video)
        msvd_evaluate/features/clip4clip_ft_video.npz  with --checkpoint

Example:
    python msvd_evaluate/clip4clip_processing.py
    python msvd_evaluate/clip4clip_processing.py --checkpoint msvd_evaluate/checkpoints/clip4clip_msvd
    python msvd_evaluate/clip4clip_processing.py --limit 20
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import clip_encoder
from msvd_common import (
    FEATURE_DIR, JSON_FILE, VIDEO_DIR, load_records, read_frames, setup_stdout, video_path,
)

MODEL_NAME = "Searchium-ai/clip4clip-webvid150k"
NUM_FRAMES = 12
OUTPUT_FILE = FEATURE_DIR / "clip4clip_video.npz"
FT_OUTPUT_FILE = FEATURE_DIR / "clip4clip_ft_video.npz"  # with --checkpoint


def main():
    setup_stdout()

    parser = argparse.ArgumentParser(description="Embed MSVD test videos with CLIP4Clip.")
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Fine-tuned model folder (clip4clip_finetune.py) instead of the WebVid model.")
    parser.add_argument("--frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_name = str(args.checkpoint.resolve()) if args.checkpoint else MODEL_NAME
    output = args.output or (FT_OUTPUT_FILE if args.checkpoint else OUTPUT_FILE)

    records = load_records(args.json)[:args.limit]
    processor, model = clip_encoder.load_vision(model_name)

    video_ids, vectors, failed = [], [], []

    for r in tqdm(records, desc="CLIP4Clip video"):
        try:
            frames = read_frames(video_path(r["video_id"], args.video_dir), args.frames)
        except RuntimeError as e:
            failed.append(r["video_id"])
            print(f"\n{e}")
            continue
        video_ids.append(r["video_id"])
        vectors.append(clip_encoder.encode_video(processor, model, frames))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        video_ids=np.array(video_ids),
        video=np.stack(vectors).astype(np.float32),
        model=np.array(model_name),
        frames=np.array(args.frames),
    )

    print(f"\nEncoded {len(video_ids)} videos ({len(failed)} failed) -> {output}")


if __name__ == "__main__":
    main()
