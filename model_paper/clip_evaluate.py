"""
CLIP zero-shot: text-to-video retrieval on MSVD test.

Needs msvd_evaluate/features/clip_video.npz (run clip_processing.py first).
Captions are encoded with the CLIP text tower; score = cosine.

Output: msvd_evaluate/results/clip_results.json

Example:
    python msvd_evaluate/clip_evaluate.py
    python msvd_evaluate/clip_evaluate.py --all-videos
"""

import argparse
from pathlib import Path

import clip_encoder
from msvd_common import (
    FEATURE_DIR, JSON_FILE, RESULT_DIR, flatten_captions, load_features, load_records,
    score_and_report, select_candidates, setup_stdout,
)

MODEL_NAME = "openai/clip-vit-base-patch32"
FEATURE_FILE = FEATURE_DIR / "clip_video.npz"
OUTPUT_FILE = RESULT_DIR / "clip_results.json"


def main():
    setup_stdout()

    parser = argparse.ArgumentParser(description="Evaluate CLIP zero-shot on MSVD.")
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--features", type=Path, default=FEATURE_FILE)
    parser.add_argument("--all-videos", action="store_true",
                        help="Do not restrict candidates to the Qwen index's videos.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    feats = load_features(args.features)
    records, rows = select_candidates(load_records(args.json), feats["video_ids"], args.all_videos)
    captions, owner = flatten_captions(records)
    print(f"Candidates: {len(records)} videos   Queries: {len(captions)} captions")

    tokenizer, model = clip_encoder.load_text(MODEL_NAME)
    text = clip_encoder.encode_texts(tokenizer, model, captions, args.batch_size)

    sim = text @ feats["video"][rows].T

    score_and_report(
        sim, records, owner,
        {"model": MODEL_NAME, "frames": int(feats["frames"]), "pooling": "mean of frame embeddings"},
        args.output,
    )


if __name__ == "__main__":
    main()
