"""
CLIP4Clip: text-to-video retrieval on MSVD test.

Default: Searchium-ai/clip4clip-webvid150k (WebVid-150k, zero-shot transfer).
--checkpoint: a model fine-tuned on MSVD train by clip4clip_finetune.py.

Needs the matching video features from clip4clip_processing.py
(same --checkpoint). Captions are encoded with the same model's text tower;
score = cosine.

Output: msvd_evaluate/results/clip4clip_results.json
        msvd_evaluate/results/clip4clip_ft_results.json  with --checkpoint

Example:
    python msvd_evaluate/clip4clip_evaluate.py
    python msvd_evaluate/clip4clip_evaluate.py --checkpoint msvd_evaluate/checkpoints/clip4clip_msvd
    python msvd_evaluate/clip4clip_evaluate.py --all-videos
"""

import argparse
from pathlib import Path

import clip_encoder
from msvd_common import (
    FEATURE_DIR, JSON_FILE, RESULT_DIR, flatten_captions, load_features, load_records,
    score_and_report, select_candidates, setup_stdout,
)

MODEL_NAME = "Searchium-ai/clip4clip-webvid150k"
FEATURE_FILE = FEATURE_DIR / "clip4clip_video.npz"
OUTPUT_FILE = RESULT_DIR / "clip4clip_results.json"
FT_FEATURE_FILE = FEATURE_DIR / "clip4clip_ft_video.npz"   # with --checkpoint
FT_OUTPUT_FILE = RESULT_DIR / "clip4clip_ft_results.json"


def main():
    setup_stdout()

    parser = argparse.ArgumentParser(description="Evaluate CLIP4Clip on MSVD.")
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Fine-tuned model folder (clip4clip_finetune.py) instead of the WebVid model.")
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--all-videos", action="store_true",
                        help="Do not restrict candidates to the Qwen index's videos.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_name = str(args.checkpoint.resolve()) if args.checkpoint else MODEL_NAME
    features = args.features or (FT_FEATURE_FILE if args.checkpoint else FEATURE_FILE)
    output = args.output or (FT_OUTPUT_FILE if args.checkpoint else OUTPUT_FILE)

    feats = load_features(features)
    if str(feats["model"]) != model_name:
        raise ValueError(f"{features} was made with {feats['model']}, not {model_name}.")

    records, rows = select_candidates(load_records(args.json), feats["video_ids"], args.all_videos)
    captions, owner = flatten_captions(records)
    print(f"Candidates: {len(records)} videos   Queries: {len(captions)} captions")

    tokenizer, model = clip_encoder.load_text(model_name)
    text = clip_encoder.encode_texts(tokenizer, model, captions, args.batch_size)

    sim = text @ feats["video"][rows].T

    score_and_report(
        sim, records, owner,
        {"model": model_name, "frames": int(feats["frames"]), "pooling": "mean of frame embeddings"},
        output,
    )


if __name__ == "__main__":
    main()
