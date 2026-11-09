"""
LanguageBind_Video_FT: text-to-video retrieval on MSVD test.

(This file used to evaluate X-CLIP / microsoft/xclip-base-patch32; that
version is kept in xclip_ni_backup/.)

Needs the video features from xclip_processing.py. Score = cosine between
the caption embedding and the video embedding (both 768-D, projected).

Output: msvd_evaluate/results/languagebind_results.json      (base model)
        msvd_evaluate/results/languagebind_ft_results.json   with --checkpoint

Example:
    python msvd_evaluate/xclip_evaluate.py
    python msvd_evaluate/xclip_evaluate.py --all-videos
    python msvd_evaluate/xclip_evaluate.py --checkpoint msvd_evaluate/checkpoints/languagebind_msvd
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from languagebind_model import BASE_MODEL, load_model, load_tokenizer, tokenize
from msvd_common import (
    DEVICE, DTYPE, FEATURE_DIR, JSON_FILE, RESULT_DIR, flatten_captions, load_features,
    load_records, normalize, score_and_report, select_candidates, setup_stdout,
)

MODEL_NAME = BASE_MODEL
FEATURE_FILE = FEATURE_DIR / "languagebind_video.npz"
OUTPUT_FILE = RESULT_DIR / "languagebind_results.json"
FT_FEATURE_FILE = FEATURE_DIR / "languagebind_ft_video.npz"   # with --checkpoint
FT_OUTPUT_FILE = RESULT_DIR / "languagebind_ft_results.json"


@torch.no_grad()
def encode_texts(tokenizer, model, texts, batch_size):
    """
    L2-normalised caption embeddings (float32).
    """
    out = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Caption"):
        tok = tokenize(tokenizer, texts[i:i + batch_size], DEVICE)
        out.append(normalize(model.get_text_features(**tok).float()))

    return torch.cat(out)


@torch.no_grad()
def similarity(text, videos):
    """
    captions x videos cosine matrix (fp32).
    """
    video = normalize(torch.as_tensor(videos, device=DEVICE).float())
    return (text.float() @ video.T).cpu().numpy()


def main():
    setup_stdout()

    parser = argparse.ArgumentParser(description="Evaluate LanguageBind_Video_FT on MSVD.")
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Fine-tuned model folder (xclip_finetune.py) instead of the base model.")
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

    model, _ = load_model(model_name, DEVICE, DTYPE)
    tokenizer = load_tokenizer(model_name)

    text = encode_texts(tokenizer, model, captions, args.batch_size)
    sim = similarity(text, feats["video"][rows])

    score_and_report(
        sim, records, owner,
        {"model": model_name, "frames": int(feats["frames"]),
         "score": "cosine(caption embedding, video embedding)"},
        output,
    )


if __name__ == "__main__":
    main()
