"""
LanguageBind_Video_FT: embed the MSVD test videos.

(This file used to run X-CLIP / microsoft/xclip-base-patch32; that version is
kept in xclip_ni_backup/.)

Model: LanguageBind/LanguageBind_Video_FT (Zhu et al., ICLR 2024), CLIP
ViT-L/14 + temporal attention, trained on VIDAL-10M, not on MSVD.
8 frames -> one 768-D video vector (see languagebind_model.py).

Input : MSVD_data/msvd_test.json, MSVD_data/Video/test/*.avi
Output: msvd_evaluate/features/languagebind_video.npz      (base model)
        msvd_evaluate/features/languagebind_ft_video.npz   with --checkpoint

Example:
    python msvd_evaluate/xclip_processing.py
    python msvd_evaluate/xclip_processing.py --checkpoint msvd_evaluate/checkpoints/languagebind_msvd
    python msvd_evaluate/xclip_processing.py --limit 20
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from languagebind_model import (
    BASE_MODEL, NUM_FRAMES, frames_to_uint8, load_model, normalize_pixels,
)
from msvd_common import (
    DEVICE, DTYPE, FEATURE_DIR, JSON_FILE, VIDEO_DIR, load_records, read_frames,
    setup_stdout, video_path,
)

MODEL_NAME = BASE_MODEL
OUTPUT_FILE = FEATURE_DIR / "languagebind_video.npz"
FT_OUTPUT_FILE = FEATURE_DIR / "languagebind_ft_video.npz"  # with --checkpoint


@torch.no_grad()
def encode_video(model, frames):
    pixels = normalize_pixels(frames_to_uint8(frames)[None], DEVICE, DTYPE)
    return model.get_video_features(pixels)[0].float().cpu().numpy()


def main():
    setup_stdout()

    parser = argparse.ArgumentParser(description="Embed MSVD test videos with LanguageBind_Video_FT.")
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Fine-tuned model folder (xclip_finetune.py) instead of the base model.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_name = str(args.checkpoint.resolve()) if args.checkpoint else MODEL_NAME
    output = args.output or (FT_OUTPUT_FILE if args.checkpoint else OUTPUT_FILE)

    model, _ = load_model(model_name, DEVICE, DTYPE)

    records = load_records(args.json)[:args.limit]
    video_ids, videos, failed = [], [], []

    for r in tqdm(records, desc="LanguageBind video"):
        try:
            frames = read_frames(video_path(r["video_id"], args.video_dir), NUM_FRAMES)
        except RuntimeError as e:
            failed.append(r["video_id"])
            print(f"\n{e}")
            continue
        video_ids.append(r["video_id"])
        videos.append(encode_video(model, frames))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        video_ids=np.array(video_ids),
        video=np.stack(videos).astype(np.float32),
        model=np.array(model_name),
        frames=np.array(NUM_FRAMES),
    )

    print(f"\nEncoded {len(video_ids)} videos ({len(failed)} failed) -> {output}")


if __name__ == "__main__":
    main()
