"""
Fine-tune CLIP4Clip (meanP) on the MSVD train split.

As in the CLIP4Clip paper, training starts from the original CLIP
(openai/clip-vit-base-patch32): 12 frames per video, each frame encoded by
the image tower, frame embeddings L2-normalised and mean-pooled into the
video embedding; text from the CLIP text tower; symmetric InfoNCE loss.
meanP adds no new layers, so every weight is a CLIP weight (--lr).

Differences from the paper (single 6 GB GPU): batch 32 instead of 128,
bf16 autocast + gradient checkpointing, one random caption per video per
epoch. The best epoch on the val split is kept.

Data  : MSVD_data/msvd_train.json, msvd_val.json, Video/train, Video/val
Output: msvd_evaluate/checkpoints/clip4clip_msvd/  (HF CLIPModel + training_log.json)

Then evaluate on the test split:
    python msvd_evaluate/clip4clip_processing.py --checkpoint msvd_evaluate/checkpoints/clip4clip_msvd
    python msvd_evaluate/clip4clip_evaluate.py   --checkpoint msvd_evaluate/checkpoints/clip4clip_msvd

Example:
    python msvd_evaluate/clip4clip_finetune.py
    python msvd_evaluate/clip4clip_finetune.py --batch-size 16 --lr 5e-6
    python msvd_evaluate/clip4clip_finetune.py --limit 64 --epochs 1   # smoke test
"""

import argparse

import numpy as np
import torch

from finetune_common import CHECKPOINT_DIR, add_common_args, symmetric_infonce, train
from msvd_common import DEVICE, normalize, pooled, setup_stdout

BASE_MODEL = "openai/clip-vit-base-patch32"
NUM_FRAMES = 12
MAX_WORDS = 32  # CLIP4Clip setting; MSVD captions are much shorter


class Clip4ClipAdapter:
    name = f"CLIP4Clip meanP from {BASE_MODEL}"
    tag = "clip"
    num_frames = NUM_FRAMES
    new_prefixes = ()

    def __init__(self, base_model):
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

        self.processor = CLIPImageProcessor.from_pretrained(BASE_MODEL)
        self.tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL)
        self.model = CLIPModel.from_pretrained(base_model).to(DEVICE)
        self.mean, self.std = self.processor.image_mean, self.processor.image_std

    def to_uint8(self, frames):
        x = self.processor(
            images=frames, do_rescale=False, do_normalize=False, return_tensors="pt"
        )["pixel_values"]
        return x.round().clamp(0, 255).to(torch.uint8)

    def tokenize(self, texts, max_length=MAX_WORDS):
        return self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(DEVICE)

    def video_embeds(self, pixels):
        b, t = pixels.shape[:2]
        frames = pooled(self.model.get_image_features(pixel_values=pixels.flatten(0, 1)))
        return normalize(normalize(frames).view(b, t, -1).mean(dim=1))

    def text_embeds(self, tokens):
        return normalize(pooled(self.model.get_text_features(**tokens)))

    def loss(self, pixels, tokens):
        video = self.video_embeds(pixels)
        text = self.text_embeds(tokens)
        logits = self.model.logit_scale.exp() * text @ video.T
        return symmetric_infonce(logits.float())

    def similarity(self, pixel_batches, captions, batch_size=256):
        video = torch.cat([self.video_embeds(p).float() for p in pixel_batches])
        text = torch.cat([
            self.text_embeds(self.tokenize(captions[i:i + batch_size], 77)).float()
            for i in range(0, len(captions), batch_size)
        ])
        # fp32 even under autocast: bf16 scores tie and inflate ranks
        with torch.autocast(DEVICE, enabled=False):
            return (text @ video.T).cpu().numpy()

    def save(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def main():
    setup_stdout()

    parser = argparse.ArgumentParser(description="Fine-tune CLIP4Clip (meanP) on MSVD train.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    add_common_args(
        parser, CHECKPOINT_DIR / "clip4clip_msvd",
        lr=1e-6, new_lr=1e-6, batch_size=32, epochs=20,
    )
    args = parser.parse_args()

    train(Clip4ClipAdapter(args.base_model), args)


if __name__ == "__main__":
    main()
