"""
Fine-tune LanguageBind_Video_FT on the MSVD train split (LoRA).

(This file used to fine-tune X-CLIP / microsoft/xclip-base-patch32; that
version is kept in xclip_ni_backup/.)

Starts from LanguageBind/LanguageBind_Video_FT (CLIP ViT-L/14 + temporal
attention, 8 frames, ~430M parameters). Full fine-tuning with AdamW needs
~7 GB for weights + gradients + optimiser state alone, more than a 6 GB GPU,
so the backbone is frozen and LoRA adapters (r=16, alpha=16, dropout 0.1,
as in the LanguageBind paper) are trained on every attention projection
(q/k/v/out) of both towers, spatial and temporal. Loss: symmetric InfoNCE
over the batch, one random caption per video per epoch, bf16 autocast +
gradient checkpointing. The best epoch on the val split is kept; LoRA is
merged into the weights when saving, so the checkpoint loads like the base.

Data  : MSVD_data/msvd_train.json, msvd_val.json, Video/train, Video/val
Output: msvd_evaluate/checkpoints/languagebind_msvd/  (pytorch_model.bin + config/tokenizer + training_log.json)

Then evaluate on the test split:
    python msvd_evaluate/xclip_processing.py --checkpoint msvd_evaluate/checkpoints/languagebind_msvd
    python msvd_evaluate/xclip_evaluate.py   --checkpoint msvd_evaluate/checkpoints/languagebind_msvd

Example:
    python msvd_evaluate/xclip_finetune.py
    python msvd_evaluate/xclip_finetune.py --batch-size 8
    python msvd_evaluate/xclip_finetune.py --limit 64 --epochs 1   # smoke test
"""

import argparse

import torch

from finetune_common import CHECKPOINT_DIR, add_common_args, symmetric_infonce, train
from languagebind_model import (
    BASE_MODEL, MEAN, NUM_FRAMES, STD, frames_to_uint8, load_model, load_tokenizer,
    save_model, tokenize,
)
from msvd_common import DEVICE, normalize, setup_stdout


class LanguageBindAdapter:
    tag = "languagebind"
    num_frames = NUM_FRAMES
    new_prefixes = ("lora_",)  # every trainable parameter is a LoRA weight
    mean, std = MEAN, STD

    def __init__(self, base_model, lora_r, lora_alpha, lora_dropout):
        self.name = f"LanguageBind_Video_FT + LoRA (r={lora_r}) from {base_model}"
        self.model, self.base_folder = load_model(base_model, DEVICE, torch.float32)
        self.tokenizer = load_tokenizer(base_model)
        n = self.model.add_lora(lora_r, lora_alpha, lora_dropout)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA: {n / 1e6:.2f}M trainable of {total / 1e6:.1f}M parameters")

    def to_uint8(self, frames):
        return frames_to_uint8(frames)

    def tokenize(self, texts):
        return tokenize(self.tokenizer, texts, DEVICE)

    def loss(self, pixels, tokens):
        video = normalize(self.model.get_video_features(pixels))
        text = normalize(self.model.get_text_features(**tokens))
        logits = self.model.logit_scale.exp() * text @ video.T
        return symmetric_infonce(logits.float())

    def similarity(self, pixel_batches, captions, batch_size=256):
        video = torch.cat([normalize(self.model.get_video_features(p).float()) for p in pixel_batches])
        text = torch.cat([
            normalize(self.model.get_text_features(**self.tokenize(captions[i:i + batch_size])).float())
            for i in range(0, len(captions), batch_size)
        ])
        # fp32 even under autocast: bf16 scores tie and inflate ranks
        with torch.autocast(DEVICE, enabled=False):
            return (text @ video.T).cpu().numpy()

    def save(self, path):
        save_model(self.model, self.base_folder, path)


def main():
    setup_stdout()

    parser = argparse.ArgumentParser(description="Fine-tune LanguageBind_Video_FT (LoRA) on MSVD train.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    add_common_args(
        parser, CHECKPOINT_DIR / "languagebind_msvd",
        lr=1e-4, new_lr=1e-4, batch_size=16, epochs=10,
    )
    args = parser.parse_args()

    train(LanguageBindAdapter(args.base_model, args.lora_r, args.lora_alpha, args.lora_dropout), args)


if __name__ == "__main__":
    main()
