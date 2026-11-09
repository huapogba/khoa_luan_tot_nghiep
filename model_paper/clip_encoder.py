"""
CLIP-architecture encoder shared by clip_* and clip4clip_* scripts.

Video = mean of the L2-normalised frame embeddings (CLIP4Clip "meanP").
Text  = CLIP text tower, L2-normalised. Score = cosine.
"""

import numpy as np
import torch
from tqdm import tqdm

from msvd_common import DEVICE, DTYPE, normalize, pooled

# Image preprocessing and tokenizer of CLIP ViT-B/32 (CLIP4Clip uses the same).
PROCESSOR_NAME = "openai/clip-vit-base-patch32"


def load_vision(model_name):
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    processor = CLIPImageProcessor.from_pretrained(PROCESSOR_NAME)
    model = CLIPVisionModelWithProjection.from_pretrained(model_name, dtype=DTYPE)
    return processor, model.to(DEVICE).eval()


def load_text(model_name):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(PROCESSOR_NAME)
    model = CLIPTextModelWithProjection.from_pretrained(model_name, dtype=DTYPE)
    return tokenizer, model.to(DEVICE).eval()


@torch.no_grad()
def encode_video(processor, model, frames):
    pixels = processor(images=frames, return_tensors="pt")["pixel_values"]
    emb = normalize(pooled(model(pixel_values=pixels.to(DEVICE, DTYPE))).float())
    return normalize(emb.mean(dim=0)).cpu().numpy()


@torch.no_grad()
def encode_texts(tokenizer, model, texts, batch_size):
    out = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Caption"):
        tok = tokenizer(
            texts[i:i + batch_size], padding=True, truncation=True,
            max_length=77, return_tensors="pt",
        ).to(DEVICE)
        out.append(normalize(pooled(model(**tok)).float()).cpu().numpy())

    return np.concatenate(out)
