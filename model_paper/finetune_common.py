"""
Shared training code for clip4clip_finetune.py and xclip_finetune.py.

Data: MSVD official split from MSVD_data/ (python down_msvd.py --splits train val test)
    train 1,200 videos -> training
    val     100 videos -> model selection (best val t2v R@1 is kept)
    test    670 videos -> never touched here; use *_processing / *_evaluate

Frames are decoded once and cached as uint8 (already resized and cropped by
the model's own image processor) in msvd_evaluate/cache/, so an epoch does
not re-decode the .avi files. Normalisation (mean/std) is done on the GPU.
    CLIP4Clip, 12 frames: train ~2.2 GB, val ~0.2 GB
    X-CLIP,     8 frames: train ~1.5 GB, val ~0.1 GB

Each epoch: one pass over the training videos, one random caption per video
(MSVD has ~40 captions per video, so every epoch sees different text).
Loss: symmetric InfoNCE over the batch (text->video and video->text).
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from msvd_common import (
    DEVICE, EVAL_DIR, MSVD_DIR, calculate_metrics, flatten_captions, load_records,
    print_metrics, read_frames, save_json, t2v_ranks, v2t_ranks, video_path,
)

CACHE_DIR = EVAL_DIR / "cache"
CHECKPOINT_DIR = EVAL_DIR / "checkpoints"


def split_json(split):
    return MSVD_DIR / f"msvd_{split}.json"


def split_video_dir(split):
    return MSVD_DIR / "Video" / split


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Frame cache
# ---------------------------------------------------------------------
def load_split(split, tag, num_frames, to_uint8, limit=None):
    """
    (records, frames) for an MSVD split. frames: uint8 memmap
    (videos, num_frames, 3, 224, 224), row i = records[i].

    to_uint8(list of HxWx3 RGB frames) -> uint8 tensor (T, 3, 224, 224),
    i.e. the model's processor with rescale/normalise turned off.
    """
    records = [
        r for r in load_records(split_json(split))
        if video_path(r["video_id"], split_video_dir(split)).exists()
    ]
    if limit is not None:
        records = records[:limit]

    suffix = f"_n{limit}" if limit is not None else ""
    frames_file = CACHE_DIR / f"{tag}_{split}_{num_frames}f{suffix}.npy"
    ids_file = frames_file.with_suffix(".json")

    ids = [r["video_id"] for r in records]

    if frames_file.exists() and ids_file.exists():
        with open(ids_file, "r", encoding="utf-8") as f:
            if json.load(f) == ids:
                return records, np.load(frames_file, mmap_mode="r")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = frames_file.with_name(frames_file.stem + "_tmp.npy")
    frames = np.lib.format.open_memmap(
        tmp_file, mode="w+", dtype=np.uint8,
        shape=(len(records), num_frames, 3, 224, 224),
    )

    for i, r in enumerate(tqdm(records, desc=f"Caching {split} frames ({tag})")):
        raw = read_frames(video_path(r["video_id"], split_video_dir(split)), num_frames)
        frames[i] = to_uint8(raw).numpy()

    frames.flush()
    del frames
    tmp_file.replace(frames_file)

    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    return records, np.load(frames_file, mmap_mode="r")


def to_pixels(frames_uint8, mean, std, dtype):
    """
    uint8 (B, T, 3, H, W) numpy/tensor -> normalised float on DEVICE.
    """
    x = torch.from_numpy(np.array(frames_uint8)).to(DEVICE, non_blocking=True)
    x = x.float().div_(255.0)
    mean = torch.tensor(mean, device=DEVICE).view(1, 1, 3, 1, 1)
    std = torch.tensor(std, device=DEVICE).view(1, 1, 3, 1, 1)
    return ((x - mean) / std).to(dtype)


# ---------------------------------------------------------------------
# Loss / optimiser
# ---------------------------------------------------------------------
def symmetric_infonce(logits_t2v):
    """
    logits_t2v: (B texts, B videos), matching pairs on the diagonal.
    """
    target = torch.arange(logits_t2v.shape[0], device=logits_t2v.device)
    return (F.cross_entropy(logits_t2v, target) + F.cross_entropy(logits_t2v.T, target)) / 2


def build_optimizer(model, lr, new_lr, new_prefixes, weight_decay):
    """
    AdamW with two learning rates: pretrained CLIP weights (lr) and
    parameters whose name contains one of new_prefixes (new_lr).
    No weight decay on biases, norms, embeddings and logit_scale.
    """
    groups = {(n, d): [] for n in (False, True) for d in (False, True)}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_new = any(p in name for p in new_prefixes)
        decay = p.ndim >= 2 and "embedding" not in name
        groups[(is_new, decay)].append(p)

    param_groups = [
        {"params": ps, "lr": new_lr if is_new else lr,
         "weight_decay": weight_decay if decay else 0.0}
        for (is_new, decay), ps in groups.items() if ps
    ]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)


def cosine_schedule(optimizer, total_steps, warmup_ratio):
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast_dtype():
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def retrieval_metrics(sim, owner):
    return {
        "t2v": calculate_metrics(t2v_ranks(sim, owner)),
        "v2t": calculate_metrics(v2t_ranks(sim, owner)),
    }


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train(adapter, args):
    """
    adapter provides:
        name, num_frames, mean, std, tag
        model                      (torch module, on DEVICE)
        to_uint8(frames)           processor without rescale/normalise
        tokenize(texts)            dict of tensors on DEVICE
        loss(pixels, tokens)       scalar contrastive loss
        similarity(pixels_iter, captions) -> captions x videos numpy matrix
        new_prefixes               parameter names trained with --new-lr
        save(path)                 save_pretrained model (+ processor)
    """
    seed_everything(args.seed)

    train_records, train_frames = load_split(
        "train", adapter.tag, adapter.num_frames, adapter.to_uint8, args.limit
    )
    val_records, val_frames = load_split(
        "val", adapter.tag, adapter.num_frames, adapter.to_uint8, args.limit
    )
    val_captions, val_owner = flatten_captions(val_records)

    print(
        f"Train: {len(train_records)} videos   Val: {len(val_records)} videos, "
        f"{len(val_captions)} captions   Frames: {adapter.num_frames}"
    )

    model = adapter.model
    if args.grad_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except (ValueError, AttributeError) as e:
            print(f"Gradient checkpointing not available: {e}")

    steps_per_epoch = len(train_records) // args.batch_size
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs

    optimizer = build_optimizer(model, args.lr, args.new_lr, adapter.new_prefixes, args.weight_decay)
    scheduler = cosine_schedule(optimizer, total_steps, args.warmup)
    amp_dtype = autocast_dtype()
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    out_dir = Path(args.output)
    history = []
    best = -1.0

    def validate(epoch):
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
            batches = (
                to_pixels(val_frames[i:i + args.eval_batch_size], adapter.mean, adapter.std, amp_dtype)
                for i in range(0, len(val_records), args.eval_batch_size)
            )
            sim = adapter.similarity(batches, val_captions)
        metrics = retrieval_metrics(sim, val_owner)
        print(f"\n[val] epoch {epoch}")
        print_metrics(metrics)
        return metrics

    # Zero-shot reference before any update.
    history.append({"epoch": 0, "val": validate(0)})

    print(
        f"\nTraining {args.epochs} epochs x {steps_per_epoch} steps "
        f"(batch {args.batch_size}, lr {args.lr}, new lr {args.new_lr}, amp {amp_dtype})"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_records))
        start, losses = time.time(), []

        bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}")
        for step in bar:
            idx = np.sort(order[step * args.batch_size:(step + 1) * args.batch_size])
            texts = [random.choice(train_records[i]["captions"]) for i in idx]

            pixels = to_pixels(train_frames[idx], adapter.mean, adapter.std, torch.float32)
            tokens = adapter.tokenize(texts)

            with torch.autocast("cuda", dtype=amp_dtype):
                loss = adapter.loss(pixels, tokens)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.append(loss.item())
            bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        metrics = validate(epoch)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "seconds": round(time.time() - start, 1),
            "val": metrics,
        })

        score = metrics["t2v"]["R@1"]
        if score > best:
            best = score
            adapter.save(out_dir)
            print(f"New best val t2v R@1 {score * 100:.2f}% -> saved {out_dir}")

        save_json(
            {"model": adapter.name, "args": {k: str(v) for k, v in vars(args).items()},
             "best_val_t2v_R@1": best, "history": history},
            out_dir / "training_log.json",
        )

    print(f"\nDone. Best val t2v R@1: {best * 100:.2f}%   Checkpoint: {out_dir}")


def add_common_args(parser, default_output, lr, new_lr, batch_size, epochs):
    parser.add_argument("--output", type=Path, default=default_output,
                        help="Folder for the best checkpoint and training_log.json.")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help="Videos per step (= negatives in the contrastive loss).")
    parser.add_argument("--lr", type=float, default=lr, help="LR of the pretrained CLIP weights.")
    parser.add_argument("--new-lr", type=float, default=new_lr, help="LR of model-specific modules.")
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup fraction of all steps.")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false",
                        help="Faster but needs more VRAM.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N train/val videos (smoke test).")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
