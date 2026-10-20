"""
MSVD evaluation: text-to-video retrieval with the VLM only (Qwen3-VL-Embedding).

MSVD has no on-screen text or speech worth fusing, so this measures the
VLM alone. Only loads files produced offline, no video is reprocessed:
    MSVD_data/msvd_test.json         -> 670 test videos, ~40 captions each (down_msvd.py)
    data/bin/index_msvd.faiss        -> video vectors (video_processing.py, 768-D)
    data/bin/metadata_msvd.json      -> video_id of every FAISS row

Build the index first:
    python down_msvd.py
    python video_processing.py --clips-dir MSVD_data/Video/test \
        --output-index data/bin/index_msvd.faiss \
        --output-metadata data/bin/metadata_msvd.json

Protocol (standard MSVD multi-caption, as in CLIP4Clip / X-CLIP):
    t2v: EVERY caption (27,763) is a query, ranked against the 670 test
         videos. Rank = number of videos whose score >= the correct one
         (ties count against it, so ranks are never optimistic).
    v2t: every video is a query against every caption; its rank is the
         rank of its best-scoring own caption among all captions.
    --first-caption: one query per video (its first caption), the same
         protocol as evaluation_textVR_full.py, for a direct comparison.

Score: cos(Qwen(caption, VIDEO_INSTRUCTION), video vector), same instruction
as the index. Caption embeddings are cached in data/bin/ so a re-run only
re-scores.

Metrics: R@1, R@5, R@10, MRR, MdR, Mean Rank.

Example:
    python evaluation_msvd.py
    python evaluation_msvd.py --first-caption
    python evaluation_msvd.py --limit 50 --batch-size 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from evaluation_textVR_full import (
    MODEL_NAME,
    VIDEO_INSTRUCTION,
    calculate_metrics,
    encode_texts,
    load_json,
    load_model,
    load_video_vectors,
)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BIN_DIR = PROJECT_ROOT / "data" / "bin"

JSON_FILE = PROJECT_ROOT / "MSVD_data" / "msvd_test.json"
INDEX_FILE = BIN_DIR / "index_msvd.faiss"
METADATA_FILE = BIN_DIR / "metadata_msvd.json"
CACHE_FILE = BIN_DIR / "msvd_caption_emb.npz"

DEFAULT_BATCH_SIZE = 16
TOP_K_SAVED = 5


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_records(json_file):
    """
    [{"video_id", "captions": [...]}], one per test video.
    """
    records = []

    for item in load_json(json_file):
        video_id = str(item.get("video_id") or Path(item.get("video", "")).stem)
        captions = [str(c).strip() for c in item.get("caption") or [] if str(c).strip()]

        if video_id and captions:
            records.append({"video_id": video_id, "captions": captions})

    return records


def encode_captions(captions, embed_dim, batch_size, cache_file):
    """
    Caption embeddings, reused from cache_file when it holds exactly the
    same captions with the same model and dimension.
    """
    key = f"{MODEL_NAME}|{embed_dim}|{VIDEO_INSTRUCTION}"

    if cache_file and cache_file.exists():
        cached = np.load(cache_file, allow_pickle=False)
        if str(cached["key"]) == key and list(cached["captions"]) == captions:
            print(f"Caption embeddings loaded from cache: {cache_file}")
            return cached["embeddings"]

    model = load_model(embed_dim)

    print("\nEncoding captions...")
    embeddings = encode_texts(
        model, captions, VIDEO_INSTRUCTION, batch_size, "Caption"
    )

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_file,
            key=np.array(key),
            captions=np.array(captions),
            embeddings=embeddings,
        )

    return embeddings


# ---------------------------------------------------------------------
# Ranks
# ---------------------------------------------------------------------
def t2v_ranks(sim, owner):
    """
    sim: captions x videos. owner[i] = column of caption i's video.
    """
    target = sim[np.arange(len(owner)), owner][:, None]

    return (sim >= target).sum(axis=1)


def v2t_ranks(sim, owner):
    """
    For each video: 1 + number of OTHER videos' captions scoring >= its
    best own caption.
    """
    ranks = np.empty(sim.shape[1], dtype=np.int64)

    for v in range(sim.shape[1]):
        column = sim[:, v]
        own = owner == v
        best_own = column[own].max()
        ranks[v] = 1 + int(np.sum(column[~own] >= best_own))

    return ranks


def print_metrics(metrics):
    print(f"{'Task':<8}{'N':>8}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'MdR':>8}{'MeanR':>9}")

    for name, m in metrics.items():
        print(
            f"{name:<8}{m['queries']:>8}"
            f"{m['R@1'] * 100:>7.2f}%"
            f"{m['R@5'] * 100:>7.2f}%"
            f"{m['R@10'] * 100:>7.2f}%"
            f"{m['MRR']:>8.4f}"
            f"{m['MdR']:>8.1f}"
            f"{m['MeanRank']:>9.2f}"
        )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate(args):
    print("=" * 70)
    print("MSVD EVALUATION (VLM only)")
    print("=" * 70)

    records = load_records(args.json)
    print(f"Test videos in JSON: {len(records)}")

    vectors, missing, embed_dim = load_video_vectors(
        args.index, args.metadata, [r["video_id"] for r in records]
    )

    if missing:
        print(
            f"WARNING: {len(missing)} test videos are not in the FAISS index "
            f"(failed to encode?), removed from queries and candidates."
        )
        missing_set = set(missing)
        records = [r for r in records if r["video_id"] not in missing_set]

    if not records:
        raise RuntimeError("No test video was found in the FAISS index.")

    if args.limit is not None:
        records = records[:args.limit]

    if args.first_caption:
        for r in records:
            r["captions"] = r["captions"][:1]

    video_ids = [r["video_id"] for r in records]
    video_matrix = np.stack([vectors[v] for v in video_ids]).astype(np.float32)

    captions, owner = [], []
    for v, r in enumerate(records):
        captions.extend(r["captions"])
        owner.extend([v] * len(r["captions"]))
    owner = np.asarray(owner, dtype=np.int64)

    protocol = "first caption per video" if args.first_caption else "all captions"
    print(
        f"Candidates: {len(video_ids)} videos   Queries: {len(captions)} ({protocol})   "
        f"Embedding: {embed_dim}-D"
    )

    start_time = time.time()

    cache_file = None if args.no_cache else args.cache
    caption_emb = encode_captions(captions, embed_dim, args.batch_size, cache_file)

    sim = caption_emb @ video_matrix.T

    ranks = {"t2v": t2v_ranks(sim, owner), "v2t": v2t_ranks(sim, owner)}
    metrics = {name: calculate_metrics(r) for name, r in ranks.items()}

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print_metrics(metrics)

    print("\nt2v rank distribution:")
    for threshold in [t for t in (1, 5, 10, 50, 100) if t < len(video_ids)] + [len(video_ids)]:
        count = int(np.sum(ranks["t2v"] <= threshold))
        print(
            f"  Rank <= {threshold:4d}: "
            f"{count:6d}/{len(captions)} ({100.0 * count / len(captions):6.2f}%)"
        )

    # Hardest videos: mean t2v rank of their captions.
    mean_rank_per_video = np.bincount(owner, weights=ranks["t2v"]) / np.bincount(owner)
    print("\nHardest videos (mean t2v rank over their captions):")
    for v in np.argsort(-mean_rank_per_video)[:10]:
        print(f"  {video_ids[v]:<28}{mean_rank_per_video[v]:>8.1f}   \"{records[v]['captions'][0]}\"")

    # -----------------------------------------------------------------
    # Detailed results
    # -----------------------------------------------------------------
    results = []

    for q, caption in enumerate(captions):
        top = np.argsort(-sim[q], kind="stable")[:TOP_K_SAVED]

        results.append(
            {
                "query_index": q,
                "video_id": video_ids[owner[q]],
                "query": caption,
                "rank": int(ranks["t2v"][q]),
                f"top_{TOP_K_SAVED}": [
                    {"video_id": video_ids[c], "score": round(float(sim[q, c]), 4)}
                    for c in top
                ],
            }
        )

    output_file = args.output or (
        PROJECT_ROOT
        / ("evaluation_msvd_first_caption_results.json" if args.first_caption
           else "evaluation_msvd_results.json")
    )

    output = {
        "evaluation": {
            "dataset": "MSVD (test, official split)",
            "candidate_source": str(args.json),
            "protocol": protocol,
            "candidate_videos": len(video_ids),
            "queries": len(captions),
            "model": MODEL_NAME,
            "embedding_dimension": embed_dim,
            "video_instruction": VIDEO_INSTRUCTION,
            "score": "cosine(caption embedding, video embedding)",
            "rank": "t2v: #videos with score >= target; "
                    "v2t: 1 + #other captions with score >= best own caption",
            "files": {"index": str(args.index), "metadata": str(args.metadata)},
        },
        "metrics": metrics,
        "timing": {"encoding_and_scoring_seconds": elapsed},
        "per_video_mean_t2v_rank": {
            video_ids[v]: round(float(mean_rank_per_video[v]), 2)
            for v in range(len(video_ids))
        },
        "results": results,
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to:\n{output_file}")

    return metrics


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-Embedding text-video retrieval on MSVD."
    )
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--index", type=Path, default=INDEX_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument(
        "--first-caption",
        action="store_true",
        help="One query per video (its first caption) instead of all captions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Caption embedding batch size. Reduce to 4 or 2 if VRAM is insufficient.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N videos for a quick test.",
    )
    parser.add_argument("--cache", type=Path, default=CACHE_FILE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    for name in ("json", "index", "metadata"):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"--{name} not found: {path}")

    evaluate(args)


if __name__ == "__main__":
    main()
