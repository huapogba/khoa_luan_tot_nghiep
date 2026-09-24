"""
MSR-VTT 1K full evaluation: VLM + OCR + audio score fusion.

Only loads files produced offline, no video is reprocessed:
    data/msrvtt_test_1k.json        -> queries (caption) + candidate pool
    data/bin/index_512.faiss        -> video vectors (video_processing.py)
    data/bin/metadata_512.json      -> video_id of every FAISS row
    data/bin/metadata_ocr.json      -> OCR text per video (ocr_processing.py)
    data/bin/metadata_audio.json    -> transcript per video (audio_processing.py)

Evaluation (same protocol as evaluate_msrvtt_1k_json.py):
    caption -> retrieve the correct video among the videos in the JSON.
    Every candidate is scored, so the exact rank is known.

Metrics: R@1, R@5, R@10, MRR, MdR, Mean Rank, reported for:
    vlm / ocr / audio -> each source alone
    fusion            -> fixed weights for every query (rule [4])
    router            -> weights chosen from the query text (rule [6]);
                         the detailed results file follows this ranking.

========================================================================
SCORE FUSION RULES
========================================================================
For every query q and every candidate video v:

  [1] Raw score per source (all cosine similarity, vectors L2-normalized)
        vlm(q, v)   = cos(Qwen(q, VIDEO_INSTRUCTION), video vector of v)
        ocr(q, v)   = cos(Qwen(q, TEXT_INSTRUCTION),  Qwen(OCR text of v))
        audio(q, v) = cos(Qwen(q, TEXT_INSTRUCTION),  Qwen(transcript of v))
      OCR / audio text is encoded here with the same Qwen model, because
      the offline files only store text, not vectors.

  [2] Missing source
      A video with no OCR text (or no speech) has NO score for that
      source. After normalization it contributes 0.

  [3] Per-query min-max normalization
      The three sources have different cosine ranges (e.g. vlm ~0.1-0.5,
      text-text ~0.3-0.8), so raw values cannot be added directly.
      For each query and each source, over the candidates that have that
      source:
          norm = (raw - min) / (max - min)          -> [0, 1]
      Best candidate for that source = 1, worst = 0.

  [4] Weighted sum (default weights 0.4 / 0.3 / 0.3, see --weights)
          final(q, v) = 0.4 * vlm_norm + 0.3 * ocr_norm + 0.3 * audio_norm

  [5] Ranking
      Candidates are sorted by final score, highest first.
      Rank of the correct video = number of candidates whose score is
      >= its score (ties count against it, so ranks are never optimistic).

  [6] Query router (decides which search method to trust more)
      The query text is matched against keyword lists (OCR_KEYWORDS,
      AUDIO_KEYWORDS, English + Vietnamese, whole-word, case-insensitive):
          mentions on-screen text only  -> route "ocr"    0.3 / 0.5 / 0.2
          mentions speech/singing only  -> route "audio"  0.3 / 0.2 / 0.5
          mentions both                 -> route "both"   0.3 / 0.35 / 0.35
          mentions neither              -> route "vlm"    --weights
                                           (default 0.4 / 0.3 / 0.3)
      The router only re-weights; no source is switched off, so a wrong
      route cannot remove the correct video from the ranking.
      Keyword-based on purpose: offline, deterministic, reproducible.
========================================================================

Example:
    python evaluate_msrvtt_1K_full.py
    python evaluate_msrvtt_1K_full.py --weights 0.5 0.25 0.25
    python evaluate_msrvtt_1K_full.py --limit 50 --batch-size 4
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BIN_DIR = PROJECT_ROOT / "data" / "bin"

JSON_FILE = PROJECT_ROOT / "data" / "msrvtt_test_1k.json"
INDEX_FILE = BIN_DIR / "index_512.faiss"
METADATA_FILE = BIN_DIR / "metadata_512.json"
OCR_FILE = BIN_DIR / "metadata_ocr.json"
AUDIO_FILE = BIN_DIR / "metadata_audio.json"

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

# Must match the instruction used by video_processing.py for the index.
VIDEO_INSTRUCTION = "Represent the video for retrieval."
TEXT_INSTRUCTION = "Represent the text for retrieval."

# OCR / transcript text longer than this is truncated when encoded.
MAX_TEXT_TOKENS = 512

SOURCES = ("vlm", "ocr", "audio")
DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)
DEFAULT_BATCH_SIZE = 8

# ---------------------------------------------------------------------
# Query router (fusion rule [6])
# Keywords are regex fragments matched as whole words. Inflections are
# listed explicitly: an open suffix like "sing\w*" would also hit
# "single", "yell\w*" -> "yellow", "text\w*" -> "texture".
# ---------------------------------------------------------------------
OCR_KEYWORDS = [
    # English
    r"texts?", r"words?", r"writ(?:e|es|ing|ten)", r"wrote", r"letters?",
    r"titles?", r"subtitles?", r"captions?", r"logos?", r"labels?",
    r"signs?", r"signboards?", r"banners?", r"posters?", r"headlines?",
    # "on (the) screen" is left out: MSR-VTT uses it for "is shown",
    # not for on-screen text.
    r"menus?", r"numbers?", r"digits?", r"score\s?boards?", r"slideshows?",
    r"screenshots?",
    # Vietnamese
    r"chữ", r"văn bản", r"tiêu đề", r"phụ đề", r"biển", r"nhãn",
    r"bảng", r"con số", r"dòng chữ",
]

AUDIO_KEYWORDS = [
    # English
    r"talk(?:s|ing|ed)?", r"speak(?:s|ing|er|ers)?", r"spoke", r"speech(?:es)?",
    r"say(?:s|ing)?", r"said", r"tell(?:s|ing)?", r"told",
    r"explain(?:s|ing|ed)?", r"discuss(?:es|ing|ed|ion)?",
    r"interview(?:s|ing|ed|er)?", r"narrat(?:e|es|ing|ed|or|ion)",
    r"voices?", r"sing(?:s|ing|er|ers)?", r"sang", r"songs?", r"lyrics?",
    r"rap(?:s|ping|ped|per|pers)?", r"announc(?:e|es|ing|ed|er|ers)",
    r"commentat(?:or|ors|ing)", r"commentary", r"conversations?",
    r"lectur(?:e|es|ing|er)", r"describ(?:e|es|ing|ed)",
    r"reporters?", r"shout(?:s|ing|ed)?", r"yell(?:s|ing|ed)?",
    r"chat(?:s|ting|ted)?", r"podcasts?", r"news",
    # Vietnamese
    r"nói", r"giọng", r"lời", r"hát", r"bài hát", r"phỏng vấn",
    r"thuyết minh", r"bình luận", r"giải thích", r"trò chuyện", r"kể",
]

# route -> (vlm, ocr, audio). Route "vlm" uses --weights.
ROUTE_WEIGHTS = {
    "ocr": (0.3, 0.5, 0.2),
    "audio": (0.3, 0.2, 0.5),
    "both": (0.3, 0.35, 0.35),
}


def compile_keywords(keywords):
    return re.compile(
        r"(?<!\w)(?:" + "|".join(keywords) + r")(?!\w)",
        re.IGNORECASE,
    )


OCR_PATTERN = compile_keywords(OCR_KEYWORDS)
AUDIO_PATTERN = compile_keywords(AUDIO_KEYWORDS)


# ---------------------------------------------------------------------
# Loading offline files
# ---------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list.")

    return data


def load_queries(json_file):
    records = load_json(json_file)

    unique = {}

    for i, item in enumerate(records):
        if not item.get("video_id") or not item.get("caption"):
            raise ValueError(f"Record {i} needs a non-empty video_id and caption.")

        # Keep the first caption of each video.
        unique.setdefault(item["video_id"], item)

    if len(unique) != len(records):
        print(f"WARNING: {len(records) - len(unique)} duplicate video IDs removed.")

    return list(unique.values())


def load_video_vectors(index_file, metadata_file, video_ids):
    """
    Returns (vectors aligned with video_ids, missing ids, dimension).
    Row i of the FAISS index belongs to metadata[i].
    """
    index = faiss.read_index(str(index_file))
    metadata = load_json(metadata_file)

    if len(metadata) != index.ntotal:
        raise ValueError(
            f"FAISS index and metadata differ: {index.ntotal} vs {len(metadata)}"
        )

    row_of = {item.get("video_id"): i for i, item in enumerate(metadata)}
    all_vectors = index.reconstruct_n(0, index.ntotal)

    vectors = {}
    missing = []

    for video_id in video_ids:
        if video_id in row_of:
            vectors[video_id] = all_vectors[row_of[video_id]]
        else:
            missing.append(video_id)

    return vectors, missing, index.d


def load_source_texts(path, text_field, video_ids):
    """
    One text per candidate video ("" when the video has no OCR / speech).
    """
    by_id = {item.get("video_id"): item for item in load_json(path)}

    texts = []

    for video_id in video_ids:
        raw = str(by_id.get(video_id, {}).get(text_field, "") or "")

        # OCR repeats the same line across sampled frames; keep one copy.
        lines = dict.fromkeys(
            line.strip() for line in raw.splitlines() if line.strip()
        )

        texts.append(" ".join(lines))

    return texts


# ---------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------
def load_model(embed_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {MODEL_NAME} ({embed_dim}-D) on {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        truncate_dim=embed_dim,
    )
    model.max_seq_length = MAX_TEXT_TOKENS

    return model


def encode_texts(model, texts, instruction, batch_size, label):
    chunks = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        chunks.append(
            model.encode(
                batch,
                batch_size=len(batch),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                prompt=instruction,
            ).astype(np.float32)
        )

        print(f"\r  {label}: {start + len(batch)}/{len(texts)}", end="", flush=True)

    print()

    return np.vstack(chunks)


def text_source_scores(model, query_emb, texts, batch_size, label):
    """
    Raw cosine scores (queries x candidates). NaN = video has no text
    for this source (fusion rule [2]).
    """
    scores = np.full((len(query_emb), len(texts)), np.nan, dtype=np.float32)

    with_text = [i for i, text in enumerate(texts) if text]

    print(f"  {label}: {len(with_text)}/{len(texts)} videos have text")

    if with_text:
        doc_emb = encode_texts(
            model,
            [texts[i] for i in with_text],
            TEXT_INSTRUCTION,
            batch_size,
            f"Encoding {label} text",
        )
        scores[:, with_text] = query_emb @ doc_emb.T

    return scores


# ---------------------------------------------------------------------
# Fusion + metrics
# ---------------------------------------------------------------------
def minmax_per_query(raw):
    """
    Fusion rules [2] + [3]: per-query min-max over candidates that have
    the source; candidates without it get 0.
    """
    valid = ~np.isnan(raw)

    if not valid.any():
        return np.zeros_like(raw)

    lo = np.nanmin(raw, axis=1, keepdims=True)
    hi = np.nanmax(raw, axis=1, keepdims=True)
    span = hi - lo

    # Only one distinct value -> every candidate that has it is "best".
    norm = np.where(span > 1e-8, (raw - lo) / np.where(span > 1e-8, span, 1.0), 1.0)

    return np.where(valid, norm, 0.0).astype(np.float32)


def route_query(query, default_weights):
    """
    Fusion rule [6]. Returns (route name, matched keywords, weights).
    """
    ocr_hits = sorted({m.lower() for m in OCR_PATTERN.findall(query)})
    audio_hits = sorted({m.lower() for m in AUDIO_PATTERN.findall(query)})

    if ocr_hits and audio_hits:
        route = "both"
    elif ocr_hits:
        route = "ocr"
    elif audio_hits:
        route = "audio"
    else:
        route = "vlm"

    weights = ROUTE_WEIGHTS.get(route, default_weights)

    return route, ocr_hits + audio_hits, dict(zip(SOURCES, weights))


def compute_ranks(scores):
    """
    Fusion rule [5]. Query i's correct video is candidate i.
    """
    target = np.diag(scores)[:, None]

    return (scores >= target).sum(axis=1)


def calculate_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)

    return {
        "queries": int(len(ranks)),
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MRR": float(np.mean(1.0 / ranks)),
        "MdR": float(np.median(ranks)),
        "MeanRank": float(np.mean(ranks)),
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate(args):
    weights = dict(zip(SOURCES, args.weights))

    print("=" * 70)
    print("MSR-VTT 1K FULL EVALUATION (VLM + OCR + AUDIO)")
    print("=" * 70)
    print(f"Weights: {weights}")

    # -----------------------------------------------------------------
    # Queries + candidate pool
    # -----------------------------------------------------------------
    records = load_queries(args.json)

    if args.limit is not None:
        records = records[:args.limit]

    print(f"Queries / candidates: {len(records)}")

    vectors, missing, embed_dim = load_video_vectors(
        args.index,
        args.metadata,
        [r["video_id"] for r in records],
    )

    if missing:
        print(
            f"WARNING: {len(missing)} videos missing from the FAISS index, "
            f"removed from queries and candidates: {missing[:10]}"
        )
        missing_set = set(missing)
        records = [r for r in records if r["video_id"] not in missing_set]

    if not records:
        raise RuntimeError("No JSON video was found in the FAISS index.")

    candidate_ids = [r["video_id"] for r in records]
    queries = [r["caption"] for r in records]

    video_matrix = np.stack([vectors[v] for v in candidate_ids]).astype(np.float32)

    ocr_texts = load_source_texts(args.ocr, "ocr", candidate_ids)
    audio_texts = load_source_texts(args.audio, "audio", candidate_ids)

    # -----------------------------------------------------------------
    # Raw scores per source (fusion rule [1])
    # -----------------------------------------------------------------
    model = load_model(embed_dim)

    start_time = time.time()

    print("\nEncoding captions...")
    query_video_emb = encode_texts(
        model, queries, VIDEO_INSTRUCTION, args.batch_size, "Query (video)"
    )
    query_text_emb = encode_texts(
        model, queries, TEXT_INSTRUCTION, args.batch_size, "Query (text)"
    )

    print("\nScoring sources...")
    raw = {
        "vlm": query_video_emb @ video_matrix.T,
        "ocr": text_source_scores(
            model, query_text_emb, ocr_texts, args.batch_size, "ocr"
        ),
        "audio": text_source_scores(
            model, query_text_emb, audio_texts, args.batch_size, "audio"
        ),
    }

    scoring_time = time.time() - start_time

    # -----------------------------------------------------------------
    # Normalize + weighted sum (fusion rules [2]-[4])
    # -----------------------------------------------------------------
    norm = {source: minmax_per_query(raw[source]) for source in SOURCES}

    fused = sum(weights[source] * norm[source] for source in SOURCES)

    # -----------------------------------------------------------------
    # Query router: per-query weights (fusion rule [6])
    # -----------------------------------------------------------------
    routes = [route_query(query, args.weights) for query in queries]

    route_weights = {
        source: np.array([r[2][source] for r in routes], dtype=np.float32)[:, None]
        for source in SOURCES
    }

    routed = sum(route_weights[source] * norm[source] for source in SOURCES)

    # -----------------------------------------------------------------
    # Ranks + metrics (fusion rule [5])
    # -----------------------------------------------------------------
    ranks = {source: compute_ranks(norm[source]) for source in SOURCES}
    ranks["fusion"] = compute_ranks(fused)
    ranks["router"] = compute_ranks(routed)

    metrics = {name: calculate_metrics(r) for name, r in ranks.items()}

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Videos: {len(candidate_ids)}   Queries: {len(queries)}")
    print(
        f"OCR text: {sum(1 for t in ocr_texts if t)} videos   "
        f"Speech: {sum(1 for t in audio_texts if t)} videos"
    )
    print()
    print(f"{'Source':<8}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'MdR':>8}{'MeanR':>9}")

    for name, m in metrics.items():
        print(
            f"{name:<8}"
            f"{m['R@1'] * 100:>7.2f}%"
            f"{m['R@5'] * 100:>7.2f}%"
            f"{m['R@10'] * 100:>7.2f}%"
            f"{m['MRR']:>8.4f}"
            f"{m['MdR']:>8.1f}"
            f"{m['MeanRank']:>9.2f}"
        )

    # Per route: fixed-weight fusion vs router on the same queries.
    route_names = [r[0] for r in routes]

    print("\nRouter decisions (R@1 fusion -> router, MRR fusion -> router):")

    for route in ("vlm", "ocr", "audio", "both"):
        idx = [i for i, name in enumerate(route_names) if name == route]

        if not idx:
            print(f"  {route:<6}: 0 queries")
            continue

        before = calculate_metrics(ranks["fusion"][idx])
        after = calculate_metrics(ranks["router"][idx])

        print(
            f"  {route:<6}: {len(idx):4d} queries | "
            f"R@1 {before['R@1'] * 100:6.2f}% -> {after['R@1'] * 100:6.2f}% | "
            f"MRR {before['MRR']:.4f} -> {after['MRR']:.4f}"
        )

    print("\nRouter rank distribution:")

    for threshold in [1, 5, 10, 50, 100, 500, 1000]:
        count = int(np.sum(ranks["router"] <= threshold))
        print(
            f"  Rank <= {threshold:4d}: "
            f"{count:4d}/{len(queries)} ({100.0 * count / len(queries):6.2f}%)"
        )

    # -----------------------------------------------------------------
    # Detailed results
    # -----------------------------------------------------------------
    results = []

    for q, record in enumerate(records):
        top = np.argsort(-routed[q], kind="stable")[:10]
        route, matched, query_weights = routes[q]

        results.append(
            {
                "query_index": q,
                "query_video_id": record["video_id"],
                "caption": record["caption"],
                "category": record.get("category"),
                "route": route,
                "matched_keywords": matched,
                "weights": query_weights,
                "rank": int(ranks["router"][q]),
                "rank_fixed_fusion": int(ranks["fusion"][q]),
                "rank_per_source": {s: int(ranks[s][q]) for s in SOURCES},
                "reciprocal_rank": 1.0 / int(ranks["router"][q]),
                "top_10": [
                    {
                        "rank": i + 1,
                        "video_id": candidate_ids[c],
                        "score": round(float(routed[q, c]), 4),
                        "vlm": round(float(norm["vlm"][q, c]), 4),
                        "ocr": round(float(norm["ocr"][q, c]), 4),
                        "audio": round(float(norm["audio"][q, c]), 4),
                    }
                    for i, c in enumerate(top)
                ],
            }
        )

    output_file = args.output or (
        Path(args.json).parent / "evaluation_msrvtt_1k_full_results.json"
    )

    output = {
        "evaluation": {
            "dataset": "MSR-VTT",
            "candidate_source": str(args.json),
            "candidate_videos": len(candidate_ids),
            "queries": len(queries),
            "model": MODEL_NAME,
            "embedding_dimension": embed_dim,
            "video_instruction": VIDEO_INSTRUCTION,
            "text_instruction": TEXT_INSTRUCTION,
            "weights": weights,
            "fusion": "per-query min-max per source, missing source = 0, "
                      "weighted sum, rank = #candidates with score >= target",
            "router": {
                "rule": "keyword match on the query picks per-query weights; "
                        "route 'vlm' uses the default weights",
                "route_weights": {
                    route: dict(zip(SOURCES, w))
                    for route, w in ROUTE_WEIGHTS.items()
                },
                "route_counts": {
                    route: route_names.count(route)
                    for route in ("vlm", "ocr", "audio", "both")
                },
                "ocr_keywords": OCR_KEYWORDS,
                "audio_keywords": AUDIO_KEYWORDS,
            },
            "files": {
                "index": str(args.index),
                "metadata": str(args.metadata),
                "ocr": str(args.ocr),
                "audio": str(args.audio),
            },
        },
        "metrics": metrics,
        "timing": {"encoding_and_scoring_seconds": scoring_time},
        "results": results,
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to:\n{output_file}")

    return metrics


def main():
    # Captions / OCR text may not fit the Windows console codepage.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Evaluate VLM + OCR + audio fusion on MSR-VTT 1K."
    )
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--index", type=Path, default=INDEX_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--ocr", type=Path, default=OCR_FILE)
    parser.add_argument("--audio", type=Path, default=AUDIO_FILE)
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_WEIGHTS),
        metavar=("VLM", "OCR", "AUDIO"),
        help="Fusion weights (default: 0.4 0.3 0.3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Text embedding batch size. Reduce to 4 or 2 if VRAM is insufficient.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N records for a quick test.",
    )
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    for name in ("json", "index", "metadata", "ocr", "audio"):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"--{name} not found: {path}")

    evaluate(args)


if __name__ == "__main__":
    main()
