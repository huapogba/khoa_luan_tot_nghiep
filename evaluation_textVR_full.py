"""
TextVR full evaluation: VLM + OCR score fusion (English only).

Only loads files produced offline, no video is reprocessed:
    TextVR_data/TextVR_test.json        -> queries (caption) + candidate pool
    data/bin/index_textVR.faiss         -> video vectors (video_processing.py, 768-D)
    data/bin/metadata_textVR.json       -> video_id of every FAISS row
    data/bin/metadata_textvr_ocr.json   -> OCR text per video (ocr_processing_textVR.py)

Evaluation:
    caption -> retrieve the correct video among every test video of
    TextVR_test.json (2727 video-caption pairs). Every candidate is
    scored, so the exact rank is known. Metrics are also reported per
    category (Activity, Cooking, Driving, Games, News_Movie, Sports,
    Street_View_Indoor, Street_View_Outdoor).

The video_id of a test entry is the file stem of its "path"
("Sports/G7ll8RnSgvU_00300_00310.mp4" -> "G7ll8RnSgvU_00300_00310"),
same as video_processing.py / ocr_processing_textVR.py.

Metrics: R@1, R@5, R@10, MRR, MdR, Mean Rank, reported for:
    vlm / ocr -> each source alone (raw cosine / BM25)
    fusion    -> VLM + confidence bonus from OCR (rule [4]);
                 the detailed results file follows this ranking.

========================================================================
SCORE FUSION RULES
========================================================================
VLM is the backbone. OCR never replaces it; it only adds a bonus to a
video when its match is clearly stronger than the rest of the pool.
With no confident match the fused ranking equals VLM exactly.

For every query q and every candidate video v:

  [1] Raw score per source
        vlm(q, v) = cos(Qwen(q, VIDEO_INSTRUCTION), video vector of v)
        ocr(q, v) = BM25(caption words of q, OCR words of v)
      OCR uses lexical BM25, not embeddings: on-screen text is useful
      for exact matches (names, brands, numbers, rare words), while a
      semantic match between a scene caption and OCR fragments is noise.
      Stopwords are dropped and IDF down-weights words common to many
      videos, so only shared rare words score. A video with text but no
      shared word scores 0.

  [2] Per-query z-score (VLM)
      For each query, over all candidates:
          z_vlm = (vlm - mean) / std

  [3] Confidence bonus (OCR)
          bonus_ocr = max(0, bm25_ocr - ocr_min)    (see --ocr-min-score)
      Thresholded on the raw BM25 score instead of a z-score: BM25 is
      sparse (most videos score 0), so any single shared word would
      already be a large z outlier. A video with no OCR text gets
      bonus 0: neutral, not a penalty.

  [3b] Caption-side rarity gate
      A bonus is kept only if the shared words include at least one
      RARE caption word:
          caption_df(t) = number of captions containing t
          t is rare      <=> caption_df(t) <= max(1, rare_ratio * #captions)
      (see --rare-ratio). Names, brands and specific words pass; generic
      scene words ("man", "people", "street") do not.
      Only caption text is used, never the ground-truth pairing.

  [4] Final score (default ocr_min = 1, b_ocr = 0.2, see --bonus-weight)
      Chosen with 5-fold cross-validation over the TextVR test queries
      (Kwai OCR): every fold picked the same pair. TextVR captions are a
      single sentence, so BM25 scores are much lower than on MultiVENT's
      long descriptions; the MultiVENT values (15, 0.05) almost never fire.
      R@1 stays ~34-35.5% for ocr_min 0-4 and b_ocr 0.15-0.2.
          final(q, v) = z_vlm + b_ocr * bonus_ocr
      z_vlm is a per-query affine transform of the raw VLM score, so
      with every bonus 0 the ranking is identical to VLM alone.

  [5] Ranking
      Candidates are sorted by final score, highest first.
      Rank of the correct video = number of candidates whose score is
      >= its score (ties count against it, so ranks are never optimistic).
========================================================================

Example:
    python evaluation_textVR_full.py
    python evaluation_textVR_full.py --ocr-min-score 15 --bonus-weight 0.05   # MultiVENT values
    python evaluation_textVR_full.py --limit 50 --batch-size 4
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
from sentence_transformers.base.modality import infer_modality


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BIN_DIR = PROJECT_ROOT / "data" / "bin"

JSON_FILE = PROJECT_ROOT / "TextVR_data" / "TextVR_test.json"
INDEX_FILE = BIN_DIR / "index_textVR.faiss"
METADATA_FILE = BIN_DIR / "metadata_textVR.json"
OCR_FILE = BIN_DIR / "metadata_textvr_ocr_2.json"

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

# Must match the instruction used by video_processing.py for the index.
VIDEO_INSTRUCTION = "Represent the video for retrieval."

SOURCES = ("vlm", "ocr")
DEFAULT_BONUS_WEIGHT = 0.2
DEFAULT_OCR_MIN_SCORE = 1.0
DEFAULT_RARE_RATIO = 0.01
DEFAULT_BATCH_SIZE = 8

# BM25 for OCR (fusion rule [1]); standard Okapi parameters.
BM25_K1 = 1.2
BM25_B = 0.75
MIN_TOKEN_LENGTH = 2

TOKEN_PATTERN = re.compile(r"\w+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "from", "by", "about", "as", "into", "onto", "over",
    "up", "down", "out", "off", "is", "are", "was", "were", "be", "been",
    "being", "am", "has", "have", "had", "do", "does", "did", "it", "its",
    "this", "that", "these", "those", "there", "here", "he", "she", "they",
    "them", "his", "her", "their", "him", "we", "you", "your", "our", "i",
    "me", "my", "who", "what", "which", "while", "when", "where", "how",
    "some", "other", "another", "each", "all", "very", "then", "than",
    "not", "no", "so", "if", "can", "will", "just", "also",
}


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
    """
    One query per test video: its (first) caption. The category is the
    folder part of "path".
    """
    records = load_json(json_file)

    unique = {}
    empty = 0

    for item in records:
        path = str(item.get("path") or "").strip()
        captions = item.get("captions_info") or []
        query = str(captions[0].get("caption") or "").strip() if captions else ""

        if not path or not query:
            empty += 1
            continue

        video_id = Path(path).stem
        category = Path(path).parent.name or "unknown"

        unique.setdefault(
            video_id,
            {"video_id": video_id, "path": path, "category": category, "query": query},
        )

    if empty:
        print(f"WARNING: {empty} records without path / caption skipped.")

    duplicates = len(records) - empty - len(unique)
    if duplicates:
        print(f"WARNING: {duplicates} duplicate video IDs removed.")

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

    row_of = {str(item.get("video_id")): i for i, item in enumerate(metadata)}
    all_vectors = index.reconstruct_n(0, index.ntotal)

    vectors = {}
    missing = []

    for video_id in video_ids:
        if video_id in row_of:
            vectors[video_id] = all_vectors[row_of[video_id]]
        else:
            missing.append(video_id)

    return vectors, missing, index.d


def load_ocr_texts(path, video_ids):
    """
    One text per candidate video ("" when the video has no OCR text).
    """
    by_id = {str(item.get("video_id")): item for item in load_json(path)}

    texts = []

    for video_id in video_ids:
        raw = str(by_id.get(video_id, {}).get("ocr", "") or "")

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

    # Same dtype as video_processing.py used for the index.
    model_kwargs = {"torch_dtype": torch.bfloat16} if device == "cuda" else {}

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs=model_kwargs,
        truncate_dim=embed_dim,
    )

    return model


def as_text_input(text):
    """
    sentence-transformers guesses the modality of a string: a bare URL or
    a media path is loaded as a video / image / audio. Such strings are
    wrapped as {"text": ...} to force text; normal text is left unchanged.
    """
    if infer_modality(text) != "text":
        return {"text": text}

    return text


def encode_texts(model, texts, instruction, batch_size, label):
    chunks = []

    for start in range(0, len(texts), batch_size):
        batch = [as_text_input(text) for text in texts[start:start + batch_size]]

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


# ---------------------------------------------------------------------
# BM25 (OCR)
# ---------------------------------------------------------------------
def tokenize(text):
    text = URL_PATTERN.sub(" ", text.lower())

    return [
        token
        for token in TOKEN_PATTERN.findall(text)
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS
    ]


def rare_caption_terms(queries, rare_ratio):
    """
    Fusion rule [3b]: words that appear in few captions.
    """
    caption_df = {}
    for query in queries:
        for token in set(tokenize(query)):
            caption_df[token] = caption_df.get(token, 0) + 1

    max_df = max(1, int(rare_ratio * len(queries)))

    return {token for token, df in caption_df.items() if df <= max_df}


def bm25_source_scores(queries, texts, rare_terms, label):
    """
    Returns (raw BM25 scores, rare-match mask), both queries x candidates.
    Scores: NaN = video has no text. Query terms are counted once
    (binary query), so a word repeated in the caption is not over-weighted.
    Mask: True when the shared words include a rare caption word.
    """
    scores = np.full((len(queries), len(texts)), np.nan, dtype=np.float32)
    rare_match = np.zeros((len(queries), len(texts)), dtype=bool)

    doc_tokens = {i: tokenize(text) for i, text in enumerate(texts) if text}
    doc_tokens = {i: tokens for i, tokens in doc_tokens.items() if tokens}
    with_text = list(doc_tokens)

    print(f"  {label}: {len(with_text)}/{len(texts)} videos have usable words (BM25)")

    if not with_text:
        return scores, rare_match

    vocab = {}
    for tokens in doc_tokens.values():
        for token in tokens:
            vocab.setdefault(token, len(vocab))

    # Term frequency matrix (docs x vocab).
    tf = np.zeros((len(with_text), len(vocab)), dtype=np.float32)
    for row, i in enumerate(with_text):
        for token in doc_tokens[i]:
            tf[row, vocab[token]] += 1

    n_docs = len(with_text)
    df = (tf > 0).sum(axis=0)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

    doc_len = tf.sum(axis=1, keepdims=True)
    norm = BM25_K1 * (1.0 - BM25_B + BM25_B * doc_len / doc_len.mean())
    term_weight = idf * tf * (BM25_K1 + 1.0) / (tf + norm)

    # Binary query-term matrix (queries x vocab); unknown words are ignored.
    query_terms = np.zeros((len(queries), len(vocab)), dtype=np.float32)
    for q, query in enumerate(queries):
        for token in set(tokenize(query)):
            if token in vocab:
                query_terms[q, vocab[token]] = 1.0

    scores[:, with_text] = query_terms @ term_weight.T

    rare_mask = np.zeros(len(vocab), dtype=np.float32)
    for token, column in vocab.items():
        if token in rare_terms:
            rare_mask[column] = 1.0

    rare_match[:, with_text] = ((query_terms * rare_mask) @ (tf > 0).T) > 0

    return scores, rare_match


def matched_terms(query, text):
    return sorted(set(tokenize(query)) & set(tokenize(text)))


# ---------------------------------------------------------------------
# Fusion + metrics
# ---------------------------------------------------------------------
def zscore_per_query(raw):
    """
    Fusion rule [2]: per-query z-score over candidates.
    """
    mean = raw.mean(axis=1, keepdims=True)
    std = raw.std(axis=1, keepdims=True) + 1e-6

    return ((raw - mean) / std).astype(np.float32)


def confidence_bonus(raw, rare_match, min_score):
    """
    Fusion rules [3] + [3b]: only candidates whose BM25 score exceeds
    min_score AND that share a rare caption word get a bonus; videos
    without OCR text (NaN) get 0.
    """
    bonus = np.maximum(0.0, np.nan_to_num(raw, nan=0.0) - min_score)

    return np.where(rare_match, bonus, 0.0).astype(np.float32)


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
    print("=" * 70)
    print("TEXTVR FULL EVALUATION (VLM + OCR)")
    print("=" * 70)
    print(
        f"Bonus weight (ocr): {args.bonus_weight}   "
        f"min BM25 (ocr): {args.ocr_min_score}   "
        f"rare ratio: {args.rare_ratio}"
    )

    # -----------------------------------------------------------------
    # Queries + candidate pool
    # -----------------------------------------------------------------
    records = load_queries(args.json)

    print(f"Records in JSON: {len(records)}")

    vectors, missing, embed_dim = load_video_vectors(
        args.index,
        args.metadata,
        [r["video_id"] for r in records],
    )

    if missing:
        print(
            f"{len(missing)} JSON videos are not in the FAISS index, "
            f"removed from queries and candidates."
        )
        missing_set = set(missing)
        records = [r for r in records if r["video_id"] not in missing_set]

    if not records:
        raise RuntimeError("No JSON video was found in the FAISS index.")

    if args.limit is not None:
        records = records[:args.limit]

    print(f"Queries / candidates: {len(records)}   Embedding: {embed_dim}-D")

    candidate_ids = [r["video_id"] for r in records]
    queries = [r["query"] for r in records]

    video_matrix = np.stack([vectors[v] for v in candidate_ids]).astype(np.float32)

    ocr_texts = load_ocr_texts(args.ocr, candidate_ids)

    # -----------------------------------------------------------------
    # Raw scores per source (fusion rule [1])
    # -----------------------------------------------------------------
    model = load_model(embed_dim)

    start_time = time.time()

    print("\nEncoding queries (caption)...")
    query_video_emb = encode_texts(
        model, queries, VIDEO_INSTRUCTION, args.batch_size, "Query (video)"
    )

    print("\nScoring sources...")
    rare_terms = rare_caption_terms(queries, args.rare_ratio)
    print(f"  rare query words: {len(rare_terms)}")

    raw = {"vlm": query_video_emb @ video_matrix.T}
    raw["ocr"], rare_match = bm25_source_scores(queries, ocr_texts, rare_terms, "ocr")

    scoring_time = time.time() - start_time

    # -----------------------------------------------------------------
    # z-score + confidence bonus (fusion rules [2]-[4])
    # -----------------------------------------------------------------
    z_vlm = zscore_per_query(raw["vlm"])
    bonus = confidence_bonus(raw["ocr"], rare_match, args.ocr_min_score)

    fused = z_vlm + args.bonus_weight * bonus

    # -----------------------------------------------------------------
    # Ranks + metrics (fusion rule [5])
    # OCR alone cannot retrieve a video it has no text for: -inf.
    # -----------------------------------------------------------------
    ranks = {
        source: compute_ranks(np.nan_to_num(raw[source], nan=-np.inf))
        for source in SOURCES
    }
    ranks["fusion"] = compute_ranks(fused)

    metrics = {name: calculate_metrics(r) for name, r in ranks.items()}

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Videos: {len(candidate_ids)}   Queries: {len(queries)}")
    print(f"OCR text: {sum(1 for t in ocr_texts if t)} videos")
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

    # How often the bonus fires and whether it helps the correct video.
    active = bonus > 0
    print(
        f"\nBonus activity (ocr): {active.any(axis=1).sum():4d} queries with any bonus | "
        f"correct video boosted in {int(np.diag(active).sum())} queries"
    )

    helped = int(np.sum(ranks["fusion"] < ranks["vlm"]))
    hurt = int(np.sum(ranks["fusion"] > ranks["vlm"]))
    print(f"\nFusion vs VLM: better in {helped} queries, worse in {hurt} queries")

    # Per-category breakdown (the query is ranked against the whole pool).
    categories = np.array([r["category"] for r in records])
    metrics_by_category = {}

    print(f"\nPer category{'':<10}{'N':>6}{'R@1 vlm':>10}{'R@1 fus':>10}"
          f"{'R@10 vlm':>10}{'R@10 fus':>10}{'MRR fus':>9}")

    for category in sorted(set(categories)):
        mask = categories == category
        metrics_by_category[category] = {
            name: calculate_metrics(r[mask]) for name, r in ranks.items()
        }
        vlm_m = metrics_by_category[category]["vlm"]
        fus_m = metrics_by_category[category]["fusion"]
        print(
            f"  {category:<20}{int(mask.sum()):>6}"
            f"{vlm_m['R@1'] * 100:>9.2f}%{fus_m['R@1'] * 100:>9.2f}%"
            f"{vlm_m['R@10'] * 100:>9.2f}%{fus_m['R@10'] * 100:>9.2f}%"
            f"{fus_m['MRR']:>9.4f}"
        )

    print("\nFusion rank distribution:")

    thresholds = [t for t in (1, 5, 10, 50, 100, 500, 1000) if t < len(queries)]

    for threshold in thresholds + [len(queries)]:
        count = int(np.sum(ranks["fusion"] <= threshold))
        print(
            f"  Rank <= {threshold:4d}: "
            f"{count:4d}/{len(queries)} ({100.0 * count / len(queries):6.2f}%)"
        )

    # -----------------------------------------------------------------
    # Detailed results
    # -----------------------------------------------------------------
    results = []

    for q, record in enumerate(records):
        top = np.argsort(-fused[q], kind="stable")[:10]

        results.append(
            {
                "query_index": q,
                "query_video_id": record["video_id"],
                "query": record["query"],
                "category": record["category"],
                "path": record["path"],
                "rare_query_terms": sorted(
                    set(tokenize(record["query"])) & rare_terms
                ),
                "rank": int(ranks["fusion"][q]),
                "rank_per_source": {s: int(ranks[s][q]) for s in SOURCES},
                "reciprocal_rank": 1.0 / int(ranks["fusion"][q]),
                "top_10": [
                    {
                        "rank": i + 1,
                        "video_id": candidate_ids[c],
                        "score": round(float(fused[q, c]), 4),
                        "z_vlm": round(float(z_vlm[q, c]), 4),
                        "bm25_ocr": round(float(np.nan_to_num(raw["ocr"][q, c])), 4),
                        "bonus_ocr": round(float(bonus[q, c]), 4),
                        "ocr_matched_terms": matched_terms(
                            record["query"], ocr_texts[c]
                        ),
                    }
                    for i, c in enumerate(top)
                ],
            }
        )

    output_file = args.output or (PROJECT_ROOT / "evaluation_textVR_full_results.json")

    output = {
        "evaluation": {
            "dataset": "TextVR",
            "candidate_source": str(args.json),
            "query_field": "captions_info[0].caption",
            "candidate_videos": len(candidate_ids),
            "queries": len(queries),
            "model": MODEL_NAME,
            "embedding_dimension": embed_dim,
            "video_instruction": VIDEO_INSTRUCTION,
            "bonus_weight_ocr": args.bonus_weight,
            "ocr_min_score": args.ocr_min_score,
            "rare_ratio": args.rare_ratio,
            "rare_caption_terms": len(rare_terms),
            "text_matching": f"BM25 (k1={BM25_K1}, b={BM25_B}) for OCR, "
                             f"stopwords removed, URLs removed, "
                             f"tokens >= {MIN_TOKEN_LENGTH} chars",
            "fusion": "final = z_vlm + b_ocr * max(0, bm25_ocr - ocr_min_score); "
                      "bonus kept only if a shared word is rare among captions; "
                      "no OCR text = no bonus; "
                      "rank = #candidates with score >= target",
            "files": {
                "index": str(args.index),
                "metadata": str(args.metadata),
                "ocr": str(args.ocr),
            },
        },
        "metrics": metrics,
        "metrics_by_category": metrics_by_category,
        "fusion_vs_vlm": {"better": helped, "worse": hurt},
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
        description="Evaluate VLM + OCR confidence-bonus fusion on TextVR."
    )
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--index", type=Path, default=INDEX_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--ocr", type=Path, default=OCR_FILE)
    parser.add_argument(
        "--bonus-weight",
        type=float,
        default=DEFAULT_BONUS_WEIGHT,
        help=f"Weight of the OCR confidence bonus (default: {DEFAULT_BONUS_WEIGHT}).",
    )
    parser.add_argument(
        "--ocr-min-score",
        type=float,
        default=DEFAULT_OCR_MIN_SCORE,
        help=f"BM25 score above which OCR adds a bonus (default: {DEFAULT_OCR_MIN_SCORE:g}).",
    )
    parser.add_argument(
        "--rare-ratio",
        type=float,
        default=DEFAULT_RARE_RATIO,
        help="A caption word is rare if it appears in at most this fraction "
             f"of captions (default: {DEFAULT_RARE_RATIO}). A bonus needs a shared rare word.",
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
        help="Evaluate only the first N records for a quick test.",
    )
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    for name in ("json", "index", "metadata", "ocr"):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"--{name} not found: {path}")

    evaluate(args)


if __name__ == "__main__":
    main()
