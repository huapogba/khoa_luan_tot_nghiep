"""
MultiVENT full evaluation: VLM + OCR + audio score fusion.

Only loads files produced offline, no video is reprocessed:
    multivent_base.json             -> queries (video_description) + candidate pool
    data/bin/index_new_512.faiss    -> video vectors (video_processing.py, 768-D)
    data/bin/metadata_new_512.json  -> video_id of every FAISS row
    data/bin/metadata_new_ocr.json  -> OCR text per video (ocr_processing.py)
    data/bin/metadata_audio.json    -> transcript per video (audio_processing.py)

Evaluation:
    video_description -> retrieve the correct video among every video of
    multivent_base.json that is in the FAISS index. Every candidate is
    scored, so the exact rank is known. Metrics are also reported per
    language (english, korean, chinese, russian, arabic).

Multilingual text matching: words are split on \\w+, then
    - Chinese / Japanese runs (no spaces) -> character bigrams,
    - Korean words -> character bigrams (particles are glued to the noun:
      "지진이" / "지진을" both share "지진"),
    - Arabic words -> common attached prefixes stripped ("الزلزال" -> "زلزال"),
    - URLs are dropped.
The OCR engine outputs Arabic letters in reversed (left-to-right) order,
so Arabic words of the OCR text are reversed back before matching.
Video IDs with a yt-dlp format suffix ("abc.f609") are normalized ("abc").

Metrics: R@1, R@5, R@10, MRR, MdR, Mean Rank, reported for:
    vlm / ocr / audio -> each source alone (raw cosine / BM25)
    fusion            -> VLM + confidence bonus from OCR / audio (rule [4]);
                         the detailed results file follows this ranking.

========================================================================
SCORE FUSION RULES
========================================================================
VLM is the backbone. OCR / audio never replace it; they only add a
bonus to a video when their match is clearly stronger than the rest of
the pool. With no confident match the fused ranking equals VLM exactly.

For every query q and every candidate video v:

  [1] Raw score per source
        vlm(q, v)   = cos(Qwen(q, VIDEO_INSTRUCTION), video vector of v)
        ocr(q, v)   = BM25(caption words of q, OCR words of v)
        audio(q, v) = BM25(caption words of q, transcript words of v)
      OCR / audio use lexical BM25, not embeddings: on-screen text and
      speech are useful for exact matches (names, brands, numbers, rare
      words), while a semantic match between a scene caption and OCR
      fragments / spoken sentences is noise.
      Stopwords are dropped and IDF down-weights words common to many
      videos, so only shared rare words score. A video with text but no
      shared word scores 0. IDF is computed per source (OCR and
      transcripts are separate collections).

  [2] Per-query z-score (VLM)
      For each query, over all candidates:
          z_vlm = (vlm - mean) / std
      Puts VLM on a scale where the bonuses below are comparable
      across queries.

  [3] Confidence bonus (OCR / audio)
          bonus_ocr   = max(0, bm25_ocr   - ocr_min)    (default 15, --ocr-min-score)
          bonus_audio = max(0, bm25_audio - audio_min)  (default 15, --audio-min-score)
      Thresholded on the raw BM25 score instead of a z-score: BM25 is
      sparse (most videos score 0), so any single shared word would
      already be a large z outlier. The raw score grows with the number
      and rarity of shared words, so a high threshold keeps only
      multi-word / rare-word matches.
      A video with no OCR text / no speech gets bonus 0: neutral, not a
      penalty.

  [3b] Caption-side rarity gate
      BM25's IDF only looks at the video side: "game" or "song" are rare
      in transcripts, yet very common in captions ("someone is playing a
      game"), so one video saying "game" would be boosted for every
      gaming caption (a hub). A bonus is therefore kept only if the
      shared words include at least one RARE caption word:
          caption_df(t) = number of captions containing t
          t is rare      <=> caption_df(t) <= max(1, rare_ratio * #captions)
      (default rare_ratio = 0.01, see --rare-ratio). Names, brands and
      specific topics ("christie", "poll", "classroom") pass; generic
      scene words ("game", "playing", "song", "man", "people") do not.
      Only caption text is used, never the ground-truth pairing.
      ("caption" = the query text, here video_description.)

  [4] Final score (default b_ocr = b_audio = 0.05, see --bonus-weights)
      Chosen with 2-fold cross-validation on MultiVENT: long descriptions
      give large BM25 scores, so a small weight is enough to lift a
      strong match; 0.5 overrode VLM and hurt English queries.
          final(q, v) = z_vlm + b_ocr * bonus_ocr + b_audio * bonus_audio
      z_vlm is a per-query affine transform of the raw VLM score, so
      with every bonus 0 the ranking is identical to VLM alone.

  [5] Ranking
      Candidates are sorted by final score, highest first.
      Rank of the correct video = number of candidates whose score is
      >= its score (ties count against it, so ranks are never optimistic).
========================================================================

Example:
    python evaluate_msrvtt_1K_full.py
    python evaluate_msrvtt_1K_full.py --ocr-min-score 12 --audio-min-score 12
    python evaluate_msrvtt_1K_full.py --bonus-weights 0.3 0.3
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
from sentence_transformers.base.modality import infer_modality


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BIN_DIR = PROJECT_ROOT / "data" / "bin"

JSON_FILE = PROJECT_ROOT / "multivent_base.json"
INDEX_FILE = BIN_DIR / "index_new_512.faiss"
METADATA_FILE = BIN_DIR / "metadata_new_512.json"
OCR_FILE = BIN_DIR / "metadata_new_ocr.json"
AUDIO_FILE = BIN_DIR / "metadata_audio.json"

QUERY_FIELD = "video_description"

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

# Must match the instruction used by video_processing.py for the index.
VIDEO_INSTRUCTION = "Represent the video for retrieval."

SOURCES = ("vlm", "ocr", "audio")
BONUS_SOURCES = ("ocr", "audio")
DEFAULT_BONUS_WEIGHTS = (0.05, 0.05)
DEFAULT_OCR_MIN_SCORE = 15.0
DEFAULT_AUDIO_MIN_SCORE = 15.0
DEFAULT_RARE_RATIO = 0.01
DEFAULT_BATCH_SIZE = 8

# BM25 for OCR / audio (fusion rule [1]); standard Okapi parameters.
BM25_K1 = 1.2
BM25_B = 0.75
MIN_TOKEN_LENGTH = 2

TOKEN_PATTERN = re.compile(r"\w+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
# Chinese / Japanese kanji + kana: no spaces between words.
CJK_PATTERN = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]+")
HANGUL_PATTERN = re.compile(r"[가-힣]+")
ARABIC_PATTERN = re.compile(r"[؀-ۿ]+")
# Attached Arabic prefixes (and+the, with+the, the, ...), longest first.
ARABIC_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ب", "ل", "ف")
# yt-dlp leaves a format suffix on some downloads: "abc.f609" -> "abc".
FORMAT_SUFFIX_PATTERN = re.compile(r"\.f\d+$")

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


def normalize_video_id(video_id):
    return FORMAT_SUFFIX_PATTERN.sub("", str(video_id or "").strip())


def load_queries(json_file, query_field):
    records = load_json(json_file)

    unique = {}
    empty = 0

    for item in records:
        video_id = normalize_video_id(item.get("video_id"))
        query = str(item.get(query_field) or "").strip()

        if not video_id or not query:
            empty += 1
            continue

        # Keep the first description of each video.
        unique.setdefault(video_id, {**item, "video_id": video_id, "query": query})

    if empty:
        print(f"WARNING: {empty} records without video_id / {query_field} skipped.")

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

    row_of = {
        normalize_video_id(item.get("video_id")): i for i, item in enumerate(metadata)
    }
    all_vectors = index.reconstruct_n(0, index.ntotal)

    vectors = {}
    missing = []

    for video_id in video_ids:
        if video_id in row_of:
            vectors[video_id] = all_vectors[row_of[video_id]]
        else:
            missing.append(video_id)

    return vectors, missing, index.d


def fix_reversed_arabic(text):
    """
    The OCR engine reads Arabic left to right, so every Arabic word comes
    out with its letters reversed ("ةيرابخإلا" instead of "الإخبارية").
    """
    return ARABIC_PATTERN.sub(lambda match: match.group()[::-1], text)


def load_source_texts(path, text_field, video_ids, reverse_arabic=False):
    """
    One text per candidate video ("" when the video has no OCR / speech).
    """
    by_id = {normalize_video_id(item.get("video_id")): item for item in load_json(path)}

    texts = []

    for video_id in video_ids:
        raw = str(by_id.get(video_id, {}).get(text_field, "") or "")

        if reverse_arabic:
            raw = fix_reversed_arabic(raw)

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
    sentence-transformers guesses the modality of a string: a bare URL
    (e.g. a description that is only "https://www.youtube.com/watch?v=...")
    or a media path is loaded as a video / image / audio. Such strings are
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
# BM25 (OCR, audio)
# ---------------------------------------------------------------------
def split_cjk(word):
    """
    Splits runs of CJK characters into overlapping bigrams; other parts
    of the word (latin letters, digits, hangul, ...) stay whole.
    """
    parts = []
    last = 0

    for match in CJK_PATTERN.finditer(word):
        parts.append(word[last:match.start()])

        run = match.group()
        if len(run) == 1:
            parts.append(run)
        else:
            parts.extend(run[i:i + 2] for i in range(len(run) - 1))

        last = match.end()

    parts.append(word[last:])

    return parts


def char_bigrams(word):
    if len(word) <= 2:
        return [word]

    return [word[i:i + 2] for i in range(len(word) - 1)]


def strip_arabic_prefix(word):
    for prefix in ARABIC_PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 2:
            return word[len(prefix):]

    return word


def normalize_token(token):
    """
    Korean: particles are glued to the word, so match on character bigrams.
    Arabic: articles / conjunctions are glued to the word, so strip them.
    """
    if HANGUL_PATTERN.fullmatch(token):
        return char_bigrams(token)

    if ARABIC_PATTERN.fullmatch(token):
        return [strip_arabic_prefix(token)]

    return [token]


def tokenize(text):
    text = URL_PATTERN.sub(" ", text.lower())

    tokens = [
        token
        for word in TOKEN_PATTERN.findall(text)
        for token in split_cjk(word)
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS
    ]

    return [piece for token in tokens for piece in normalize_token(token)]


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
    Scores: NaN = video has no text for this source. Query terms are
    counted once (binary query), so a word repeated in the caption is
    not over-weighted.
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
    Fusion rule [2]: per-query z-score over candidates that have the
    source; candidates without it (NaN) get 0 (the pool mean).
    """
    if not (~np.isnan(raw)).any():
        return np.zeros_like(raw)

    mean = np.nanmean(raw, axis=1, keepdims=True)
    std = np.nanstd(raw, axis=1, keepdims=True) + 1e-6

    return np.nan_to_num((raw - mean) / std, nan=0.0).astype(np.float32)


def confidence_bonus(raw, rare_match, min_score):
    """
    Fusion rules [3] + [3b]: only candidates whose BM25 score exceeds
    min_score AND that share a rare caption word get a bonus; videos
    without the source (NaN) get 0.
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
    bonus_weights = dict(zip(BONUS_SOURCES, args.bonus_weights))

    print("=" * 70)
    print("MULTIVENT FULL EVALUATION (VLM + OCR + AUDIO)")
    print("=" * 70)
    print(
        f"Query field: {args.query_field}   "
        f"Bonus weights: {bonus_weights}   "
        f"min BM25: ocr {args.ocr_min_score}, audio {args.audio_min_score}   "
        f"rare ratio: {args.rare_ratio}"
    )

    # -----------------------------------------------------------------
    # Queries + candidate pool
    # -----------------------------------------------------------------
    records = load_queries(args.json, args.query_field)

    print(f"Records in JSON: {len(records)}")

    # multivent_base.json lists more videos than were downloaded / indexed.
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

    ocr_texts = load_source_texts(args.ocr, "ocr", candidate_ids, reverse_arabic=True)
    audio_texts = load_source_texts(args.audio, "audio", candidate_ids)

    # -----------------------------------------------------------------
    # Raw scores per source (fusion rule [1])
    # -----------------------------------------------------------------
    model = load_model(embed_dim)

    start_time = time.time()

    print(f"\nEncoding queries ({args.query_field})...")
    query_video_emb = encode_texts(
        model, queries, VIDEO_INSTRUCTION, args.batch_size, "Query (video)"
    )

    print("\nScoring sources...")
    rare_terms = rare_caption_terms(queries, args.rare_ratio)
    print(f"  rare query words: {len(rare_terms)}")

    raw = {"vlm": query_video_emb @ video_matrix.T}
    rare_match = {}

    for source, texts in (("ocr", ocr_texts), ("audio", audio_texts)):
        raw[source], rare_match[source] = bm25_source_scores(
            queries, texts, rare_terms, source
        )

    scoring_time = time.time() - start_time

    # -----------------------------------------------------------------
    # z-score + confidence bonus (fusion rules [2]-[4])
    # -----------------------------------------------------------------
    z_vlm = zscore_per_query(raw["vlm"])
    min_score = {"ocr": args.ocr_min_score, "audio": args.audio_min_score}
    bonus = {
        source: confidence_bonus(raw[source], rare_match[source], min_score[source])
        for source in BONUS_SOURCES
    }

    fused = z_vlm + sum(
        bonus_weights[source] * bonus[source] for source in BONUS_SOURCES
    )

    # -----------------------------------------------------------------
    # Ranks + metrics (fusion rule [5])
    # A source alone cannot retrieve a video it has no text for: -inf.
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

    # How often the bonus fires and whether it helps the correct video.
    print("\nBonus activity:")

    for source in BONUS_SOURCES:
        active = bonus[source] > 0
        target_active = int(np.diag(active).sum())
        print(
            f"  {source:<6}: {active.any(axis=1).sum():4d} queries with any bonus | "
            f"correct video boosted in {target_active} queries"
        )

    helped = int(np.sum(ranks["fusion"] < ranks["vlm"]))
    hurt = int(np.sum(ranks["fusion"] > ranks["vlm"]))
    print(f"\nFusion vs VLM: better in {helped} queries, worse in {hurt} queries")

    # Per-language breakdown (the query is ranked against the whole pool).
    languages = np.array([str(r.get("language") or "unknown") for r in records])
    metrics_by_language = {}

    print(f"\nPer language{'':<6}{'N':>6}{'R@1 vlm':>10}{'R@1 fus':>10}"
          f"{'R@10 vlm':>10}{'R@10 fus':>10}{'MRR fus':>9}")

    for language in sorted(set(languages)):
        mask = languages == language
        metrics_by_language[language] = {
            name: calculate_metrics(r[mask]) for name, r in ranks.items()
        }
        vlm_m = metrics_by_language[language]["vlm"]
        fus_m = metrics_by_language[language]["fusion"]
        print(
            f"  {language:<16}{int(mask.sum()):>6}"
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
                "language": record.get("language"),
                "event_category": record.get("event_category"),
                "event_name": record.get("event_name"),
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
                        "bm25_audio": round(float(np.nan_to_num(raw["audio"][q, c])), 4),
                        "bonus_ocr": round(float(bonus["ocr"][q, c]), 4),
                        "bonus_audio": round(float(bonus["audio"][q, c]), 4),
                        "ocr_matched_terms": matched_terms(
                            record["query"], ocr_texts[c]
                        ),
                        "audio_matched_terms": matched_terms(
                            record["query"], audio_texts[c]
                        ),
                    }
                    for i, c in enumerate(top)
                ],
            }
        )

    output_file = args.output or (
        Path(args.json).parent / "evaluation_multivent_full_results.json"
    )

    output = {
        "evaluation": {
            "dataset": "MultiVENT",
            "candidate_source": str(args.json),
            "query_field": args.query_field,
            "candidate_videos": len(candidate_ids),
            "queries": len(queries),
            "model": MODEL_NAME,
            "embedding_dimension": embed_dim,
            "video_instruction": VIDEO_INSTRUCTION,
            "bonus_weights": bonus_weights,
            "ocr_min_score": args.ocr_min_score,
            "audio_min_score": args.audio_min_score,
            "rare_ratio": args.rare_ratio,
            "rare_caption_terms": len(rare_terms),
            "text_matching": f"BM25 (k1={BM25_K1}, b={BM25_B}) for OCR and audio, "
                             f"stopwords removed, URLs removed, CJK split into "
                             f"bigrams, tokens >= {MIN_TOKEN_LENGTH} chars",
            "fusion": "final = z_vlm + b_ocr * max(0, bm25_ocr - ocr_min_score) "
                      "+ b_audio * max(0, bm25_audio - audio_min_score); "
                      "bonus kept only if a shared word is rare among captions; "
                      "missing source = no bonus; "
                      "rank = #candidates with score >= target",
            "files": {
                "index": str(args.index),
                "metadata": str(args.metadata),
                "ocr": str(args.ocr),
                "audio": str(args.audio),
            },
        },
        "metrics": metrics,
        "metrics_by_language": metrics_by_language,
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
    # Descriptions / OCR text may not fit the Windows console codepage.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Evaluate VLM + OCR/audio confidence-bonus fusion on MultiVENT."
    )
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument(
        "--query-field",
        default=QUERY_FIELD,
        help="JSON field used as the query text (default: video_description).",
    )
    parser.add_argument("--index", type=Path, default=INDEX_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--ocr", type=Path, default=OCR_FILE)
    parser.add_argument("--audio", type=Path, default=AUDIO_FILE)
    parser.add_argument(
        "--bonus-weights",
        type=float,
        nargs=2,
        default=list(DEFAULT_BONUS_WEIGHTS),
        metavar=("OCR", "AUDIO"),
        help="Weight of the OCR / audio confidence bonus (default: 0.05 0.05).",
    )
    parser.add_argument(
        "--ocr-min-score",
        type=float,
        default=DEFAULT_OCR_MIN_SCORE,
        help="BM25 score above which OCR adds a bonus (default: 15).",
    )
    parser.add_argument(
        "--audio-min-score",
        type=float,
        default=DEFAULT_AUDIO_MIN_SCORE,
        help="BM25 score above which audio adds a bonus (default: 15).",
    )
    parser.add_argument(
        "--rare-ratio",
        type=float,
        default=DEFAULT_RARE_RATIO,
        help="A caption word is rare if it appears in at most this fraction "
             "of captions (default: 0.01). A bonus needs a shared rare word.",
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

    for name in ("json", "index", "metadata", "ocr", "audio"):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"--{name} not found: {path}")

    evaluate(args)


if __name__ == "__main__":
    main()
