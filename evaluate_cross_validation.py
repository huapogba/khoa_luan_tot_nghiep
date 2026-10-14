"""
MultiVENT cross-validation: VLM + OCR + audio (BM25) fusion with the fusion
parameters chosen on a train split and reported on a held-out test split.

Same data, BM25 scoring and fusion rules as evaluate_msrvtt_1K_full.py
(helpers imported from it). That script uses fixed default parameters;
here they are selected on 70% of the queries and measured on the other 30%,
so the reported numbers were never used to pick the parameters.

========================================================================
PROTOCOL
========================================================================
  [1] Split by event (70% train / 30% test of the queries)
      Videos of the same event have near-identical descriptions (same
      places, names, dates). A random split would put them on both sides,
      so the train side would indirectly "see" the test side. Whole events
      are therefore assigned to train or test (group split on event_name).

  [2] Candidate pool unchanged
      Every query (train or test) is still ranked against all videos.
      Only the queries are split: parameters are chosen from the ranks of
      the correct videos of train queries, and never look at test answers.
      BM25 IDF and the rare-word gate use texts only (no answers), so they
      are computed once over the whole collection.

  [3] Grid search on train (selection metric: MRR, see --select-metric)
          min BM25 score  t   (--grid-min-score)
          bonus weight    w   (--grid-weight)
          margin rule     m   (--grid-margin)
      Fusion (as in evaluate_msrvtt_1K_full.py, t and w shared by OCR and
      audio to keep the grid small):
          bonus_src = max(0, bm25_src - t)      if a rare query word is shared
          final     = z_vlm + w * bonus_ocr + w * bonus_audio
      Margin rule (m = "off" disables it):
          the bonus is kept only for the video with the highest BM25 score
          of the query, and only if it beats the second one by >= m.
          Targets queries where several videos pass the threshold and a
          wrong one gets lifted above a correct VLM answer.

  [4] Test once with the chosen parameters.

  [5] Repeat [1]-[4] with different seeds (--repeats) and report
      mean +- std: with ~420 test queries one split varies by +-2-3 R@1.
========================================================================

Example:
    python evaluate_cross_validation.py
    python evaluate_cross_validation.py --repeats 20 --test-ratio 0.3
    python evaluate_cross_validation.py --select-metric R@1
"""

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

import evaluate_msrvtt_1K_full as base


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULT_TEST_RATIO = 0.3
DEFAULT_REPEATS = 10
DEFAULT_SEED = 0
DEFAULT_GRID_MIN_SCORE = (5, 10, 15, 20, 30)
DEFAULT_GRID_WEIGHT = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5)
DEFAULT_GRID_MARGIN = ("off", "0", "2", "5", "10")
SELECT_METRICS = ("MRR", "R@1", "R@5", "R@10")

# Fixed parameters of evaluate_msrvtt_1K_full.py, reported for comparison.
FIXED_PARAMS = (base.DEFAULT_OCR_MIN_SCORE, base.DEFAULT_BONUS_WEIGHTS[0], None)

CACHE_DIR = base.PROJECT_ROOT / "data" / "cache_cv"


# ---------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------
def vlm_query_embeddings(args, queries, embed_dim):
    """
    Qwen query vectors, cached: the grid search itself needs no GPU.
    """
    key = hashlib.sha1(
        json.dumps([queries, embed_dim, base.MODEL_NAME, base.VIDEO_INSTRUCTION],
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    path = Path(args.cache_dir) / f"vlm_queries_{key}.npy"

    if path.exists() and not args.no_cache:
        print(f"Loading cached query embeddings: {path}")
        return np.load(path)

    model = base.load_model(embed_dim)
    print(f"\nEncoding queries ({args.query_field})...")
    embeddings = base.encode_texts(
        model, queries, base.VIDEO_INSTRUCTION, args.batch_size, "Query (video)"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    print(f"Query embeddings cached to: {path}")

    return embeddings


def top_bm25_mask(raw, margin):
    """
    Margin rule: True only for the top BM25 video of each query, and only
    if it beats the second one by >= margin.
    """
    scores = np.nan_to_num(raw, nan=0.0)
    ordered = np.sort(scores, axis=1)
    first, second = ordered[:, -1:], ordered[:, -2:-1]

    return (scores >= first) & (first > 0) & ((first - second) >= margin)


def fused_scores(z_vlm, raw, rare_match, params):
    min_score, weight, margin = params
    fused = z_vlm.copy()

    for source in base.BONUS_SOURCES:
        bonus = base.confidence_bonus(raw[source], rare_match[source], min_score)

        if margin is not None:
            bonus = np.where(top_bm25_mask(raw[source], margin), bonus, 0.0)

        fused += weight * bonus

    return fused


def params_label(params):
    min_score, weight, margin = params
    margin = "off" if margin is None else f"{margin:g}"
    return f"t={min_score:g} w={weight:g} m={margin}"


# ---------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------
def event_split(events, test_ratio, seed):
    """
    Group split: whole events go to test (shuffled) until the test side
    holds test_ratio of the queries. Returns a boolean test mask.
    """
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(events)))
    rng.shuffle(unique)

    counts = {event: int(np.sum(events == event)) for event in unique}
    target = test_ratio * len(events)

    test_events = set()
    total = 0

    for event in unique:
        if total >= target:
            break
        test_events.add(event)
        total += counts[event]

    return np.array([event in test_events for event in events])


def metrics_of(ranks):
    return base.calculate_metrics(ranks) if len(ranks) else None


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std())}


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate(args):
    margins = [None if m == "off" else float(m) for m in args.grid_margin]
    grid = list(itertools.product(args.grid_min_score, args.grid_weight, margins))

    print("=" * 70)
    print("MULTIVENT CROSS-VALIDATION (VLM + OCR + AUDIO, BM25)")
    print("=" * 70)
    print(
        f"Split: {100 * (1 - args.test_ratio):.0f}% train / "
        f"{100 * args.test_ratio:.0f}% test by event   repeats: {args.repeats}   "
        f"select: {args.select_metric}   grid: {len(grid)} configs"
    )

    # -----------------------------------------------------------------
    # Data (same as evaluate_msrvtt_1K_full.py)
    # -----------------------------------------------------------------
    records = base.load_queries(args.json, args.query_field)

    vectors, missing, embed_dim = base.load_video_vectors(
        args.index, args.metadata, [r["video_id"] for r in records]
    )

    if missing:
        missing_set = set(missing)
        records = [r for r in records if r["video_id"] not in missing_set]

    if not records:
        raise RuntimeError("No JSON video was found in the FAISS index.")

    candidate_ids = [r["video_id"] for r in records]
    queries = [r["query"] for r in records]
    events = np.array([str(r.get("event_name") or r["video_id"]) for r in records])
    languages = np.array([str(r.get("language") or "unknown") for r in records])

    print(f"Queries / candidates: {len(records)}   Events: {len(set(events))}")

    video_matrix = np.stack([vectors[v] for v in candidate_ids]).astype(np.float32)

    ocr_texts = base.load_source_texts(
        args.ocr, "ocr", candidate_ids, reverse_arabic=True
    )
    audio_texts = base.load_source_texts(args.audio, "audio", candidate_ids)

    # -----------------------------------------------------------------
    # Scores, computed once (no answers involved)
    # -----------------------------------------------------------------
    start_time = time.time()

    query_emb = vlm_query_embeddings(args, queries, embed_dim)
    z_vlm = base.zscore_per_query(query_emb @ video_matrix.T)

    print("\nScoring sources...")
    rare_terms = base.rare_caption_terms(queries, args.rare_ratio)
    raw, rare_match = {}, {}

    for source, texts in (("ocr", ocr_texts), ("audio", audio_texts)):
        raw[source], rare_match[source] = base.bm25_source_scores(
            queries, texts, rare_terms, source
        )

    # Ranks of every config for every query; a split only selects rows.
    print(f"Ranking {len(grid)} fusion configs...")
    ranks_of = {
        params: base.compute_ranks(fused_scores(z_vlm, raw, rare_match, params))
        for params in grid
    }

    fixed = FIXED_PARAMS
    if fixed not in ranks_of:
        ranks_of[fixed] = base.compute_ranks(fused_scores(z_vlm, raw, rare_match, fixed))

    vlm_ranks = base.compute_ranks(z_vlm)

    scoring_time = time.time() - start_time

    # -----------------------------------------------------------------
    # Repeated 70/30 splits
    # -----------------------------------------------------------------
    def score(ranks):
        return base.calculate_metrics(ranks)[args.select_metric]

    splits = []

    print(f"\n{'Split':<7}{'train':>7}{'test':>6}  {'chosen params':<22}"
          f"{'train ' + args.select_metric:>11}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}"
          f"{'VLM R@1':>9}")

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        test = event_split(events, args.test_ratio, seed)
        train = ~test

        best = max(grid, key=lambda params: score(ranks_of[params][train]))

        test_metrics = {
            "fusion_cv": metrics_of(ranks_of[best][test]),
            "fusion_fixed": metrics_of(ranks_of[fixed][test]),
            "vlm": metrics_of(vlm_ranks[test]),
        }

        by_language = {
            language: {
                "queries": int(np.sum(test & (languages == language))),
                "fusion_cv": metrics_of(ranks_of[best][test & (languages == language)]),
                "vlm": metrics_of(vlm_ranks[test & (languages == language)]),
            }
            for language in sorted(set(languages[test]))
        }

        splits.append(
            {
                "seed": seed,
                "train_queries": int(train.sum()),
                "test_queries": int(test.sum()),
                "test_events": sorted(set(events[test])),
                "chosen_params": {
                    "min_score": best[0],
                    "weight": best[1],
                    "margin": best[2],
                },
                "train_score": score(ranks_of[best][train]),
                "test": test_metrics,
                "test_by_language": by_language,
                "fusion_vs_vlm": {
                    "better": int(np.sum(ranks_of[best][test] < vlm_ranks[test])),
                    "worse": int(np.sum(ranks_of[best][test] > vlm_ranks[test])),
                },
            }
        )

        m = test_metrics["fusion_cv"]
        print(
            f"{seed:<7}{int(train.sum()):>7}{int(test.sum()):>6}  "
            f"{params_label(best):<22}{score(ranks_of[best][train]):>11.4f}"
            f"{m['R@1'] * 100:>7.2f}%{m['R@5'] * 100:>7.2f}%{m['R@10'] * 100:>7.2f}%"
            f"{m['MRR']:>8.4f}{test_metrics['vlm']['R@1'] * 100:>8.2f}%"
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    summary = {
        name: {
            metric: mean_std([s["test"][name][metric] for s in splits])
            for metric in ("R@1", "R@5", "R@10", "MRR", "MdR")
        }
        for name in ("vlm", "fusion_fixed", "fusion_cv")
    }

    print("\n" + "=" * 70)
    print(f"TEST RESULTS (mean +- std over {args.repeats} splits)")
    print("=" * 70)
    print(f"{'Method':<30}{'R@1':>14}{'R@5':>14}{'R@10':>14}{'MRR':>16}")

    labels = {
        "vlm": "VLM only",
        "fusion_fixed": f"fusion fixed ({params_label(fixed)})",
        "fusion_cv": "fusion CV-selected",
    }

    for name, label in labels.items():
        s = summary[name]
        print(
            f"{label:<30}"
            + "".join(
                f"{s[k]['mean'] * 100:>8.2f}+-{s[k]['std'] * 100:<4.2f}"
                for k in ("R@1", "R@5", "R@10")
            )
            + f"{s['MRR']['mean']:>9.4f}+-{s['MRR']['std']:.4f}"
        )

    print("\nPer language, test R@1 (mean over splits containing the language):")
    print(f"  {'language':<12}{'avg N':>7}{'VLM':>9}{'fusion CV':>11}")

    summary_by_language = {}

    for language in sorted(set(languages)):
        rows = [s["test_by_language"][language] for s in splits
                if language in s["test_by_language"]]
        vlm_r1 = np.mean([r["vlm"]["R@1"] for r in rows])
        cv_r1 = np.mean([r["fusion_cv"]["R@1"] for r in rows])
        avg_n = np.mean([r["queries"] for r in rows])
        summary_by_language[language] = {
            "splits": len(rows),
            "avg_queries": float(avg_n),
            "vlm_R@1": float(vlm_r1),
            "fusion_cv_R@1": float(cv_r1),
        }
        print(f"  {language:<12}{avg_n:>7.0f}{vlm_r1 * 100:>8.2f}%{cv_r1 * 100:>10.2f}%")

    chosen = {}
    for s in splits:
        label = params_label(
            (s["chosen_params"]["min_score"], s["chosen_params"]["weight"],
             s["chosen_params"]["margin"])
        )
        chosen[label] = chosen.get(label, 0) + 1

    print("\nChosen params (how many splits):")
    for label, count in sorted(chosen.items(), key=lambda item: -item[1]):
        print(f"  {label:<24}{count}")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    output_file = Path(
        args.output
        or Path(args.json).parent / "evaluation_multivent_cross_validation.json"
    )

    output = {
        "evaluation": {
            "dataset": "MultiVENT",
            "candidate_source": str(args.json),
            "query_field": args.query_field,
            "queries": len(records),
            "events": len(set(events)),
            "split": "group split on event_name",
            "test_ratio": args.test_ratio,
            "repeats": args.repeats,
            "seed": args.seed,
            "select_metric": args.select_metric,
            "grid": {
                "min_score": list(args.grid_min_score),
                "weight": list(args.grid_weight),
                "margin": list(args.grid_margin),
            },
            "fixed_params": params_label(fixed),
            "rare_ratio": args.rare_ratio,
            "model": base.MODEL_NAME,
            "embedding_dimension": embed_dim,
            "fusion": "final = z_vlm + w * sum_src max(0, bm25_src - t) "
                      "[shared rare query word; margin rule: only the top BM25 "
                      "video, if it beats the second by >= m]",
        },
        "summary": summary,
        "summary_by_language": summary_by_language,
        "chosen_params_count": chosen,
        "timing": {"scoring_seconds": scoring_time},
        "splits": splits,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to:\n{output_file}")

    return summary


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="70/30 event-split cross-validation of VLM + BM25 OCR/audio "
                    "fusion on MultiVENT."
    )
    parser.add_argument("--json", type=Path, default=base.JSON_FILE)
    parser.add_argument("--query-field", default=base.QUERY_FIELD)
    parser.add_argument("--index", type=Path, default=base.INDEX_FILE)
    parser.add_argument("--metadata", type=Path, default=base.METADATA_FILE)
    parser.add_argument("--ocr", type=Path, default=base.OCR_FILE)
    parser.add_argument("--audio", type=Path, default=base.AUDIO_FILE)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Fraction of queries in the test split (default: 0.3).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of random event splits (default: 10).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--select-metric",
        choices=SELECT_METRICS,
        default="MRR",
        help="Metric maximized on the train split (default: MRR).",
    )
    parser.add_argument(
        "--grid-min-score",
        type=float,
        nargs="+",
        default=list(DEFAULT_GRID_MIN_SCORE),
        help="BM25 thresholds to try.",
    )
    parser.add_argument(
        "--grid-weight",
        type=float,
        nargs="+",
        default=list(DEFAULT_GRID_WEIGHT),
        help="Bonus weights to try.",
    )
    parser.add_argument(
        "--grid-margin",
        nargs="+",
        default=list(DEFAULT_GRID_MARGIN),
        help='Margin rule values to try ("off" disables the rule).',
    )
    parser.add_argument("--rare-ratio", type=float, default=base.DEFAULT_RARE_RATIO)
    parser.add_argument("--batch-size", type=int, default=base.DEFAULT_BATCH_SIZE)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    for name in ("json", "index", "metadata", "ocr", "audio"):
        path = getattr(args, name)
        if not path.exists():
            raise FileNotFoundError(f"--{name} not found: {path}")

    if not 0 < args.test_ratio < 1:
        raise ValueError("--test-ratio must be between 0 and 1.")

    for margin in args.grid_margin:
        if margin != "off":
            float(margin)

    evaluate(args)


if __name__ == "__main__":
    main()
