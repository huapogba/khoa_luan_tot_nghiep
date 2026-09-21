"""
MSR-VTT 1K JSON evaluation for Qwen3-VL-Embedding-2B + FAISS.

Input JSON format:
[
  {
    "video_id": "video7020",
    "video": "video7020.mp4",
    "caption": "...",
    "category": 10,
    ...
  },
  ...
]

Evaluation:
    caption/text query -> retrieve the correct video
    Candidate pool = exactly the videos listed in the JSON file.

Metrics:
    R@1, R@5, R@10
    MRR
    MdR (median rank)
    Mean Rank

Important:
- The candidate pool is NOT the full 10,000-video index.
- It is restricted to the video IDs in msrvtt_test_1k.json.
- The search is performed over ALL candidates, so the exact rank is known.
- The embedding model/instruction/normalization match the visual index created
  by your video_processing.py:
      Qwen/Qwen3-VL-Embedding-2B
      truncate_dim=768
      instruction = "Represent the video for retrieval."
      L2 normalization

Example:
    python evaluate_msrvtt_1k_json.py

Optional:
    python evaluate_msrvtt_1k_json.py --json data/msrvtt_test_1k.json
    python evaluate_msrvtt_1k_json.py --batch-size 4
    python evaluate_msrvtt_1k_json.py --limit 10
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Defaults - adjust these only if your project uses different paths.
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

JSON_FILE = PROJECT_ROOT / "data" / "msrvtt_test_1k.json"
INDEX_FILE = PROJECT_ROOT / "data" / "bin" / "index.faiss"
METADATA_FILE = PROJECT_ROOT / "data" / "bin" / "metadata.json"

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
EMBED_DIM = 768
INSTRUCTION = "Represent the video for retrieval."

DEFAULT_BATCH_SIZE = 8


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of records.")

    return data


def validate_records(records):
    required = {"video_id", "caption"}

    for i, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Record {i} is not a JSON object.")

        missing = required - set(item.keys())
        if missing:
            raise ValueError(
                f"Record {i} is missing required fields: {sorted(missing)}"
            )

        if not item["video_id"]:
            raise ValueError(f"Record {i} has an empty video_id.")

        if not item["caption"]:
            raise ValueError(f"Record {i} has an empty caption.")


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError("metadata.json must contain a list.")

    return metadata


def build_metadata_lookup(metadata):
    lookup = {}

    for i, item in enumerate(metadata):
        video_id = item.get("video_id")
        if video_id is None:
            continue

        lookup[video_id] = {
            "index": i,
            "metadata": item,
        }

    return lookup


def load_model():
    print(f"Loading model: {MODEL_NAME}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        truncate_dim=EMBED_DIM,
    )

    return model


def encode_queries(model, queries, batch_size):
    """
    Encode text queries using the same embedding model and instruction
    used for the video index.
    """

    all_embeddings = []

    total = len(queries)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = queries[start:end]

        embeddings = model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt=INSTRUCTION,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        all_embeddings.append(embeddings)

        print(
            f"\rEncoding queries: {end}/{total}",
            end="",
            flush=True,
        )

    print()

    embeddings = np.vstack(all_embeddings)

    return embeddings


def create_candidate_index(full_index, metadata, candidate_ids):
    """
    Build an exact FAISS IndexFlatIP containing only the 1K candidate videos.

    The original index is expected to contain vectors in the same order as
    metadata.json.
    """

    metadata_lookup = build_metadata_lookup(metadata)

    candidate_vectors = []
    candidate_metadata_indices = []
    missing = []

    for video_id in candidate_ids:
        if video_id not in metadata_lookup:
            missing.append(video_id)
            continue

        meta_index = metadata_lookup[video_id]["index"]

        vector = full_index.reconstruct(meta_index)

        candidate_vectors.append(vector)
        candidate_metadata_indices.append(meta_index)

    if missing:
        print(
            f"WARNING: {len(missing)} JSON videos were not found "
            f"in metadata.json."
        )

        print("First missing IDs:")
        for video_id in missing[:20]:
            print(f"  {video_id}")

    if not candidate_vectors:
        raise RuntimeError(
            "None of the JSON video IDs were found in metadata.json."
        )

    candidate_vectors = np.asarray(candidate_vectors, dtype=np.float32)

    # Vectors are already normalized, so inner product = cosine similarity.
    candidate_index = faiss.IndexFlatIP(candidate_vectors.shape[1])
    candidate_index.add(candidate_vectors)

    return (
        candidate_index,
        candidate_metadata_indices,
        missing,
    )


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


def evaluate(
    json_file,
    index_file,
    metadata_file,
    batch_size,
    limit=None,
    output_file=None,
):
    print("=" * 70)
    print("MSR-VTT 1K JSON RETRIEVAL EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load evaluation JSON
    # ---------------------------------------------------------------
    print(f"\nJSON:      {json_file}")
    records = load_json(json_file)
    validate_records(records)

    print(f"Records:   {len(records)}")

    # Ensure each video occurs once.
    video_ids = [x["video_id"] for x in records]

    duplicates = [
        video_id
        for video_id in set(video_ids)
        if video_ids.count(video_id) > 1
    ]

    if duplicates:
        print(
            f"WARNING: JSON contains {len(duplicates)} duplicate video IDs."
        )
        print("The evaluator will keep the first occurrence.")

        seen = set()
        unique_records = []

        for item in records:
            if item["video_id"] not in seen:
                seen.add(item["video_id"])
                unique_records.append(item)

        records = unique_records

    if limit is not None:
        records = records[:limit]
        print(f"Limit:     {len(records)}")

    candidate_ids = [x["video_id"] for x in records]
    queries = [x["caption"] for x in records]

    # ---------------------------------------------------------------
    # Load full video index
    # ---------------------------------------------------------------
    print(f"\nFAISS:     {index_file}")

    full_index = faiss.read_index(str(index_file))

    print(f"Full vectors: {full_index.ntotal}")
    print(f"Dimension:    {full_index.d}")

    if full_index.d != EMBED_DIM:
        raise ValueError(
            f"Expected {EMBED_DIM}-D embeddings, "
            f"but FAISS index has {full_index.d}-D."
        )

    # ---------------------------------------------------------------
    # Load metadata
    # ---------------------------------------------------------------
    print(f"Metadata:  {metadata_file}")

    metadata = load_metadata(metadata_file)

    print(f"Metadata entries: {len(metadata)}")

    if len(metadata) != full_index.ntotal:
        raise ValueError(
            "FAISS index and metadata.json have different numbers of entries: "
            f"{full_index.ntotal} vs {len(metadata)}"
        )

    # ---------------------------------------------------------------
    # Create 1K candidate-only index
    # ---------------------------------------------------------------
    print("\nBuilding candidate-only FAISS index...")

    candidate_index, candidate_metadata_indices, missing = (
        create_candidate_index(
            full_index,
            metadata,
            candidate_ids,
        )
    )

    print(f"Candidate videos: {candidate_index.ntotal}")

    if missing:
        print(
            f"WARNING: {len(missing)} candidates were missing. "
            "Metrics are computed only over matched candidates."
        )

        # Remove missing records from queries so the denominator is correct.
        missing_set = set(missing)

        filtered_records = [
            x for x in records
            if x["video_id"] not in missing_set
        ]

        records = filtered_records
        candidate_ids = [x["video_id"] for x in records]
        queries = [x["caption"] for x in records]

        candidate_index, candidate_metadata_indices, _ = (
            create_candidate_index(
                full_index,
                metadata,
                candidate_ids,
            )
        )

        print(f"Final candidate videos: {candidate_index.ntotal}")

    # ---------------------------------------------------------------
    # Map candidate FAISS position -> video_id
    # ---------------------------------------------------------------
    candidate_faiss_to_video = {
        i: video_id
        for i, video_id in enumerate(candidate_ids)
    }

    candidate_video_to_faiss = {
        video_id: i
        for i, video_id in enumerate(candidate_ids)
    }

    # ---------------------------------------------------------------
    # Load Qwen embedding model
    # ---------------------------------------------------------------
    model = load_model()

    # ---------------------------------------------------------------
    # Encode text queries
    # ---------------------------------------------------------------
    print("\nEncoding captions...")

    start_time = time.time()

    query_embeddings = encode_queries(
        model,
        queries,
        batch_size,
    )

    encode_time = time.time() - start_time

    print(f"Encoding time: {encode_time:.2f} sec")

    # ---------------------------------------------------------------
    # Search entire 1K candidate pool
    # ---------------------------------------------------------------
    print("\nSearching entire candidate pool...")

    search_start = time.time()

    # IMPORTANT:
    # Search all candidates, not only top-10.
    # This is necessary for exact MRR/MdR/MeanRank.
    scores, indices = candidate_index.search(
        query_embeddings,
        candidate_index.ntotal,
    )

    search_time = time.time() - search_start

    print(f"Search time: {search_time:.2f} sec")

    # ---------------------------------------------------------------
    # Calculate exact ranks
    # ---------------------------------------------------------------
    ranks = []
    results = []

    for query_idx, record in enumerate(records):
        query_video_id = record["video_id"]

        target_position = candidate_video_to_faiss[query_video_id]

        row = indices[query_idx]

        matches = np.where(row == target_position)[0]

        if len(matches) == 0:
            raise RuntimeError(
                f"Could not find target video {query_video_id} "
                f"in its own candidate ranking."
            )

        rank = int(matches[0]) + 1

        ranks.append(rank)

        top_n = min(10, len(row))

        top_results = []

        for rank_idx in range(top_n):
            candidate_position = int(row[rank_idx])
            video_id = candidate_faiss_to_video[candidate_position]

            top_results.append(
                {
                    "rank": rank_idx + 1,
                    "video_id": video_id,
                    "score": float(scores[query_idx][rank_idx]),
                }
            )

        results.append(
            {
                "query_index": query_idx,
                "query_video_id": query_video_id,
                "caption": record["caption"],
                "category": record.get("category"),
                "rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "top_10": top_results,
            }
        )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------
    metrics = calculate_metrics(ranks)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"Videos      : {candidate_index.ntotal}")
    print(f"Queries     : {metrics['queries']}")
    print(f"R@1         : {metrics['R@1']:.4f} ({metrics['R@1'] * 100:.2f}%)")
    print(f"R@5         : {metrics['R@5']:.4f} ({metrics['R@5'] * 100:.2f}%)")
    print(f"R@10        : {metrics['R@10']:.4f} ({metrics['R@10'] * 100:.2f}%)")
    print(f"MRR         : {metrics['MRR']:.4f}")
    print(f"MdR         : {metrics['MdR']:.1f}")
    print(f"Mean Rank   : {metrics['MeanRank']:.2f}")

    # ---------------------------------------------------------------
    # Rank distribution
    # ---------------------------------------------------------------
    print("\nRank distribution:")

    for threshold in [1, 5, 10, 50, 100, 500, 1000]:
        count = int(np.sum(np.asarray(ranks) <= threshold))
        percentage = 100.0 * count / len(ranks)

        print(
            f"  Rank <= {threshold:4d}: "
            f"{count:4d}/{len(ranks)} ({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------------
    # Save detailed results
    # ---------------------------------------------------------------
    if output_file is None:
        output_file = (
            Path(json_file).parent
            / "evaluation_msrvtt_1k_results.json"
        )

    output = {
        "evaluation": {
            "dataset": "MSR-VTT",
            "candidate_source": str(json_file),
            "candidate_videos": candidate_index.ntotal,
            "queries": len(records),
            "model": MODEL_NAME,
            "embedding_dimension": EMBED_DIM,
            "instruction": INSTRUCTION,
            "normalization": "L2 / cosine similarity via IndexFlatIP",
            "search_pool": "all candidate videos in JSON",
        },
        "metrics": metrics,
        "timing": {
            "query_encoding_seconds": encode_time,
            "search_seconds": search_time,
        },
        "results": results,
    }

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to:")
    print(output_file)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen video retrieval on MSR-VTT 1K JSON."
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=JSON_FILE,
        help="Path to msrvtt_test_1k.json",
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=INDEX_FILE,
        help="Path to FAISS video index",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_FILE,
        help="Path to metadata.json",
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

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path.",
    )

    args = parser.parse_args()

    if not args.json.exists():
        raise FileNotFoundError(f"JSON not found: {args.json}")

    if not args.index.exists():
        raise FileNotFoundError(f"FAISS index not found: {args.index}")

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata not found: {args.metadata}")

    evaluate(
        json_file=args.json,
        index_file=args.index,
        metadata_file=args.metadata,
        batch_size=args.batch_size,
        limit=args.limit,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
