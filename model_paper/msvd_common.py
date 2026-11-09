"""
Shared code for the MSVD baselines in this folder (CLIP, CLIP4Clip, X-CLIP).

Data (from MSVD_data/, produced by down_msvd.py):
    MSVD_data/msvd_test.json      670 test videos, ~40 captions each
    MSVD_data/Video/test/*.avi    the test clips

Outputs (inside msvd_evaluate/):
    features/<model>_video.npz    video embeddings  (*_processing.py)
    results/<model>_results.json  metrics            (*_evaluate.py)

Protocol = evaluation_msvd.py (Qwen3-VL), so numbers are comparable:
    t2v: every caption is a query against all candidate videos;
         rank = #videos scoring >= the correct one (ties count against it)
    v2t: every video is a query against all captions; rank of its
         best-scoring own caption
    R@1, R@5, R@10, MRR, MdR, MeanRank
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent

MSVD_DIR = PROJECT_ROOT / "MSVD_data"
JSON_FILE = MSVD_DIR / "msvd_test.json"
VIDEO_DIR = MSVD_DIR / "Video" / "test"

FEATURE_DIR = EVAL_DIR / "features"
RESULT_DIR = EVAL_DIR / "results"

# Videos the Qwen index holds (663 of 670); used by default so every model
# is ranked over exactly the same candidates as Qwen3-VL.
QWEN_METADATA = PROJECT_ROOT / "data" / "bin" / "metadata_msvd.json"
QWEN_RESULTS = PROJECT_ROOT / "evaluation_msvd_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def setup_stdout():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_records(json_file=JSON_FILE):
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


def video_path(video_id, video_dir=VIDEO_DIR):
    return Path(video_dir) / f"{video_id}.avi"


def read_frames(path, num_frames):
    """
    num_frames RGB frames, one at the centre of each of num_frames equal
    segments of the clip.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"Cannot read video: {path}")

    idx = ((np.arange(num_frames) + 0.5) * len(frames) / num_frames).astype(int)
    idx = np.clip(idx, 0, len(frames) - 1)

    return [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx]


def load_features(path):
    """
    {"video_ids": [...], <array name>: array aligned with video_ids, ...}
    """
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    out["video_ids"] = [str(v) for v in out["video_ids"]]
    return out


def select_candidates(records, feature_ids, all_videos=False):
    """
    Records that have a feature vector, restricted to the Qwen index's
    videos unless all_videos is set. Returns (records, row of each record
    in the feature arrays).
    """
    row_of = {v: i for i, v in enumerate(feature_ids)}
    keep = set(row_of)

    if not all_videos and QWEN_METADATA.exists():
        keep &= {str(m.get("video_id")) for m in load_json(QWEN_METADATA)}
        print(f"Candidates restricted to the Qwen index ({QWEN_METADATA.name}).")

    records = [r for r in records if r["video_id"] in keep]
    rows = np.array([row_of[r["video_id"]] for r in records], dtype=np.int64)

    return records, rows


def flatten_captions(records):
    """
    All captions, and owner[i] = index (in records) of caption i's video.
    """
    captions, owner = [], []

    for v, r in enumerate(records):
        captions.extend(r["captions"])
        owner.extend([v] * len(r["captions"]))

    return captions, np.asarray(owner, dtype=np.int64)


# ---------------------------------------------------------------------
# Torch helpers
# ---------------------------------------------------------------------
def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def pooled(output):
    """
    get_*_features returns a tensor or (transformers >= 5) a ModelOutput.
    """
    if isinstance(output, torch.Tensor):
        return output
    for name in ("text_embeds", "image_embeds"):
        if getattr(output, name, None) is not None:
            return getattr(output, name)
    return output.pooler_output


# ---------------------------------------------------------------------
# Ranks + metrics
# ---------------------------------------------------------------------
def t2v_ranks(sim, owner):
    """
    sim: captions x videos.
    """
    target = sim[np.arange(len(owner)), owner][:, None]
    return (sim >= target).sum(axis=1)


def v2t_ranks(sim, owner):
    ranks = np.empty(sim.shape[1], dtype=np.int64)

    for v in range(sim.shape[1]):
        column = sim[:, v]
        own = owner == v
        ranks[v] = 1 + int(np.sum(column[~own] >= column[own].max()))

    return ranks


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


def print_metrics(metrics):
    print(f"{'Task':<8}{'N':>8}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'MRR':>8}{'MdR':>7}{'MnR':>8}")

    for name, m in metrics.items():
        print(
            f"{name:<8}{m['queries']:>8}"
            f"{m['R@1'] * 100:>7.2f}%{m['R@5'] * 100:>7.2f}%{m['R@10'] * 100:>7.2f}%"
            f"{m['MRR']:>8.4f}{m['MdR']:>7.1f}{m['MeanRank']:>8.2f}"
        )


def score_and_report(sim, records, owner, info, output_file):
    """
    Metrics from a captions x videos similarity matrix; prints and saves.
    """
    metrics = {
        "t2v": calculate_metrics(t2v_ranks(sim, owner)),
        "v2t": calculate_metrics(v2t_ranks(sim, owner)),
    }

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print_metrics(metrics)

    output = {
        "evaluation": {
            "dataset": "MSVD (test, official split)",
            "protocol": "all captions",
            "candidate_videos": len(records),
            "queries": int(len(owner)),
            **info,
        },
        "metrics": metrics,
    }
    save_json(output, output_file)
    print(f"\nSaved: {output_file}")

    return metrics
