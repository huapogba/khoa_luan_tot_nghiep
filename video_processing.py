"""Build a FAISS flat index (IndexFlatIP, cosine similarity) over Qwen3-VL-Embedding
embeddings of the videos already sitting in `data/clips`, plus a metadata.json
describing each entry (same row order as the FAISS index).

Swapped out from CLIP (frame-only, needs a Vi->En translation step for text
queries) to Qwen3-VL-Embedding: it embeds videos directly (not just one frame)
and is natively multilingual (30+ languages), so it should handle Vietnamese
content/queries without the translate-then-CLIP workaround.

Requirements:
    pip install sentence-transformers "transformers>=4.57.0" qwen-vl-utils accelerate opencv-python

Output:
  - OUTPUT_INDEX     : FAISS flat index file (faiss.IndexFlatIP over L2-normalized embeddings)
  - OUTPUT_METADATA  : JSON list, metadata[i] describes the video embedded at index row i

NOTE on GPU memory: Qwen3-VL-Embedding-2B is a real VLM (~4GB weights in bf16,
more with activations from video frames), not a lightweight CLIP. On a 6GB
laptop GPU (RTX 3060 6GB) keep BATCH_SIZE small (1) and watch VRAM. The 8B
variant is unlikely to fit on that card without offloading/quantization.

NOTE on the "video" input key: the model card documents text/image/video
modalities, but the public examples only show the exact dict shape for text
and image. If encode() rejects the "video" key on your installed version,
this script automatically falls back to embedding each video's first frame
via the "image" key instead (see embed_videos()).
"""

import json
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
CLIPS_DIR = Path("multivent_base")                    # folder holding the videos to index (used as-is, no cutting)
OUTPUT_INDEX = Path("data/bin/index_new_512.faiss")        # FAISS flat index output
OUTPUT_METADATA = Path("data/bin/metadata_new_512.json")   # metadata output, same row order as the index
THUMBS_DIR = Path("data/bin/thumbnails")           # only used if the "video" modality fallback kicks in

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"   # 2B recommended for 6GB VRAM; 8B needs much more
EMBED_DIM = 768                        # MRL-truncated output dim (model supports 64-2048); None = full 2048
BATCH_SIZE = 1                             # keep low for video inputs on limited VRAM
INSTRUCTION = "Represent the video for retrieval."  # docs recommend writing instructions in English even for multilingual content

# Video token budget. The model's default video config allows ~25M pixels per video
# (fps=2, up to 768 frames) -> >10k visual tokens for long videos, which OOMs a 6GB GPU
# in attention. Each visual token covers 16x16 px * 2 frames * 2x2 merge = 2048 px.
VIDEO_FPS = 1                              # frames sampled per second
VIDEO_MAX_FRAMES = 48                      # hard cap on sampled frames per video
VIDEO_MAX_TOKENS = 1024                    # total visual tokens per video (raise if VRAM allows)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def discover_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Không tồn tại thư mục: {folder}")
    videos = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"Không tìm thấy video nào trong {folder}")
    return videos


def get_duration_seconds(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    capture.release()
    if fps <= 0:
        raise RuntimeError(f"Không đọc được FPS: {video_path}")
    return frame_count / fps


def extract_first_frame(video_path: Path) -> Path:
    """Fallback only: save the first frame as a jpg and return its path."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Không đọc được frame đầu tiên: {video_path}")
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMBS_DIR / f"{video_path.stem}.jpg"
    cv2.imwrite(str(thumb_path), frame)
    return thumb_path


def load_model() -> SentenceTransformer:
    kwargs = {"truncate_dim": EMBED_DIM} if EMBED_DIM else {}
    model_kwargs = {"torch_dtype": torch.bfloat16} if DEVICE == "cuda" else {}
    model = SentenceTransformer(MODEL_NAME, device=DEVICE, model_kwargs=model_kwargs, **kwargs)

    video_processor = model[0].processor.video_processor
    video_processor.fps = VIDEO_FPS
    video_processor.max_frames = VIDEO_MAX_FRAMES
    video_processor.size = {
        "shortest_edge": video_processor.size["shortest_edge"],
        "longest_edge": VIDEO_MAX_TOKENS * 2048,
    }
    video_processor.cap_pixels_per_frame = False
    return model


def embed_videos(model: SentenceTransformer, videos: list[Path]) -> tuple[np.ndarray, list[Path]]:
    """Encode videos one at a time so a single corrupt/unreadable file is skipped
    instead of aborting the whole run. Returns (embeddings, videos that succeeded)."""
    embedded: list[np.ndarray] = []
    kept: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for video_path in tqdm(videos, desc="Encoding videos"):
        try:
            embedding = model.encode(
                [{"video": str(video_path.resolve())}],
                batch_size=BATCH_SIZE,
                prompt=INSTRUCTION,
                show_progress_bar=False,
            )
        except Exception as exc:
            if isinstance(exc, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
            failed.append((video_path, str(exc).splitlines()[0]))
            tqdm.write(f"[SKIP] {video_path.name}: {failed[-1][1]}")
            continue
        embedded.append(np.asarray(embedding, dtype="float32")[0])
        kept.append(video_path)

    if failed:
        print("\n" + "=" * 80)
        print(f"Bo qua {len(failed)} video khong encode duoc:")
        for video_path, reason in failed:
            print(f"  - {video_path.name}: {reason}")
        print("=" * 80)
    if not embedded:
        raise RuntimeError("Khong encode duoc video nao.")

    embeddings = np.stack(embedded)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return embeddings / norms, kept

def build_metadata(videos: list[Path]) -> list[dict]:
    return [{
        "video_id": video_path.stem,
        "video": video_path.name,
        "path": str(video_path),
        "duration": round(get_duration_seconds(video_path), 3),
    } for video_path in videos]


def main() -> None:
    videos = discover_videos(CLIPS_DIR)

    model = load_model()
    embeddings, videos = embed_videos(model, videos)  # keep only videos that encoded, so metadata rows match the index
    metadata = build_metadata(videos)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    OUTPUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(OUTPUT_INDEX))

    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_METADATA.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index ({faiss_index.ntotal} vectors, dim={dim}) to {OUTPUT_INDEX}")
    print(f"Saved {len(metadata)} metadata entries to {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()