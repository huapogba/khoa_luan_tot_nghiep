"""OCR version of the clip pipeline (same shape as the embedding script, but
detecting text with PaddleOCR instead of computing a Qwen3-VL vector).

Assumes clip filenames look like: <video_id>_<clip_index>.mp4
    e.g. video_01_000.mp4, video_01_001.mp4 ...
(same convention as build_clip_metadata.py -- only CLIP_NAME_PATTERN /
parse_clip_name() need to change if your naming differs)

For each 10s clip: sample FRAMES_PER_CLIP frames, run PaddleOCR on each,
and keep BOTH the recognized text AND its confidence score for every
detected phrase (line), instead of only the text.

Output per clip:
    "ocr"         -> full text, all phrases joined with "\\n" (unchanged,
                     for anything that still greps/searches the raw text)
    "ocr_phrases" -> list of {"text": str, "score": float} for every
                     detected phrase, in detection order across all
                     sampled frames (score is PaddleOCR's own recognition
                     confidence, 0-1, rounded to 4 decimals)
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

# =========================
# CONFIGURATION
# =========================
CLIPS_DIR = Path("data/clips")
OUTPUT_METADATA = Path("data/bin/metadata_ocr.json")

CLIP_SECONDS = 10.0            # nominal clip length used to compute `start`
FRAMES_PER_CLIP = 4            # frames sampled per clip for OCR
CLIP_NAME_PATTERN = re.compile(r"^(?P<video_id>.+?)_?(?P<clip_index>\d+)$")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# use_angle_cls=True + cls=True in .ocr() below handles rotated text.
# (Older API names; if you're on paddleocr>=3.x and this errors, swap to
#  use_textline_orientation=True and drop the cls= kwarg in run_ocr_on_clip.)
ocr_engine = PaddleOCR(use_angle_cls=True, lang="vi", use_gpu=True, show_log=False)


def get_duration_seconds(clip_path: Path) -> float:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được clip: {clip_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    capture.release()
    if fps <= 0:
        raise RuntimeError(f"Không đọc được FPS: {clip_path}")
    return frame_count / fps


def parse_clip_name(clip_path: Path) -> tuple[str, int]:
    """
    video0.mp4 -> video_id='video0', clip_index=0
    video1.mp4 -> video_id='video1', clip_index=1
    video2.mp4 -> video_id='video2', clip_index=2
    """
    video_id = clip_path.stem

    # Nếu mỗi file là một video độc lập thì index này chỉ dùng
    # để giữ tương thích với pipeline hiện tại.
    clip_index = 0

    return video_id, clip_index


def sample_frames(clip_path: Path, num_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được clip: {clip_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    try:
        if frame_count <= 0:
            return frames
        positions = np.linspace(
            0, max(0, frame_count - 1),
            num=min(num_frames, frame_count),
            dtype=np.int64,
        )
        for position in sorted(set(int(value) for value in positions)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if ok:
                frames.append(frame)  # keep BGR, PaddleOCR accepts it directly
    finally:
        capture.release()
    return frames


def run_ocr_on_clip(clip_path: Path) -> list[dict]:
    """Run OCR on every sampled frame of the clip and return one entry per
    detected phrase, keeping PaddleOCR's own recognition confidence score.

    PaddleOCR's ocr() result shape per detection is:
        [ [ [box_points], (text, score) ], ... ]
    -> detection[1][0] is the text, detection[1][1] is the score.
    The same phrase may appear more than once if it's visible across
    several sampled frames; this keeps every occurrence rather than
    deduplicating, since occurrence count / average score can itself be
    a useful signal for ranking OCR search hits later.
    """
    phrases: list[dict] = []
    for frame in sample_frames(clip_path, FRAMES_PER_CLIP):
        result = ocr_engine.ocr(frame, cls=True)
        for page in result or []:
            for detection in page or []:
                text = detection[1][0]
                score = detection[1][1]
                if text:
                    phrases.append({"text": text, "score": round(float(score), 4)})
    return phrases


def build_metadata_entry(clip_path: Path) -> dict:
    video_id, _ = parse_clip_name(clip_path)

    duration = get_duration_seconds(clip_path)

    phrases = run_ocr_on_clip(clip_path)
    ocr_text = "\n".join(p["text"] for p in phrases)

    return {
        "video_id": video_id,
        "video": clip_path.name,
        "path": clip_path.name,
        "clip_path": str(clip_path),
        "start": 0.0,
        "end": round(duration, 3),
        "ocr": ocr_text,
        "ocr_phrases": phrases,
    }


def discover_clips(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Không tồn tại thư mục clip: {folder}")
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def main() -> None:
    clips = discover_clips(CLIPS_DIR)
    if not clips:
        raise FileNotFoundError(f"Không tìm thấy clip trong {CLIPS_DIR}")

    metadata = []
    for index, clip_path in enumerate(clips, start=1):
        entry = build_metadata_entry(clip_path)
        metadata.append(entry)
        preview = entry["ocr_phrases"][:1]
        print(f"[{index}/{len(clips)}] {clip_path.name}: "
              f"{len(entry['ocr_phrases'])} cụm từ {preview}")

    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_METADATA.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(metadata)} clip entries to {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()