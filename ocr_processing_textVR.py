"""OCR pipeline for TextVR clips (English only).

Same shape as ocr_processing.py, but every clip in TextVR_data/Video is
OCR'd with the English PaddleOCR model -- no language lookup from a
dataset JSON.

Clip filenames look like: <youtube_id>_<start>_<end>.mp4 or <id>_<n>.mp4
    e.g. -LIS5b0AqtE_00000_00010.mp4, 05ZSU5UkXw_0.mp4
The whole file stem is used as video_id (each file is an independent clip).

For each clip: sample FRAMES_PER_CLIP frames, run PaddleOCR on each,
and keep BOTH the recognized text AND its confidence score for every
detected phrase (line).

Output per clip:
    "ocr"         -> full text, all phrases joined with "\\n"
    "ocr_phrases" -> list of {"text": str, "score": float} for every
                     detected phrase, in detection order across all
                     sampled frames (score is PaddleOCR's own recognition
                     confidence, 0-1, rounded to 4 decimals)

The output file is saved every SAVE_EVERY clips; re-running the script
skips clips already present in OUTPUT_METADATA (resume).
"""

import json
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

# =========================
# CONFIGURATION
# =========================
CLIPS_DIR = Path("TextVR_data/Video/test")          # folder holding the clips to OCR
OUTPUT_METADATA = Path("data/bin/metadata_textvr_ocr.json")

PADDLE_LANG = "en"
FRAMES_PER_CLIP = 4            # frames sampled per clip for OCR
SAVE_EVERY = 100               # checkpoint the output file every N clips
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# use_angle_cls=True + cls=True in .ocr() below handles rotated text.
# (Older API names; if you're on paddleocr>=3.x and this errors, swap to
#  use_textline_orientation=True and drop the cls= kwarg in run_ocr_on_clip.)
_ocr_engine: PaddleOCR | None = None


def get_ocr_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            use_angle_cls=True, lang=PADDLE_LANG, use_gpu=True, show_log=False
        )
    return _ocr_engine


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
    """
    ocr_engine = get_ocr_engine()
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
    duration = get_duration_seconds(clip_path)

    phrases = run_ocr_on_clip(clip_path)
    ocr_text = "\n".join(p["text"] for p in phrases)

    return {
        "video_id": clip_path.stem,
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


def load_existing(output_path: Path) -> list[dict]:
    if not output_path.is_file():
        return []
    with output_path.open(encoding="utf-8") as file:
        return json.load(file)


def save_metadata(metadata: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)


def main() -> None:
    clips = discover_clips(CLIPS_DIR)
    if not clips:
        raise FileNotFoundError(f"Không tìm thấy clip trong {CLIPS_DIR}")

    metadata = load_existing(OUTPUT_METADATA)
    done = {entry["video"] for entry in metadata}
    if done:
        print(f"Resume: đã có {len(done)} clip trong {OUTPUT_METADATA}")

    skipped: list[tuple[str, str]] = []
    processed_since_save = 0
    for index, clip_path in enumerate(clips, start=1):
        if clip_path.name in done:
            continue

        try:
            entry = build_metadata_entry(clip_path)
        except Exception as exc:
            skipped.append((clip_path.name, str(exc).splitlines()[0]))
            print(f"[{index}/{len(clips)}] {clip_path.name}: SKIP ({exc})")
            continue
        metadata.append(entry)
        preview = entry["ocr_phrases"][:1]
        print(f"[{index}/{len(clips)}] {clip_path.name}: "
              f"{len(entry['ocr_phrases'])} cụm từ {preview}")

        processed_since_save += 1
        if processed_since_save >= SAVE_EVERY:
            save_metadata(metadata, OUTPUT_METADATA)
            processed_since_save = 0

    if skipped:
        print(f"\nBỏ qua {len(skipped)} video:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    save_metadata(metadata, OUTPUT_METADATA)
    print(f"Saved {len(metadata)} clip entries to {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
