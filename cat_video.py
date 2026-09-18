"""Scan all videos in a folder, cut EACH into overlapping fixed-length clips.

Window = CLIP_SECONDS, overlap = OVERLAP_SECONDS -> step = CLIP_SECONDS - OVERLAP_SECONDS.
Example (10s window, 5s overlap): clip0 = [0,10], clip1 = [5,15], clip2 = [10,20], ...
"""

import json
import subprocess
from pathlib import Path

import cv2

# =========================
# CONFIGURATION
# =========================
VIDEO_INPUT_DIR = Path("data/videos")
CLIPS_OUTPUT_DIR = Path("data/clips")
OUTPUT_METADATA = Path("data/bin/metadata.json")

CLIP_SECONDS = 10.0
OVERLAP_SECONDS = 5.0
STEP_SECONDS = CLIP_SECONDS - OVERLAP_SECONDS  # 5.0

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Không tồn tại thư mục video: {folder}")
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


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


def cut_clip(video_path: Path, start: float, end: float, clip_path: Path) -> None:
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(end - start),
            "-c", "copy",
            str(clip_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_metadata_for_video(video_path: Path) -> list[dict]:
    video_id = video_path.stem
    duration = get_duration_seconds(video_path)

    entries = []
    start = 0.0
    clip_index = 0
    while start < duration:
        end = min(start + CLIP_SECONDS, duration)
        clip_path = CLIPS_OUTPUT_DIR / f"{video_id}_{clip_index:03d}.mp4"
        cut_clip(video_path, start, end, clip_path)

        entries.append({
            "video_id": video_id,
            "video": video_path.name,
            "path": str(video_path),
            "clip_path": str(clip_path),
            "start": round(start, 3),
            "end": round(end, 3),
        })

        if end >= duration:
            break
        start += STEP_SECONDS
        clip_index += 1
    return entries


def main() -> None:
    videos = discover_videos(VIDEO_INPUT_DIR)
    if not videos:
        raise FileNotFoundError(f"Không tìm thấy video trong {VIDEO_INPUT_DIR}")

    metadata: list[dict] = []
    for index, video_path in enumerate(videos, start=1):
        entries = build_metadata_for_video(video_path)
        metadata.extend(entries)
        print(f"[{index}/{len(videos)}] {video_path.name}: {len(entries)} clips")

    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_METADATA.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(metadata)} clip entries to {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()