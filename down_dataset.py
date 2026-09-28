"""
Download every video listed in multivent_base.json.

    multivent_base.json  -> video_id + video_URL of every record
    multivent_base/      -> <video_id>.mp4 for every downloaded video

YouTube, Twitter and news-site URLs are all handled by yt-dlp.
Videos already in the output folder are skipped, so the script can be
re-run to resume or retry. Failed downloads are written to
multivent_base/download_failed.json together with the error message.

Requirements: pip install yt-dlp, and ffmpeg on PATH (merges the video
and audio streams into one mp4).

Example:
    python down_dataset.py
    python down_dataset.py --workers 8 --max-height 480
    python down_dataset.py --limit 5
    python down_dataset.py --cookies-from-browser chrome
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yt_dlp


PROJECT_ROOT = Path(__file__).resolve().parent

JSON_FILE = PROJECT_ROOT / "multivent_base.json"
OUTPUT_DIR = PROJECT_ROOT / "multivent_base"
FAILED_FILE_NAME = "download_failed.json"

DEFAULT_WORKERS = 4
DEFAULT_MAX_HEIGHT = 720
DEFAULT_RETRIES = 3


def load_videos(json_file):
    """
    Returns [(video_id, url)], one per unique video_id.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    videos = {}

    for item in records:
        video_id = item.get("video_id")
        url = item.get("video_URL")

        if video_id and url:
            videos.setdefault(video_id, url)

    if len(videos) != len(records):
        print(f"WARNING: {len(records) - len(videos)} duplicate / incomplete records skipped.")

    return list(videos.items())


def find_downloaded(output_dir, video_id):
    """
    Existing finished file for this video (partial downloads ignored).
    """
    for path in output_dir.glob(f"{video_id}.*"):
        if path.suffix not in (".part", ".ytdl") and path.stat().st_size > 0:
            return path

    return None


class SilentLogger:
    """
    yt-dlp prints errors to stderr even when quiet; the error is already
    reported through the raised exception.
    """

    def debug(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


def download_video(video_id, url, output_dir, max_height, retries, cookies_browser=None):
    """
    Returns (status, detail): status is "ok", "skipped" or "failed".
    """
    existing = find_downloaded(output_dir, video_id)

    if existing:
        return "skipped", existing.name

    options = {
        "outtmpl": str(output_dir / f"{video_id}.%(ext)s"),
        # Best mp4 up to max_height, falling back to any single file.
        "format": (
            f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
            f"b[height<={max_height}][ext=mp4]/"
            f"bv*[height<={max_height}]+ba/"
            f"b[height<={max_height}]/b"
        ),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": retries,
        "fragment_retries": retries,
        "socket_timeout": 30,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": SilentLogger(),
    }

    if cookies_browser:
        options["cookiesfrombrowser"] = (cookies_browser,)

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as error:
        # yt-dlp messages start with "ERROR: " and may span lines.
        message = str(error).replace("ERROR: ", "").strip().splitlines()[0]
        return "failed", message

    downloaded = find_downloaded(output_dir, video_id)

    if not downloaded:
        return "failed", "yt-dlp finished but no file was written"

    return "ok", downloaded.name


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Download the MultiVENT videos listed in multivent_base.json."
    )
    parser.add_argument("--json", type=Path, default=JSON_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel downloads (default: 4).",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=DEFAULT_MAX_HEIGHT,
        help="Maximum video height in pixels (default: 720).",
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help="Use the login cookies of a browser (chrome, edge, firefox...). "
             "Needed for many Twitter/X videos and age-restricted YouTube videos.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N videos for a quick test.",
    )

    args = parser.parse_args()

    if not args.json.exists():
        raise FileNotFoundError(f"--json not found: {args.json}")

    videos = load_videos(args.json)

    if args.limit is not None:
        videos = videos[:args.limit]

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTIVENT DOWNLOAD")
    print("=" * 70)
    print(f"Videos: {len(videos)}   Output: {args.output}")
    print(f"Workers: {args.workers}   Max height: {args.max_height}p\n")

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failed = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_video,
                video_id,
                url,
                args.output,
                args.max_height,
                args.retries,
                args.cookies_from_browser,
            ): (video_id, url)
            for video_id, url in videos
        }

        for done, future in enumerate(as_completed(futures), start=1):
            video_id, url = futures[future]
            status, detail = future.result()

            counts[status] += 1

            if status == "failed":
                failed.append({"video_id": video_id, "video_URL": url, "error": detail})

            print(f"[{done:4d}/{len(videos)}] {status.upper():<7} {video_id}  {detail}")

    failed.sort(key=lambda item: item["video_id"])
    failed_file = args.output / FAILED_FILE_NAME

    with open(failed_file, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    available = counts["ok"] + counts["skipped"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total videos       : {len(videos)}")
    print(f"Downloaded now     : {counts['ok']}")
    print(f"Already downloaded : {counts['skipped']}")
    print(f"Available in folder: {available}")
    print(f"Failed             : {counts['failed']}")
    print(f"Time               : {time.time() - start_time:.1f}s")
    print(f"\nFailed list saved to:\n{failed_file}")


if __name__ == "__main__":
    main()
