"""
Download the MSVD test split (official split: 670 videos, 27,763 captions).

Source: HuggingFace dataset friedrichor/MSVD
    msvd_test.json    -> [{"video_id", "video", "caption": [...]}], 670 videos
    MSVD_Videos.zip   -> all 1,970 .avi clips (~1.8 GB)

Output:
    MSVD_data/msvd_test.json
    MSVD_data/Video/test/<video_id>.avi    (only the 670 test videos)

The zip is downloaded to the HuggingFace cache, so re-running the script
does not download it again. Videos already extracted are skipped.

Requirements: pip install huggingface_hub

Example:
    python down_msvd.py
    python down_msvd.py --delete-zip     # free ~1.8 GB once extracted
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Windows without Developer Mode has no symlinks: the HF cache still works
# (copies instead of links), the warning is just noise.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parent

REPO_ID = "friedrichor/MSVD"
ANNOTATION_FILE = "msvd_test.json"
VIDEO_ZIP = "MSVD_Videos.zip"

OUTPUT_DIR = PROJECT_ROOT / "MSVD_data"
VIDEO_DIR = OUTPUT_DIR / "Video" / "test"


def download(filename):
    print(f"Downloading {REPO_ID}/{filename} ...")

    return Path(
        hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Download the MSVD test split.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--delete-zip",
        action="store_true",
        help="Remove the cached video zip after extraction.",
    )
    args = parser.parse_args()

    video_dir = args.output / "Video" / "test"
    video_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = args.output / ANNOTATION_FILE
    shutil.copyfile(download(ANNOTATION_FILE), annotation_path)

    with open(annotation_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    test_files = {item["video"] for item in records}
    captions = sum(len(item["caption"]) for item in records)
    print(f"Test split: {len(test_files)} videos, {captions} captions")

    missing = {name for name in test_files if not (video_dir / name).exists()}

    if not missing:
        print("All test videos already extracted.")
        return

    print(f"{len(missing)} test videos to extract. The video zip is ~1.8 GB, "
          f"this can take several minutes - let it finish.")
    zip_path = download(VIDEO_ZIP)

    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name

            if member.is_dir() or name not in missing:
                continue

            with archive.open(member) as src, open(video_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)

            missing.discard(name)
            extracted += 1
            print(f"\r  extracted {extracted}", end="", flush=True)

    print()

    if missing:
        print(f"WARNING: {len(missing)} test videos not found in the zip:")
        for name in sorted(missing)[:20]:
            print(f"  - {name}")

    print(f"Videos in {video_dir}: {len(list(video_dir.iterdir()))}")

    if args.delete_zip:
        zip_path.unlink(missing_ok=True)
        print(f"Deleted {zip_path}")


if __name__ == "__main__":
    main()
