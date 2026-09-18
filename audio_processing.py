"""
Audio/speech version of the clip pipeline.

Each file in data/clips is treated as ONE independent video.

For each video:
  1. Check whether the video contains an audio stream.
  2. If there is no audio, skip the video.
  3. Extract audio to MP3 using ffmpeg.
  4. Transcribe with Faster-Whisper.
  5. Use VAD to detect speech segments.
  6. Cut the corresponding VIDEO segment for each speech segment.
  7. Save metadata containing the transcript and confidence score.

Input:
    data/clips/
        video0.mp4
        video1.mp4
        video2.mp4

Output:
    data/audio/
        video0.mp3
        video1.mp3
        ...

    data/clips_2/
        video0_000.mp4
        video0_001.mp4
        ...

    data/bin/metadata_audio.json
"""

import json
import math
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel


# =========================
# CONFIGURATION
# =========================

VIDEO_INPUT_DIR = Path("data/clips")
AUDIO_OUTPUT_DIR = Path("data/audio")
CLIPS_OUTPUT_DIR = Path("data/clips_2")
OUTPUT_METADATA = Path("data/bin/metadata_audio.json")

WHISPER_MODEL_SIZE = "small"
WHISPER_LANGUAGE = "vi"

DEVICE = "cuda"
COMPUTE_TYPE = "float16"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# =========================
# LOAD WHISPER MODEL
# =========================

def load_whisper_model() -> WhisperModel:
    print("=" * 60)
    print("Loading Faster-Whisper model...")
    print(f"Model        : {WHISPER_MODEL_SIZE}")
    print(f"Language     : {WHISPER_LANGUAGE}")
    print(f"Device       : {DEVICE}")
    print(f"Compute type : {COMPUTE_TYPE}")
    print("=" * 60, flush=True)

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    print("Whisper model loaded successfully.", flush=True)
    print()

    return model


# =========================
# DISCOVER VIDEOS
# =========================

def discover_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Không tồn tại thư mục video: {folder}"
        )

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


# =========================
# CHECK AUDIO STREAM
# =========================

def has_audio_stream(video_path: Path) -> bool:
    """
    Check whether the video contains an audio stream.

    Returns:
        True  -> video has audio
        False -> video has no audio or ffprobe failed
    """

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        return bool(result.stdout.strip())

    except Exception as error:
        print(
            f"  ffprobe error: {error}",
            flush=True,
        )
        return False


# =========================
# EXTRACT AUDIO
# =========================

def extract_audio(
    video_path: Path,
    audio_path: Path,
) -> bool:
    """
    Extract audio from video.

    Returns:
        True  -> extraction succeeded
        False -> extraction failed
    """

    audio_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"  Extracting audio: {video_path.name}",
        flush=True,
    )

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-acodec",
                "libmp3lame",
                "-q:a",
                "2",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"  Failed to extract audio: "
                f"{video_path.name}",
                flush=True,
            )

            if result.stderr:
                print(
                    result.stderr[-1000:],
                    flush=True,
                )

            return False

        print(
            f"  Audio saved: {audio_path}",
            flush=True,
        )

        return True

    except Exception as error:
        print(
            f"  Audio extraction error: {error}",
            flush=True,
        )
        return False


# =========================
# CUT VIDEO CLIP
# =========================

def cut_clip(
    video_path: Path,
    start: float,
    end: float,
    clip_path: Path,
) -> bool:
    """
    Cut a video segment.

    Returns:
        True  -> success
        False -> failed
    """

    clip_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    duration = end - start

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-i",
                str(video_path),
                "-t",
                str(duration),
                "-c",
                "copy",
                str(clip_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"    Failed to cut clip: {clip_path}",
                flush=True,
            )

            if result.stderr:
                print(
                    result.stderr[-500:],
                    flush=True,
                )

            return False

        return True

    except Exception as error:
        print(
            f"    Clip cutting error: {error}",
            flush=True,
        )
        return False


# =========================
# TRANSCRIBE AUDIO
# =========================

def transcribe_segments(
    audio_path: Path,
    whisper_model: WhisperModel,
) -> list:

    print(
        f"  Transcribing: {audio_path.name}",
        flush=True,
    )

    try:
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            vad_filter=True,
        )

        segments = list(segments)

        print(
            f"  Detected speech segments: "
            f"{len(segments)}",
            flush=True,
        )

        return segments

    except Exception as error:
        print(
            f"  Whisper transcription failed: "
            f"{error}",
            flush=True,
        )

        return []


# =========================
# PROCESS ONE VIDEO
# =========================

def build_metadata_for_video(
    video_path: Path,
    whisper_model: WhisperModel,
) -> list[dict]:

    # Each file is an independent video.
    #
    # video0.mp4 -> video_id = video0
    # video1.mp4 -> video_id = video1
    # video2.mp4 -> video_id = video2

    video_id = video_path.stem

    print()
    print("-" * 60)
    print(f"Processing video: {video_path.name}")
    print(f"Video ID       : {video_id}")
    print("-" * 60)

    # =========================
    # CHECK AUDIO
    # =========================

    print(
        "  Checking audio stream...",
        flush=True,
    )

    if not has_audio_stream(video_path):
        print(
            f"  No audio stream: "
            f"{video_path.name} -> SKIP",
            flush=True,
        )

        return []

    print(
        "  Audio stream found.",
        flush=True,
    )

    # =========================
    # EXTRACT AUDIO
    # =========================

    audio_path = (
        AUDIO_OUTPUT_DIR
        / f"{video_id}.mp3"
    )

    success = extract_audio(
        video_path,
        audio_path,
    )

    if not success:
        print(
            f"  Cannot extract audio: "
            f"{video_path.name} -> SKIP",
            flush=True,
        )

        return []

    # =========================
    # TRANSCRIBE
    # =========================

    segments = transcribe_segments(
        audio_path,
        whisper_model,
    )

    if not segments:
        print(
            f"  No speech detected: "
            f"{video_path.name}",
            flush=True,
        )

        return []

    # =========================
    # BUILD METADATA
    # =========================

    entries = []

    for clip_index, segment in enumerate(
        segments
    ):

        start = float(segment.start)
        end = float(segment.end)

        if end <= start:
            continue

        text = segment.text.strip()

        if not text:
            continue

        clip_path = (
            CLIPS_OUTPUT_DIR
            / f"{video_id}_{clip_index:03d}.mp4"
        )

        print(
            f"  [{clip_index:03d}] "
            f"{start:.3f}s -> "
            f"{end:.3f}s | "
            f"{text}",
            flush=True,
        )

        # =========================
        # CUT VIDEO
        # =========================

        success = cut_clip(
            video_path,
            start,
            end,
            clip_path,
        )

        if not success:
            continue

        # =========================
        # WHISPER SCORE
        # =========================

        avg_logprob = float(
            segment.avg_logprob
        )

        score = math.exp(avg_logprob)

        score = max(
            0.0,
            min(1.0, score),
        )

        # =========================
        # METADATA
        # =========================

        entries.append(
            {
                "video_id": video_id,

                # Original video
                "video": video_path.name,

                # Original video path
                "path": str(video_path),

                # Generated speech clip
                "clip_path": str(clip_path),

                # Timestamp in original video
                "start": round(start, 3),
                "end": round(end, 3),

                # Whisper transcript
                "audio": text,

                # Whisper-derived score
                "score": round(score, 4),
            }
        )

    print(
        f"  Finished {video_path.name}: "
        f"{len(entries)} speech clips",
        flush=True,
    )

    return entries


# =========================
# MAIN
# =========================

def main() -> None:

    # =========================
    # DISCOVER VIDEOS
    # =========================

    videos = discover_videos(
        VIDEO_INPUT_DIR
    )

    if not videos:
        raise FileNotFoundError(
            f"Không tìm thấy video trong "
            f"{VIDEO_INPUT_DIR}"
        )

    print("=" * 60)
    print("AUDIO / SPEECH PROCESSING")
    print("=" * 60)

    print(
        f"Input directory : "
        f"{VIDEO_INPUT_DIR}"
    )

    print(
        f"Number of videos: "
        f"{len(videos)}"
    )

    print(
        f"Audio output    : "
        f"{AUDIO_OUTPUT_DIR}"
    )

    print(
        f"Clip output     : "
        f"{CLIPS_OUTPUT_DIR}"
    )

    print(
        f"Metadata output : "
        f"{OUTPUT_METADATA}"
    )

    print("=" * 60)

    # =========================
    # LOAD WHISPER
    # =========================

    whisper_model = load_whisper_model()

    # =========================
    # PROCESS VIDEOS
    # =========================

    metadata: list[dict] = []

    processed_count = 0
    skipped_count = 0

    for index, video_path in enumerate(
        videos,
        start=1,
    ):

        print()
        print(
            f"VIDEO [{index}/{len(videos)}]"
        )

        entries = build_metadata_for_video(
            video_path,
            whisper_model,
        )

        if entries:
            processed_count += 1
        else:
            skipped_count += 1

        metadata.extend(entries)

    # =========================
    # SAVE METADATA
    # =========================

    OUTPUT_METADATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_METADATA.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # =========================
    # SUMMARY
    # =========================

    print()
    print("=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"Total videos    : {len(videos)}"
    )

    print(
        f"Processed videos: {processed_count}"
    )

    print(
        f"Skipped videos  : {skipped_count}"
    )

    print(
        f"Speech clips    : {len(metadata)}"
    )

    print(
        f"Metadata saved  : "
        f"{OUTPUT_METADATA}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()