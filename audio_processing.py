"""
Audio/speech version of the pipeline (same shape as ocr_processing.py).

Each file in the input folder is treated as ONE independent video.

For each video:
  1. Decode its audio track directly to 16 kHz mono (no intermediate file).
  2. Transcribe with Faster-Whisper (VAD skips silence).
  3. Keep every detected sentence together with its confidence score and
     timestamps. Videos are NOT cut into clips.

Output per video (data/bin/metadata_audio.json):
    "audio"         -> full transcript, all sentences joined with "\\n"
    "audio_phrases" -> list of {"text", "score", "start", "end"} for every
                       detected sentence, in order (score is derived from
                       Whisper's avg_logprob, 0-1)

Videos without an audio stream or without speech still get an entry, with
an empty "audio" and an empty "audio_phrases".

Videos are processed by a small thread pool so audio decoding (CPU)
overlaps with Whisper inference (GPU).
"""

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
from faster_whisper import WhisperModel, decode_audio


# =========================
# CONFIGURATION
# =========================

VIDEO_INPUT_DIR = Path("videos/test")
OUTPUT_METADATA = Path("data/bin/metadata_audio.json")

WHISPER_MODEL_SIZE = "small"
WHISPER_LANGUAGE = "vi"

DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# Greedy decoding is much faster than the default beam_size=5.
BEAM_SIZE = 1

# Threads that decode audio / feed Whisper.
WORKERS = 8

# Transcriptions Whisper runs concurrently on the GPU.
WHISPER_WORKERS = 4

SAMPLE_RATE = 16000

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
    print(f"Beam size    : {BEAM_SIZE}")
    print(f"Workers      : {WORKERS} (whisper: {WHISPER_WORKERS})")
    print("=" * 60, flush=True)

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        num_workers=WHISPER_WORKERS,
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
# VIDEO / AUDIO HELPERS
# =========================

def get_duration_seconds(video_path: Path) -> float:
    try:
        with av.open(str(video_path)) as container:
            return (container.duration or 0) / av.time_base
    except Exception:
        return 0.0


def load_audio(video_path: Path):
    """
    Decode the audio track to a 16 kHz mono float32 array.

    Returns None when the video has no audio stream or cannot be decoded.
    """

    try:
        audio = decode_audio(
            str(video_path),
            sampling_rate=SAMPLE_RATE,
        )
    except Exception:
        return None

    if audio is None or len(audio) == 0:
        return None

    return audio


# =========================
# TRANSCRIBE AUDIO
# =========================

def transcribe_phrases(
    audio,
    whisper_model: WhisperModel,
) -> list[dict]:
    try:
        segments, _ = whisper_model.transcribe(
            audio,
            language=WHISPER_LANGUAGE,
            vad_filter=True,
            beam_size=BEAM_SIZE,
            condition_on_previous_text=False,
        )

        segments = list(segments)

    except Exception as error:
        print(
            f"  Whisper transcription failed: {error}",
            flush=True,
        )

        return []

    phrases = []

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        # Whisper-derived confidence, clamped to [0, 1].
        score = max(
            0.0,
            min(1.0, math.exp(float(segment.avg_logprob))),
        )

        phrases.append(
            {
                "text": text,
                "score": round(score, 4),
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
            }
        )

    return phrases


# =========================
# PROCESS ONE VIDEO
# =========================

def build_metadata_entry(
    video_path: Path,
    whisper_model: WhisperModel,
) -> dict:
    audio = load_audio(video_path)

    if audio is None:
        phrases = []
    else:
        phrases = transcribe_phrases(
            audio,
            whisper_model,
        )

    return {
        # video0.mp4 -> video_id = video0
        "video_id": video_path.stem,
        "video": video_path.name,
        "path": video_path.name,
        "clip_path": str(video_path),
        "start": 0.0,
        "end": round(get_duration_seconds(video_path), 3),
        "audio": "\n".join(p["text"] for p in phrases),
        "audio_phrases": phrases,
    }


# =========================
# MAIN
# =========================

def main() -> None:

    # Transcripts may contain characters the Windows console codepage lacks.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    videos = discover_videos(VIDEO_INPUT_DIR)

    if not videos:
        raise FileNotFoundError(
            f"Không tìm thấy video trong {VIDEO_INPUT_DIR}"
        )

    print("=" * 60)
    print("AUDIO / SPEECH PROCESSING")
    print("=" * 60)
    print(f"Input directory : {VIDEO_INPUT_DIR}")
    print(f"Number of videos: {len(videos)}")
    print(f"Metadata output : {OUTPUT_METADATA}")
    print("=" * 60)

    whisper_model = load_whisper_model()

    metadata: list[dict] = []

    # map() yields results in input order, so metadata stays sorted.
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = executor.map(
            lambda path: build_metadata_entry(
                path,
                whisper_model,
            ),
            videos,
        )

        for index, entry in enumerate(
            results,
            start=1,
        ):
            metadata.append(entry)

            preview = entry["audio_phrases"][:1]

            print(
                f"[{index}/{len(videos)}] {entry['video']}: "
                f"{len(entry['audio_phrases'])} câu {preview}",
                flush=True,
            )

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

    with_speech = sum(1 for e in metadata if e["audio_phrases"])

    print()
    print("=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)
    print(f"Total videos      : {len(videos)}")
    print(f"Videos with speech: {with_speech}")
    print(f"Videos w/o speech : {len(videos) - with_speech}")
    print(f"Metadata saved    : {OUTPUT_METADATA}")
    print("=" * 60)


if __name__ == "__main__":
    main()
