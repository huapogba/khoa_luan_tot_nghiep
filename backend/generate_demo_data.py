import json
from pathlib import Path
import subprocess
import shutil

root = Path(__file__).resolve().parent
clip_dir = root / "data" / "clips"
bin_dir = root / "data" / "bin"
frames_dir = root / "data" / "frames"
clip_dir.mkdir(parents=True, exist_ok=True)
bin_dir.mkdir(parents=True, exist_ok=True)
frames_dir.mkdir(parents=True, exist_ok=True)

video_specs = [
    {
        "video_id": "video_01",
        "title": "Bản tin thời sự",
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Cảnh giao thông đông đúc trong thành phố, nhiều phương tiện lưu thông."},
            {"start": 5.0, "end": 10.0, "text": "Nhân viên kiểm tra an toàn và chủ động xử lý sự cố giao thông."},
            {"start": 10.0, "end": 15.0, "text": "Hệ thống giám sát camera phát hiện xe vượt đèn đỏ ở ngã tư."},
        ],
    },
    {
        "video_id": "video_02",
        "title": "Sản xuất công nghiệp",
        "segments": [
            {"start": 0.0, "end": 6.0, "text": "Hệ thống dây chuyền sản xuất hoạt động liên tục, kiểm soát chất lượng."},
            {"start": 6.0, "end": 12.0, "text": "Nhân viên giám sát và chỉnh sửa chương trình máy móc để tăng hiệu suất."},
            {"start": 12.0, "end": 18.0, "text": "Sản phẩm hoàn thiện được đóng gói và chuyển đến kho lưu trữ."},
        ],
    },
    {
        "video_id": "video_03",
        "title": "Hoạt động ngoài trời",
        "segments": [
            {"start": 0.0, "end": 7.0, "text": "Đội ngũ khảo sát thực địa kiểm tra điều kiện thời tiết và mặt đất."},
            {"start": 7.0, "end": 14.0, "text": "Nhóm kỹ thuật thu thập dữ liệu bằng máy ảnh và thiết bị đo lường."},
            {"start": 14.0, "end": 20.0, "text": "Báo cáo kết quả khảo sát được tổng hợp để đưa ra quyết định tiếp theo."},
        ],
    },
]

for spec in video_specs:
    video_id = spec["video_id"]
    video_path = clip_dir / f"{video_id}.mp4"
    if not video_path.exists():
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:s=1280x720:d=3",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=3",
                    "-shortest",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                    str(video_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            video_path.write_bytes(b"\x00\x00\x00\x00")

    frame_dir = frames_dir / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(6):
        frame_file = frame_dir / f"frame_{idx:03d}.jpg"
        if not frame_file.exists():
            frame_file.write_bytes(b"demo-frame")

all_metadata = []
ocr_entries = []
audio_entries = []

for spec in video_specs:
    video_id = spec["video_id"]
    video_name = f"{video_id}.mp4"
    video_path = f"data/clips/{video_name}"

    for idx, seg in enumerate(spec["segments"]):
        start = float(seg["start"])
        end = float(seg["end"])
        timestamp = start + 0.5
        frame_name = f"data/frames/{video_id}/frame_{idx:03d}.jpg"

        all_metadata.append({
            "video_id": video_id,
            "video": video_name,
            "path": video_path,
            "frame": frame_name,
            "timestamp": timestamp,
            "start": start,
            "end": end,
        })

        ocr_entries.append({
            "video_id": video_id,
            "video": video_name,
            "path": video_path,
            "frame": frame_name,
            "timestamp": timestamp,
            "start": start,
            "end": end,
            "ocr": f"{spec['title']} - {seg['text']} - cảnh {idx + 1}",
            "text": f"{spec['title']} - {seg['text']} - cảnh {idx + 1}",
        })

        audio_entries.append({
            "video_id": video_id,
            "video": video_name,
            "path": video_path,
            "clip_path": video_path,
            "start": start,
            "end": end,
            "text": seg["text"],
            "summary": seg["text"],
        })

with open(bin_dir / "all_metadata.json", "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

with open(bin_dir / "ocr.json", "w", encoding="utf-8") as f:
    json.dump(ocr_entries, f, ensure_ascii=False, indent=2)

with open(bin_dir / "audio.json", "w", encoding="utf-8") as f:
    json.dump(audio_entries, f, ensure_ascii=False, indent=2)

# Keep binary index absent so the app can rebuild it lazily during demo.
print("Demo data generated successfully.")
