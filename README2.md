# Cài đặt & chạy: audio_processing.py / video_processing.py / ocr_processing.py

Hướng dẫn ngắn gọn để cài môi trường và chạy 3 script xử lý dữ liệu ở thư mục gốc dự án. Cả 3 đọc/ghi trong thư mục `data/`:

```text
data/
├── videos/   # video gốc (input cho audio_processing.py)
├── clips/    # clip đã cắt theo đoạn thoại (output của audio_processing.py, input của video_processing.py và ocr_processing.py)
├── audio/    # audio .mp3 tách ra từ video (output trung gian của audio_processing.py)
└── bin/      # kết quả cuối: metadata_audio.json, metadata_ocr.json, index.faiss, metadata.json
```

## 1. Yêu cầu chung

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) cài sẵn và có trong `PATH` (dùng bởi `audio_processing.py` để tách audio/cắt clip)
- GPU NVIDIA + driver CUDA nếu muốn chạy nhanh (cả 3 script đều có thể chạy CPU, nhưng chậm hơn nhiều, đặc biệt là OCR và embedding)
- Docker Desktop (kèm NVIDIA Container Toolkit nếu muốn chạy `ocr_processing.py` bằng GPU trong container) — xem mục 4

> Lưu ý: `paddleocr` (dùng bởi `ocr_processing.py`) pin phiên bản `numpy`/`opencv` riêng, dễ xung đột với `torch`/`transformers` (dùng bởi `audio_processing.py` và `video_processing.py`). Nên **tách venv riêng cho OCR** (hoặc dùng Docker, xem mục 4) thay vì cài chung một môi trường.

## 2. Thứ tự chạy

Các script phụ thuộc nhau qua thư mục `data/`, nên chạy theo thứ tự:

1. Bỏ video gốc vào `data/videos/`
2. `audio_processing.py` — tách audio, nhận diện đoạn thoại (VAD), cắt video thành clip theo từng câu vào `data/clips/`, sinh transcript
3. `video_processing.py` — đọc clip trong `data/clips/`, encode thành vector, build FAISS index
4. `ocr_processing.py` — đọc clip trong `data/clips/`, chạy OCR trên từng clip

Bước 3 và 4 độc lập với nhau, có thể chạy song song/theo thứ tự tuỳ ý miễn sau bước 2.

## 3. Cài đặt & chạy từng script

### 3.1. audio_processing.py

Tách audio bằng ffmpeg, phiên âm bằng faster-whisper (VAD tự động bỏ khoảng lặng), cắt clip video theo từng đoạn thoại, xuất `data/bin/metadata_audio.json` (mỗi entry có `start`, `end`, `audio`, `score`).

```bash
python -m venv .venv-audio
.venv-audio\Scripts\activate        # Windows PowerShell: .venv-audio\Scripts\Activate.ps1
pip install faster-whisper
```

Chạy:

```bash
python audio_processing.py
```

Cấu hình ở đầu file (`WHISPER_MODEL_SIZE`, `WHISPER_LANGUAGE`, `DEVICE`, `COMPUTE_TYPE`): mặc định dùng model `small`, `device="cuda"`, `compute_type="float16"`. Nếu máy không có GPU, đổi `DEVICE = "cpu"` và `COMPUTE_TYPE = "int8"` (hoặc `"float32"`) trước khi chạy.

### 3.2. video_processing.py

Đọc các clip trong `data/clips/`, encode bằng Qwen3-VL-Embedding, build FAISS `IndexFlatIP`, xuất `data/bin/index.faiss` + `data/bin/metadata.json`.

```bash
python -m venv .venv-video
.venv-video\Scripts\activate
pip install opencv-python pillow numpy torch faiss-cpu
pip install -U sentence-transformers "transformers>=4.57.0" qwen-vl-utils accelerate
```

Chạy:

```bash
python video_processing.py
```

Lưu ý: `Qwen3-VL-Embedding-2B` khá nặng (~4GB weight); trên GPU nhỏ (vd. RTX 3060 6GB) giữ `BATCH_SIZE = 1` như mặc định trong script. Model tải lần đầu cần Internet.

### 3.3. ocr_processing.py

Sample vài frame mỗi clip, chạy PaddleOCR (tiếng Việt), giữ cả text và confidence score của từng cụm từ, xuất `data/bin/metadata_ocr.json`.

**Cách 1 — venv riêng (chạy trực tiếp trên máy):**

```bash
python -m venv .venv-ocr
.venv-ocr\Scripts\activate
pip install -r requirements-ocr.txt
```

Mặc định script bật `use_gpu=True` (dòng `ocr_engine = PaddleOCR(...)` trong `ocr_processing.py`). Nếu máy không có GPU hoặc cài `paddlepaddle` bản CPU, sửa `use_gpu=True` thành `use_gpu=False` trước khi chạy (PaddleOCR sẽ lỗi nếu để `True` mà chạy trên bản CPU-only).

Chạy:

```bash
python ocr_processing.py
```

**Cách 2 — chạy bằng Docker (khuyên dùng, khỏi lo xung đột phiên bản/paddle GPU, xem chi tiết mục 4.**

## 4. Chạy ocr_processing.py bằng Docker

Repo có sẵn `Dockerfile.ocr` và `docker-compose.ocr.yml`, đóng gói riêng môi trường PaddleOCR (tách biệt khỏi audio/video để không đụng version `numpy`/`opencv`).

### 4.1. Build (mặc định GPU)

```bash
docker compose -f docker-compose.ocr.yml build
```

Yêu cầu: Docker Desktop đã bật GPU support (Settings > Resources > WSL Integration) + driver NVIDIA mới, base image dùng `paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4`.

### 4.2. Chạy một lần (batch job, container tự xoá sau khi xong)

```bash
docker compose -f docker-compose.ocr.yml run --rm ocr
```

Container mount sẵn `data/clips` (đọc, read-only) và `data/bin` (ghi kết quả `metadata_ocr.json`) từ máy host, nên kết quả sau khi chạy nằm luôn ở `data/bin/metadata_ocr.json` trên máy thật, không cần copy ra khỏi container.

### 4.3. Build bản CPU (máy không có GPU NVIDIA)

```bash
docker build -f Dockerfile.ocr --build-arg BASE_IMAGE=paddlepaddle/paddle:2.6.1 -t ocr-processing:cpu .
```

Trước khi build/chạy bản này, sửa `ocr_processing.py`: đổi `use_gpu=True` thành `use_gpu=False` (dòng khởi tạo `PaddleOCR(...)`), nếu không PaddleOCR sẽ raise lỗi vì `paddlepaddle` bản CPU không có CUDA.

Chạy bản CPU (không cần cờ `--gpus`/GPU reservation của compose file):

```bash
docker run --rm -v "${PWD}/data/clips:/app/data/clips:ro" -v "${PWD}/data/bin:/app/data/bin" ocr-processing:cpu
```

### 4.4. Giữ container lại để debug (không tự xoá sau khi chạy)

```bash
docker compose -f docker-compose.ocr.yml run -d --name ocr-debug --entrypoint sleep ocr infinity
docker exec -it ocr-debug bash
# trong container: python ocr_processing.py
```

Dọn dẹp khi xong:

```bash
docker rm -f ocr-debug
```

## 5. Kết quả sau khi chạy đủ 3 script

```text
data/bin/
├── metadata_audio.json   # transcript + score từng câu thoại, theo từng clip (audio_processing.py)
├── metadata.json         # metadata clip theo đúng thứ tự vector trong index.faiss (video_processing.py)
├── index.faiss           # FAISS IndexFlatIP chứa embedding từng clip (video_processing.py)
└── metadata_ocr.json      # text + score OCR theo từng clip (ocr_processing.py)
```

## 6. Xử lý lỗi thường gặp

- **`ffmpeg: command not found`**: chưa cài ffmpeg hoặc chưa có trong `PATH` — cần cho `audio_processing.py`.
- **`RuntimeError: Không tồn tại thư mục video/clip`**: chưa bỏ video vào `data/videos/` (cho `audio_processing.py`) hoặc chưa chạy `audio_processing.py` trước để tạo `data/clips/` (cho `video_processing.py`/`ocr_processing.py`).
- **PaddleOCR raise lỗi liên quan CUDA/GPU**: đang để `use_gpu=True` nhưng chạy trên bản `paddlepaddle` CPU-only, hoặc container không thấy GPU — kiểm tra `docker run --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi` chạy được trước.
- **Xung đột `numpy`/`opencv` khi cài chung venv cho cả 3 script**: tách venv riêng cho OCR như hướng dẫn ở mục 3.3, hoặc dùng Docker (mục 4).
