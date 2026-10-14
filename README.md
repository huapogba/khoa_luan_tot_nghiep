# Video Retrieval Demo

Dự án này là một hệ thống demo về tìm kiếm video theo truy vấn ngôn ngữ tự nhiên, tích hợp các nguồn dữ liệu như:

- video frame / metadata
- OCR text trên khung hình
- audio transcript / speech text
- ranking kết hợp từ từ khóa và embedding ngữ nghĩa

Mục tiêu của demo là minh họa cách xây dựng một hệ thống truy xuất video theo câu hỏi người dùng, hiển thị các đoạn video có độ liên quan cao và cho phép xem ngữ cảnh khung hình quanh thời điểm tìm được.

## 1. Tổng quan

Project gồm 3 phần chính:

- Backend: Python + Flask
- Frontend: HTML + JavaScript
- Data demo: video, frame hình ảnh, metadata, OCR, audio

Cấu trúc thư mục:

```text
.
├── README.md
├── backend/
│   ├── app.py
│   ├── search.py
│   ├── generate_demo_data.py
│   ├── data/
│   │   ├── bin/
│   │   │   ├── all_metadata.json
│   │   │   ├── ocr.json
│   │   │   └── metadata_audio.json
│   │   ├── clips/
│   │   └── frames/
│   └── __pycache__/
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── MSR-VTT/
│   └── MSRVTT_Videos/
└── process data.ipynb
```

## 2. Chức năng chính

### 2.1 Tìm kiếm video theo truy vấn
- Nhận truy vấn dạng văn bản từ người dùng
- Tìm trên dữ liệu video theo nhiều nguồn
- Sắp xếp kết quả theo điểm score
- Trả về thời điểm bắt đầu/kết thúc, ảnh khung hình và tóm tắt

### 2.2 Tìm kiếm theo OCR
- Tìm trên văn bản được trích xuất từ hình ảnh
- Hỗ trợ tìm theo chữ, số, nhãn, tiêu đề, thông tin màn hình

### 2.3 Tìm kiếm theo Audio
- Tìm trên text transcript của audio
- Dùng index n-gram để khớp các cụm từ

### 2.4 Xem ngữ cảnh khung hình
- Khi click vào kết quả, API trả về frame xung quanh timestamp tương ứng
- Cho phép xem rõ bối cảnh của đoạn video

### 2.5 Giao diện demo
- Nhập query
- Chọn Top K
- Xem video và frame liên quan
- Phân trang kết quả

## 3. Kỹ thuật triển khai

Backend dùng các thành phần chính:

- Flask: xây dựng API
- FAISS: index vector gần đúng cho video search
- NumPy / PyTorch: xử lý embedding và score
- CLIP / Qwen: hỗ trợ embedding ngữ nghĩa nếu model có sẵn
- Googletrans: dịch tiếng Việt sang tiếng Anh
- OCR và audio indexing: tìm kiếm theo văn bản nội bộ

## 3.1. Encode video thành vector

Script [backend/video_processing.py](backend/video_processing.py) chia video thành
các clip dài 10 giây, lấy một số frame đại diện, encode bằng Qwen3-VL-2B trên
GPU (nếu có), rồi trung bình các frame thành một vector cho mỗi clip:

- `backend/data/bin/metadata.json`: metadata của từng clip, theo đúng thứ tự vector
- `backend/data/bin/videos.bin`: FAISS `IndexFlatIP` chứa các vector clip đã chuẩn hóa

Các đường dẫn và cấu hình như `VIDEO_INPUT_DIR`, `OUTPUT_METADATA`, `OUTPUT_INDEX`,
`MODEL_ID`, `CLIP_SECONDS` và `FRAMES_PER_CLIP` nằm ở đầu script để dễ sửa.
Vì Qwen3-VL là VLM sinh văn bản, script xử lý từng frame trong một GPU worker,
đây là lựa chọn an toàn hơn cho GPU RTX 3060 6 GB.

Cài thêm thư viện:

```bash
pip install opencv-python pillow faiss-cpu numpy torch
pip install -U transformers accelerate
```

Chạy từ thư mục `backend`:

```bash
cd backend
python video_processing.py
```

Muốn chỉ định danh sách video cụ thể, sửa `VIDEO_FILES` ở đầu script. Nếu để
`VIDEO_FILES = []`, script sẽ xử lý tất cả video trong `VIDEO_INPUT_DIR`. Mặc định
mỗi vector đại diện cho một đoạn 10 giây; có thể đổi bằng `CLIP_SECONDS`.

## 4. Cài đặt môi trường

```bash
cd backend
pip install flask flask-cors faiss-cpu numpy torch googletrans transformers openai-clip
```

Lưu ý:
- Nếu máy không có GPU, chạy trên CPU vẫn ổn.
- Một số model như CLIP / Qwen có thể tải lần đầu và cần Internet.
- Nếu muốn demo nhanh, dùng dữ liệu mẫu sẵn có trong `backend/data/bin`.

## 5. Tạo dữ liệu demo

```bash
cd backend
python generate_demo_data.py
```

Script này sẽ tạo:

- video demo trong `backend/data/clips`
- ảnh frame trong `backend/data/frames`
- file JSON metadata trong `backend/data/bin`

## 6. Chạy ứng dụng

### 6.1 Chạy backend API

```bash
cd backend
python app.py
```

Backend chạy mặc định tại:

```text
http://127.0.0.1:5000
```

### 6.2 Chạy frontend demo

```bash
cd frontend
python -m http.server 8080
```

Sau đó mở:

```text
http://127.0.0.1:8080
```

## 7. API chính

### GET /chat
Tìm kiếm theo kiểu chat-like, trả về dữ liệu đã phân trang.

Ví dụ:
```bash
http://127.0.0.1:5000/chat?q=xe%20v%C6%B0%E1%BB%A3t%20%C4%91%C3%A8n%20%C4%91%E1%BB%8F&topk=10&page=1
```

### GET /search
Tìm kiếm video theo query chung.

### GET /ocr
Tìm kiếm theo nội dung OCR.

### GET /audio
Tìm kiếm theo transcript audio.

### GET /clip/context
Lấy các frame xung quanh một timestamp để xem ngữ cảnh video.

## 8. Ví dụ query demo

- "xe vượt đèn đỏ"
- "giao thông đông đúc"
- "sản xuất công nghiệp"
- "kiểm tra thời tiết"
- "đóng gói sản phẩm"
- "an toàn giao thông"

## 9. Giới hạn của demo

- Demo sơ khai, chưa tối ưu cho dữ liệu thật lớn.
- Một số mô hình ngữ nghĩa có thể tốn RAM và thời gian tải ban đầu.
- Giao diện phục vụ mục đích demo, chưa có auth hay deployment production.

## 10. Mục tiêu tương lai

- Chuyển sang dữ liệu thực tế
- Tối ưu index cho hàng ngàn video
- Thêm tìm kiếm theo hình ảnh / video embedding
- Xây dựng frontend tốt hơn và API scalable
- Tích hợp multimodal search mạnh hơn

## 11. Kết luận

Đây là một demo cơ bản về hệ thống truy xuất video thông minh, hỗ trợ tìm kiếm bằng câu hỏi tự nhiên và kết hợp nhiều nguồn dữ liệu như video, OCR và audio. Dự án phù hợp để dùng trong buổi thuyết trình, demo chức năng, hoặc làm nền tảng để phát triển hệ thống lớn hơn trong tương lai.
