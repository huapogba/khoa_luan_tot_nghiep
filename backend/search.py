import faiss
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import torch
from googletrans import Translator

try:
    import clip
except Exception:
    clip = None

# ======================
# DEVICE
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD CLIP MODEL (lazy, loaded only when needed)
# ======================
model = None


def load_clip_model():
    global model

    if model is not None:
        return model

    if clip is None:
        raise RuntimeError("openai-clip package is not installed")

    try:
        model, _ = clip.load("ViT-L/14", device=device)
        model.eval()
        return model
    except Exception as exc:
        print(f"[WARN] CLIP model unavailable: {exc}")
        raise

# ======================
# QWEN3-VL SEARCH MODEL
# ======================
QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
_qwen_tokenizer = None
_qwen_model = None
_qwen_error = None


def load_qwen_encoder():
    global _qwen_tokenizer, _qwen_model, _qwen_error

    if _qwen_model is not None or _qwen_tokenizer is not None:
        return True

    try:
        from transformers import AutoTokenizer

        try:
            from transformers import Qwen3VLForConditionalGeneration as QwenVLModel
        except Exception:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
            except Exception:
                QwenVLModel = None

        if QwenVLModel is None:
            raise ImportError("No compatible Qwen VL model class found in transformers")

        _qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
        _qwen_model = QwenVLModel.from_pretrained(
            QWEN_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        _qwen_model.eval()
        return True
    except Exception as exc:
        _qwen_error = str(exc)
        print(f"[WARN] Qwen3-VL unavailable: {exc}")
        return False


def embed_text_qwen(text):
    if not load_qwen_encoder():
        raise RuntimeError(_qwen_error or "Qwen3-VL model not available")

    text = (text or "").strip()
    if not text:
        return np.zeros(1536, dtype=np.float32)

    inputs = _qwen_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = _qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden = outputs.hidden_states[-1]
    mask = attention_mask.unsqueeze(-1).expand_as(hidden)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    emb = pooled[0].detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ======================
# LOAD DATA
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BIN_DIR = os.path.join(BASE_DIR, "data", "bin")

index = None
all_videos_bin = os.path.join(DATA_BIN_DIR, "all_videos.bin")
if os.path.exists(all_videos_bin):
    index = faiss.read_index(all_videos_bin)

with open(os.path.join(DATA_BIN_DIR, "all_metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open(os.path.join(DATA_BIN_DIR, "ocr.json"), "r", encoding="utf-8") as f:
    ocr_metadata = json.load(f)

with open(os.path.join(DATA_BIN_DIR, "audio.json"), "r", encoding="utf-8") as f:
    audio_metadata = json.load(f)

translator = Translator()

# ======================
# CONFIG PAGINATION
# ======================
PAGE_SIZE = 50
MAX_PAGES = 40


# ======================
# PAGINATION (FIXED)
# ======================
def paginate(results, page=1, page_size=50, max_pages=40):

    total = len(results)

    total_pages = math.ceil(total / page_size)

    if total_pages == 0:
        total_pages = 1

    total_pages = min(total_pages, max_pages)

    page = max(1, min(page, total_pages))

    start = (page - 1) * page_size
    end = start + page_size

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "results": results[start:end]
    }

# ======================
# TRANSLATE + CLIP
# ======================
translate_cache = {}

def translate_vi_to_en(text):
    if text in translate_cache:
        return translate_cache[text]

    try:
        en = translator.translate(text, src='vi', dest='en').text
        print("Translated:", en)
    except Exception as exc:
        print("Translate error:", exc)
        en = text

    translate_cache[text] = en
    return en


def encode_text(text):
    global model

    if model is None:
        model = load_clip_model()

    text_en = translate_vi_to_en(text)
    with torch.no_grad():
        token = clip.tokenize([text_en], truncate=True).to(device)
        feat = model.encode_text(token)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


# ======================
# VIDEO SEARCH (QWEN3-VL-2B)
# ======================
video_text_documents = defaultdict(list)
for item in audio_metadata:
    video_name = str(item.get("video", "")).split(".mp4")[0].split(".avi")[0].split(".mov")[0]
    if video_name:
        text = str(item.get("text", "")).strip()
        if text:
            video_text_documents[video_name].append(text)

for item in ocr_metadata:
    video_name = str(item.get("video", "")).split(".mp4")[0].split(".avi")[0].split(".mov")[0]
    if video_name:
        text = str(item.get("ocr", "")).strip()
        if text:
            video_text_documents[video_name].append(text)

video_document_map = {}
for video_name, pieces in video_text_documents.items():
    merged = " ".join(dict.fromkeys(pieces))
    if not merged:
        continue
    video_document_map[video_name] = merged

video_search_vectors = []
video_search_labels = []
video_search_meta = {}


def normalize_text(text):
    text = text or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9à-ỹ\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_video_search_index():
    global video_search_vectors, video_search_labels, video_search_meta

    if not video_document_map:
        return None

    vectors = []
    labels = []
    meta = {}

    for video_name, doc in video_document_map.items():
        try:
            vec = embed_text_qwen(doc)
        except Exception as exc:
            print(f"[WARN] Unable to embed video doc {video_name}: {exc}")
            continue

        vectors.append(vec)
        labels.append(video_name)
        meta[video_name] = {
            "video": video_name,
            "summary": doc[:500],
            "frame": next(
                (
                    "/" + item["frame"].replace("\\", "/")
                    for item in metadata
                    if str(item.get("video", "")).split(".mp4")[0].split(".avi")[0].split(".mov")[0] == video_name
                ),
                None,
            ),
        }

    if not vectors:
        return None

    arr = np.vstack(vectors).astype(np.float32)
    faiss_index = faiss.IndexFlatIP(arr.shape[1])
    faiss_index.add(arr)

    video_search_vectors = arr
    video_search_labels = labels
    video_search_meta = meta
    return faiss_index


video_faiss_index = None


def keyword_overlap_score(query, document_text):
    q_tokens = [t for t in re.findall(r"[a-z0-9à-ỹ]+", normalize_text(query)) if t]
    d_tokens = [t for t in re.findall(r"[a-z0-9à-ỹ]+", normalize_text(document_text)) if t]
    if not q_tokens or not d_tokens:
        return 0.0

    q_set = set(q_tokens)
    overlap = len(q_set.intersection(set(d_tokens)))
    exact_phrase = 1.0 if normalize_text(query) in normalize_text(document_text) else 0.0
    return min(1.0, (overlap / max(1, len(q_set))) + 0.5 * exact_phrase)


def score_segment(query, segment, source):
    q = normalize_text(query)
    text_parts = []
    for key in ["text", "ocr", "summary", "description", "title"]:
        value = segment.get(key)
        if value:
            text_parts.append(str(value))
    text_blob = " ".join(text_parts)
    lexical = keyword_overlap_score(q, text_blob)

    semantic = 0.0
    try:
        if q and text_blob:
            q_emb = embed_text_qwen(q)
            t_emb = embed_text_qwen(text_blob)
            semantic = float(np.dot(q_emb, t_emb))
    except Exception:
        semantic = 0.0

    source_bonus = {"video": 0.12, "ocr": 0.18, "audio": 0.15}.get(source, 0.1)
    if q and q in normalize_text(text_blob):
        source_bonus += 0.12

    return float(max(0.0, min(1.0, lexical * 0.6 + max(0.0, semantic) * 0.25 + source_bonus * 0.3)))


def get_frames_for_segment(video_name, start, end):
    frames = []
    for item in metadata:
        item_video = str(item.get("video", "")).split(".mp4")[0].split(".avi")[0].split(".mov")[0]
        if item_video != video_name:
            continue
        ts = float(item.get("timestamp", -1))
        if start <= ts <= end:
            frames.append("/" + str(item.get("frame", "")).replace("\\", "/"))
    return frames[:6]


def segmentize_audio_items():
    segments = []
    for item in audio_metadata:
        video = str(item.get("video", "")).split(".mp4")[0].split(".avi")[0].split(".mov")[0]
        if not video:
            continue
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start + 1.0) or (start + 1.0))
        segments.append({
            "video": video,
            "start": start,
            "end": end,
            "frames": get_frames_for_segment(video, start, end),
            "text": item.get("text", ""),
            "source": "audio",
        })
    return segments


def segmentize_ocr_items():
    segments = []
    groups = defaultdict(list)
    for item in ocr_metadata:
        video = str(item.get("video", "")).split(".mp4")[0].split(".avi")[0].split(".mov")[0]
        ts = float(item.get("timestamp", 0.0) or 0.0)
        groups[(video, round(ts, 1))].append(item)

    for (video, ts), items in groups.items():
        text = " ".join(str(i.get("ocr", "")).strip() for i in items if str(i.get("ocr", "")).strip())
        if not text:
            continue
        segments.append({
            "video": video,
            "start": float(ts),
            "end": float(ts + 1.0),
            "frames": ["/" + str(i.get("frame", "")).replace("\\", "/") for i in items[:6]],
            "ocr": text,
            "source": "ocr",
        })
    return segments


def segmentize_video_items():
    segments = []
    for item in metadata:
        video = str(item.get("video", "")).split(".mp4")[0].split(".avi")[0].split(".mov")[0]
        ts = float(item.get("timestamp", 0.0) or 0.0)
        if not video:
            continue
        segments.append({
            "video": video,
            "start": ts,
            "end": ts + 1.0,
            "frames": ["/" + str(item.get("frame", "")).replace("\\", "/")],
            "summary": "Video frame at %.1f seconds" % ts,
            "source": "video",
        })
    return segments


def build_segment_catalog():
    catalog = []
    catalog.extend(segmentize_audio_items())
    catalog.extend(segmentize_ocr_items())
    catalog.extend(segmentize_video_items())
    return catalog


segment_catalog = build_segment_catalog()


def pick_search_modes(query):
    q = normalize_text(query)
    if not q:
        return ["video", "ocr", "audio"]

    if any(word in q for word in ["chữ", "text", "so", "số", "ghi", "nhãn", "mã", "logo", "title", "bảng", "screen", "ocr"]):
        return ["ocr", "video", "audio"]
    if any(word in q for word in ["nói", "audio", "tiếng", "speech", "voice", "thuyết minh", "lời", "sub", "subtitle"]):
        return ["audio", "video", "ocr"]
    return ["video", "ocr", "audio"]


def route_query_with_llm(query):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return pick_search_modes(query)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a video retrieval router. Select the best search function(s) for the user's query. Return JSON: {\"functions\":[\"video\",\"ocr\",\"audio\"]}.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        obj = json.loads(content)
        funcs = obj.get("functions") or obj.get("function") or pick_search_modes(query)
        if isinstance(funcs, str):
            funcs = [funcs]
        return [f for f in funcs if f in ["video", "ocr", "audio"]][:3] or pick_search_modes(query)
    except Exception:
        return pick_search_modes(query)


def search_video(query, k=20):
    global video_faiss_index

    q = (query or "").strip()
    if not q:
        return []

    q_norm = normalize_text(q)
    combined = []

    if video_faiss_index is None:
        video_faiss_index = build_video_search_index()

    if video_faiss_index is not None:
        try:
            qvec = embed_text_qwen(q).astype(np.float32).reshape(1, -1)
            D, I = video_faiss_index.search(qvec, min(k, len(video_search_labels)))

            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(video_search_labels):
                    continue
                video_name = video_search_labels[int(idx)]
                item = video_search_meta.get(video_name, {"video": video_name})
                doc_text = item.get("summary", "")
                keyword_score = keyword_overlap_score(q, doc_text)
                semantic_score = float(score)
                final_score = semantic_score * 0.75 + keyword_score * 0.25
                if q_norm and q_norm in normalize_text(doc_text):
                    final_score += 0.2
                combined.append({
                    "video": item.get("video"),
                    "score": float(final_score),
                    "summary": doc_text[:500],
                    "frame": item.get("frame"),
                })
        except Exception as exc:
            print(f"[WARN] Qwen video search failed: {exc}")

    for video_name, doc in video_document_map.items():
        if any(item["video"] == video_name for item in combined):
            continue
        keyword_score = keyword_overlap_score(q, doc)
        if keyword_score <= 0:
            continue
        combined.append({
            "video": video_name,
            "score": float(keyword_score),
            "summary": doc[:500],
            "frame": next(
                (
                    "/" + item["frame"].replace("\\", "/")
                    for item in metadata
                    if str(item.get("video", "")).split(".mp4")[0].split(".avi")[0].split(".mov")[0] == video_name
                ),
                None,
            ),
        })

    if not combined:
        return []

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:k]


def search_segment_router(query, k=20):
    selected_modes = route_query_with_llm(query)
    candidates = []

    for segment in segment_catalog:
        if segment.get("source") not in selected_modes:
            continue
        score = score_segment(query, segment, segment.get("source", "video"))
        if score <= 0:
            continue
        candidates.append({
            "video": segment.get("video"),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "frames": segment.get("frames", []),
            "score": float(score),
            "summary": segment.get("text") or segment.get("ocr") or segment.get("summary") or "",
            "source": segment.get("source", "video"),
        })

    if not candidates:
        return search_video(query, k=k)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    dedup = {}
    for item in candidates:
        key = (item["video"], round(float(item["start"] or 0), 1), round(float(item["end"] or 0), 1))
        if key not in dedup or item["score"] > dedup[key]["score"]:
            dedup[key] = item
    results = sorted(dedup.values(), key=lambda x: x["score"], reverse=True)[:k]
    return results


def search_video_api(query, page=1, page_size=50):
    results = search_segment_router(query, k=200)
    paged = paginate(results, page, page_size)

    output = []
    for item in paged["results"]:
        output.append({
            "video": item.get("video"),
            "start": item.get("start"),
            "end": item.get("end"),
            "score": round(float(item.get("score", 0)), 4),
            "summary": item.get("summary", ""),
            "frames": item.get("frames", []),
            "source": item.get("source", "video")
        })

    return {
        "query": query,
        "page": paged["page"],
        "page_size": paged["page_size"],
        "total": paged["total"],
        "total_pages": paged["total_pages"],
        "results": output
    }


def search_clip_api(query, page=1, page_size=50):
    return search_video_api(query, page=page, page_size=page_size)

# ======================
# CLIP SEARCH
# ======================
def search_clip(query, k=216):

    qvec = encode_text(query)
    D, I = index.search(qvec, k)

    results = []

    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(metadata):
            item = metadata[idx].copy()   # copy để không sửa metadata gốc
            item["score"] = float(score)
            results.append(item)

    # Sắp xếp theo score giảm dần
    results.sort(key=lambda x: x["score"], reverse=False)

    return results


# ======================
# FRAME LÂN CẬN 
# ======================

video_timeline = defaultdict(list)

for item in metadata:
    if "video" in item and "timestamp" in item:
        video_timeline[item["video"]].append(item)

for v in video_timeline:
    video_timeline[v].sort(key=lambda x: x["timestamp"])

def get_context(video, timestamp, window=20):

    timeline = video_timeline.get(video, [])
    if not timeline:
        return []

    pos = min(
        range(len(timeline)),
        key=lambda i: abs(timeline[i]["timestamp"] - timestamp)
    )

    start = max(0, pos - window)
    end = min(len(timeline), pos + window + 1)

    return timeline[start:end]

def get_clip_context_api(video, timestamp, window=20):

    if video not in video_timeline:
        return {
            "video": video,
            "center_timestamp": timestamp,
            "frames": []
        }

    if timestamp is None:
        return {
            "video": video,
            "center_timestamp": timestamp,
            "frames": []
        }

    frames = get_context(video, timestamp, window)

    return {
        "video": video,
        "center_timestamp": timestamp,
        "frames": [
            {
                "frame": "/" + f["frame"].replace("\\", "/"),
                "timestamp": f.get("timestamp")
            }
            for f in frames
        ]
    }
# ======================
# OCR INDEX
# ======================
class HashOCRIndex:

    def __init__(self):
        self.index = defaultdict(set)
        self.meta = {}

    def clean(self, text):
        text = text.lower()
        text = re.sub(r"[^\wÀ-ỹ\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def add(self, item):
        frame = item["frame"]
        self.meta[frame] = item

        words = self.clean(item.get("ocr", "")).split()

        for w in words:
            self.index[w].add(frame)

    def build(self, data):
        for item in data:
            self.add(item)

    def search(self, query, k=2000):

        words = self.clean(query).split()
        if not words:
            return []

        scores = defaultdict(int)

        for w in words:
            for frame in self.index.get(w, []):
                scores[frame] += 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []

        for frame, _ in ranked[:k]:
            if frame in self.meta:
                results.append(self.meta[frame])

        return results


# build OCR index
hash_index = HashOCRIndex()
hash_index.build(ocr_metadata)


def search_ocr(query, page=1, page_size=50):

    results = hash_index.search(query, k=200)
    return paginate(results, page, page_size)


def search_ocr_api(query, page=1, page_size=50):

    paged = search_ocr(query, page, page_size)

    output = []

    for item in paged["results"]:
        output.append({
            "frame": "/" + item["frame"].replace("\\", "/"),
            "timestamp": item.get("timestamp"),
            "video": item.get("video"),
            "ocr": item.get("ocr")
        })

    return {
        "query": query,
        "page": paged["page"],
        "page_size": paged["page_size"],
        "total": paged["total"],
        "total_pages": paged["total_pages"],
        "results": output
    }


# ======================
# AUDIO SEARCH
# ======================
class PhraseAudioIndex:

    def __init__(self, ngram=3):
        self.ngram = ngram
        self.index = defaultdict(set)
        self.meta = {}

    def clean(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9À-ỹ\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def add(self, item):
        text = self.clean(item.get("text", ""))
        tokens = text.split()

        if not tokens:
            return

        clip_id = item["clip_path"]
        if not clip_id:
           return
        self.meta[clip_id] = item

        # word + phrase index
        for n in range(1, self.ngram + 1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i:i+n])
                self.index[phrase].add(clip_id)

    def build(self, data):
        for item in data:
            self.add(item)

audio_index = PhraseAudioIndex(ngram=3)
audio_index.build(audio_metadata)

def search_audio(query, k=2000):

    query = query.lower().strip()
    query = re.sub(r"\s+", " ", query)

    tokens = query.split()

    scores = defaultdict(int)

    # ưu tiên phrase dài trước
    for n in range(min(3, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])

            for clip_id in audio_index.index.get(phrase, []):
                scores[clip_id] += n  # phrase dài điểm cao hơn

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []

    for clip_id, _ in ranked[:k]:
        item = audio_index.meta.get(clip_id)
        if not item:
            continue

        results.append({
            "video": item.get("video"),
            "clip_path": "/data/clips/" + clip_id.split("data/clips")[-1].replace("\\", "/"),
            "start": float(item.get("start", 0)),
            "end": float(item.get("end", 0)),
            "text": item.get("text")
        })

    return results

def search_audio_api(query, page=1, page_size=50):

    results = search_audio(query, k=2000)
    paged = paginate(results, page, page_size)

    return {
        "query": query,
        "page": paged["page"],
        "page_size": paged["page_size"],
        "total": paged["total"],
        "total_pages": paged["total_pages"],
        "results": paged["results"]
    }

