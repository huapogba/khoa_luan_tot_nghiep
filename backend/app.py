from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from search import (
    search_video_api,
    search_ocr_api,
    search_audio_api,
    get_clip_context_api,
    pick_search_modes,
    search_segment_router,
)

app = Flask(__name__)
CORS(app)

# ==========================
# CHAT-LIKE SEARCH ROUTER
# ==========================
@app.route("/chat")
def chat_route():
    query = request.args.get("q", "")
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("topk", 50))

    if not query:
        return jsonify({"query": query, "page": page, "page_size": page_size, "total": 0, "total_pages": 1, "results": []})

    modes = pick_search_modes(query)
    results = search_segment_router(query, k=200)
    paged = {
        "page": page,
        "page_size": page_size,
        "total": len(results),
        "total_pages": max(1, (len(results) + page_size - 1) // page_size),
        "results": results[(page - 1) * page_size: page * page_size]
    }

    output = []
    for item in paged["results"]:
        output.append({
            "video": item.get("video"),
            "start": item.get("start"),
            "end": item.get("end"),
            "score": round(float(item.get("score", 0)), 4),
            "summary": item.get("summary", ""),
            "frames": item.get("frames", []),
            "source": item.get("source", "video"),
            "selected_modes": modes,
        })

    return jsonify({
        "query": query,
        "page": paged["page"],
        "page_size": paged["page_size"],
        "total": paged["total"],
        "total_pages": paged["total_pages"],
        "selected_modes": modes,
        "results": output,
    })


@app.route("/search")
def search_route():

    query = request.args.get("q", "")
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("topk", 50))

    result = search_video_api(
        query=query,
        page=page,
        page_size=page_size
    )

    return jsonify(result)

# ==========================
# CLIP CONTEXT (CLICK FRAME)
# ==========================
@app.route("/clip/context")
def clip_context_route():

    video = request.args.get("video", "")
    timestamp = request.args.get("timestamp", type=float)
    window = int(request.args.get("window", 20))

    if not video or timestamp is None:
        return jsonify({
            "error": "video and timestamp are required"
        }), 400

    result = get_clip_context_api(
        video=video,
        timestamp=timestamp,
        window=window
    )

    return jsonify(result)

# ==========================
# OCR SEARCH
# ==========================
@app.route("/ocr")
def ocr_route():

    query = request.args.get("q", "")
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("topk", 50))

    result = search_ocr_api(
        query=query,
        page=page,
        page_size=page_size
    )

    return jsonify(result)


# ==========================
# AUDIO SEARCH
# ==========================
@app.route("/audio")
def audio_route():

    query = request.args.get("q", "")
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("topk", 50))

    result = search_audio_api(
        query=query,
        page=page,
        page_size=page_size
    )

    return jsonify(result)


# ==========================
# SERVE STATIC FILES
# ==========================
@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory("data", filename)


@app.route("/")
def home():
    return "Video Retrieval API"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True
    )