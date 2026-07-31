let currentPage = 1;
let totalPages = 1;

let currentContext = [];
let currentCenterTime = null;

/* =========================
   MODE CHANGE (show/hide Object Label input)
========================= */

function onModeChange() {

    const mode = document.getElementById("searchMode").value;
    const objectGroup = document.getElementById("objectGroup");

    if (mode === "object_clip") {
        objectGroup.style.display = "block";
    } else {
        objectGroup.style.display = "none";
    }
}

/* =========================
   SEARCH MAIN
========================= */

async function search(page = 1) {

    document.getElementById("backBtn").style.display = "none";
    currentContext = [];
    currentCenterTime = null;

    const query = document.getElementById("query").value.trim();
    const mode = document.getElementById("searchMode").value;
    const topk = document.getElementById("topk").value;
    const videoId = document.getElementById("videoId").value.trim();
    const objectLabel = document.getElementById("objectLabel").value.trim();

    let url = "";
    let params = new URLSearchParams();

    params.append("q", query);
    params.append("page", page);
    params.append("topk", topk);

    switch (mode) {

        case "clip":
            url = "/search";
            break;

        case "viclip":
            url = "/video-search";
            break;

        case "object_clip":

            if (!objectLabel) {
                alert("Please enter Object Label");
                return;
            }

            url = "/object-clip";
            params.append("object", objectLabel);
            break;

        case "clip_in_video":

            if (!videoId) {
                alert("Please enter Video ID");
                return;
            }

            url = "/clip/in-video";
            params.append("video", videoId);
            break;

        case "ocr":
            url = "/ocr";
            break;

        case "audio":
            url = "/audio";
            break;
    }

    const res = await fetch(
        `http://127.0.0.1:5000${url}?${params}`
    );

    const data = await res.json();

    currentPage = data.page || 1;
    totalPages = data.total_pages || 1;

    renderResults(data.results || []);

    updatePagination();
}

/* =========================
   PAGINATION
========================= */

function updatePagination() {

    document.getElementById("pageInfo").innerText =
        `Page ${currentPage} / ${totalPages}`;
}

function changePage(step) {

    const next = currentPage + step;

    if (next < 1 || next > totalPages)
        return;

    search(next);
}

/* =========================
   RENDER RESULTS
========================= */

function renderResults(results) {

    const gallery = document.getElementById("results");
    gallery.innerHTML = "";

    const mode = document.getElementById("searchMode").value;

    if (!results.length) {
        gallery.innerHTML = "<p>No results</p>";
        return;
    }

    results.forEach(item => {

        const card = document.createElement("div");
        card.className = "result";

        /* ======================
           AUDIO
        ====================== */
        if (mode === "audio") {

          const thumbnail = `http://127.0.0.1:5000${item.thumbnail}`;

          card.innerHTML = `
            <img class="thumb" src="${thumbnail}" alt="thumbnail">
            <div class="info">
               <p><b>${item.video}</b></p>
               <p>${item.start.toFixed(2)} - ${item.end.toFixed(2)}</p>
               <p>${item.text}</p>
            </div>
         `;
  
          card.onclick = () => playAudio(item);
        }

        /* ======================
           ViCLIP
        ====================== */
        else if (mode === "viclip") {

            const thumbnail = `http://127.0.0.1:5000/data/${item.thumbnail}`;

            card.innerHTML = `
                <img class="thumb" src="${thumbnail}" alt="thumbnail">
                <div class="info">
                    <p><b>${item.video}</b></p>
                    <p>${item.start}s - ${item.end}s</p>
                    <p>Score: ${item.score.toFixed(4)}</p>
                </div>
            `;

            card.onclick = () => playVideoClip(item);
        }

        /* ======================
           OBJECT + CLIP SEARCH
        ====================== */
        else if (mode === "object_clip") {

            const frame = `http://127.0.0.1:5000${item.frame}`;

            let objectHTML = "";

            if (item.objects && item.objects.length > 0) {

                objectHTML = item.objects
                    .map(obj =>
                        `<span>
                            ${obj.label}
                            (${obj.score.toFixed(2)})
                        </span>`
                    )
                    .join("<br>");

            }

            card.innerHTML = `
                <img src="${frame}">
                <div class="info">
                    <p><b>${item.video}</b></p>
                    <p>${item.timestamp}s</p>
                    ${item.score !== undefined ? `<p>Score: ${item.score.toFixed(4)}</p>` : ""}
                    <p>${objectHTML}</p>
                </div>
            `;

            card.onclick = () => {

                jumpToTime(item);
                loadContext(item);

            };
        }

        /* ======================
           CLIP / OCR
        ====================== */
        else {

            const frame = `http://127.0.0.1:5000${item.frame}`;

            card.innerHTML = `
                <img src="${frame}">
                <div class="info">
                    <p><b>${item.video}</b></p>
                    <p>${item.timestamp}s</p>
                    ${item.score !== undefined ? `<p>Score: ${item.score.toFixed(4)}</p>` : ""}

                    <!-- ${item.ocr ? `<p>${item.ocr}</p>` : ""} -->  

                </div>
            `;

            card.onclick = () => {

                jumpToTime(item);
                loadContext(item);

            };
        }

        gallery.appendChild(card);

    });

}

/* =========================
   ViCLIP PLAYER
========================= */

function playVideoClip(item) {

    const player = document.getElementById("videoPlayer");

    player.src = `http://127.0.0.1:5000/data/${item.path}`;

    player.load();

    player.onloadedmetadata = () => {

        // Clip đã được cắt sẵn nên bắt đầu từ đầu clip
        player.currentTime = 0;
        player.play();

    };
}
/* =========================
   AUDIO
========================= */

function playAudio(item) {

    const player = document.getElementById("videoPlayer");

    player.src = `http://127.0.0.1:5000${item.clip_path}`;

    player.load();

    player.play();

}

/* =========================
   CONTEXT
========================= */

async function loadContext(item) {

    document.getElementById("backBtn").style.display = "block";

    const url =
        `http://127.0.0.1:5000/clip/context?video=${item.video}&timestamp=${item.timestamp}&window=20`;

    const res = await fetch(url);

    const data = await res.json();

    currentContext = data.frames || [];

    currentCenterTime = data.center_timestamp;

    renderContext(currentContext);

}

function renderContext(frames) {

    const gallery = document.getElementById("results");

    gallery.innerHTML = "";

    frames.forEach(item => {

        const frame = `http://127.0.0.1:5000${item.frame}`;

        const card = document.createElement("div");

        card.className = "result";

        if (item.timestamp === currentCenterTime) {

            card.style.border = "3px solid lime";

        }

        card.innerHTML = `
            <img src="${frame}">
            <div class="info">
                <p><b>${item.video}</b></p>
                <p>${item.timestamp}s</p>
            </div>
        `;

        card.onclick = () => {

            jumpToTime(item);

            loadContext(item);

        };

        gallery.appendChild(card);

    });

}

/* =========================
   BACK
========================= */

function backToSearch() {

    document.getElementById("backBtn").style.display = "none";

    search(currentPage);

}

/* =========================
   PLAY FULL VIDEO
========================= */

function jumpToTime(item) {

    const player = document.getElementById("videoPlayer");

    player.src = `http://127.0.0.1:5000/data/clips/${item.video}.mp4`;

    player.onloadedmetadata = () => {

        player.currentTime = Number(item.timestamp);

        player.play();

    };

}

/* =========================
   INIT
========================= */

window.onload = () => {
    onModeChange();
};
