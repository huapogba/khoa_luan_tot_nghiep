let currentPage = 1;
let totalPages = 1;

let currentContext = [];
let currentCenterTime = null;

/* =========================
   SEARCH MAIN
========================= */

async function search(page = 1) {

    document.getElementById("backBtn").style.display = "none";
    currentContext = [];
    currentCenterTime = null;

    const query = document.getElementById("query").value.trim();
    const topk = document.getElementById("topk").value;

    if (!query) {
        alert("Please enter a search query");
        return;
    }

    const params = new URLSearchParams();
    params.append("q", query);
    params.append("page", page);
    params.append("topk", topk);

    const res = await fetch(
        `http://127.0.0.1:5000/chat?${params}`
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

    if (!results.length) {
        gallery.innerHTML = "<p>No results</p>";
        return;
    }

    results.forEach((item, index) => {

        const card = document.createElement("div");
        card.className = "result video-card";

        const frame = (item.frames && item.frames.length) ? `http://127.0.0.1:5000${item.frames[0]}` : "";
        const summary = item.summary ? item.summary.slice(0, 180) : "No description available";
        const modeLabel = item.source || "video";

        card.innerHTML = `
           <div class="video-thumb-wrap">
               <img src="${frame}" alt="${item.video}">
               <span class="video-rank">#${index + 1}</span>
               <span class="video-score">${item.score !== undefined ? item.score.toFixed(4) : "0.0000"}</span>
           </div>
           <div class="info">
               <div class="video-meta-row">
                   <span class="video-pill">${modeLabel}</span>
                   <h3>${item.video}</h3>
               </div>
               <p><b>Time:</b> ${Number(item.start).toFixed(1)}s - ${Number(item.end).toFixed(1)}s</p>
               <p class="video-summary">${summary}</p>
           </div>
        `;

        card.onclick = () => {
           if (item.frames && item.frames.length > 0) {
               const firstFrame = item.frames[0];
               const player = document.getElementById("videoPlayer");
               player.src = `http://127.0.0.1:5000/data/clips/${item.video}.mp4`;
               player.load();
               player.onloadedmetadata = () => {
                   player.currentTime = Number(item.start || 0);
                   player.play();
               };
           }
        };

        gallery.appendChild(card);
    });

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
    updatePagination();
};
