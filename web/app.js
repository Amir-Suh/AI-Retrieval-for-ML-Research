"use strict";

const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const rerankInput = document.getElementById("rerank");
const goBtn = document.getElementById("go");
const statusEl = document.getElementById("status");
const timingsEl = document.getElementById("timings");

const LISTS = {
  dense: document.getElementById("list-dense"),
  sparse: document.getElementById("list-sparse"),
  fused: document.getElementById("list-fused"),
  reranked: document.getElementById("list-reranked"),
};

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function arxivUrl(id) {
  // strip any version suffix for the abstract page
  return `https://arxiv.org/abs/${encodeURIComponent(id)}`;
}

function moveBadge(currentRank, previousRank) {
  if (previousRank == null) return "";
  const delta = previousRank - currentRank; // positive => moved up
  if (delta > 0) return `<span class="move up" title="was #${previousRank} after fusion">▲ ${delta}</span>`;
  if (delta < 0) return `<span class="move down" title="was #${previousRank} after fusion">▼ ${-delta}</span>`;
  return `<span class="move same" title="unchanged">●</span>`;
}

function cardHtml(hit, opts = {}) {
  const cats = (hit.categories || []).slice(0, 3)
    .map((c) => `<span class="cat">${escapeHtml(c)}</span>`).join("");
  const move = opts.showMove ? moveBadge(hit.rank, hit.fused_rank) : "";
  const year = hit.year ? `· ${hit.year}` : "";
  return `
    <div class="card">
      <div class="top">
        <span class="rank">#${hit.rank} ${move}</span>
        <span class="score">${Number(hit.score).toFixed(3)}</span>
      </div>
      <div class="title"><a href="${arxivUrl(hit.arxiv_id)}" target="_blank" rel="noopener">${escapeHtml(hit.title)}</a></div>
      <div class="meta">
        ${cats}
        <span class="id">${escapeHtml(hit.arxiv_id)} ${year}</span>
      </div>
    </div>`;
}

function renderList(el, hits, opts) {
  if (!hits || hits.length === 0) {
    el.innerHTML = `<div class="empty">No results.</div>`;
    return;
  }
  el.innerHTML = hits.map((h) => cardHtml(h, opts)).join("");
}

function showSkeletons() {
  const sk = Array.from({ length: 5 }).map(() => `<div class="skeleton"></div>`).join("");
  Object.values(LISTS).forEach((el) => (el.innerHTML = sk));
}

function renderTimings(t, cfg) {
  const items = [];
  if (t.dense_ms != null) items.push(["vector", t.dense_ms]);
  if (t.sparse_ms != null) items.push(["keyword", t.sparse_ms]);
  if (t.fused_ms != null) items.push(["fusion", t.fused_ms]);
  if (t.rerank_ms != null) items.push(["rerank", t.rerank_ms]);
  timingsEl.innerHTML = items
    .map(([k, v]) => `<span class="chip">${k} <b>${v} ms</b></span>`)
    .join("") + (cfg ? `<span class="chip">fused→rerank <b>${cfg.rerank_candidates}→${cfg.top_k}</b></span>` : "");
}

async function runSearch(query, rerank) {
  goBtn.disabled = true;
  statusEl.className = "status";
  statusEl.textContent = "Searching…";
  timingsEl.innerHTML = "";
  showSkeletons();

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, rerank }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();

    renderList(LISTS.dense, data.stages.dense, {});
    renderList(LISTS.sparse, data.stages.sparse, {});
    renderList(LISTS.fused, data.stages.fused, {});
    renderList(LISTS.reranked, data.stages.reranked, { showMove: true });
    renderTimings(data.timings_ms, data.config);

    statusEl.textContent = `Done — “${data.query}”${rerank ? "" : " (reranking off)"}`;
  } catch (err) {
    statusEl.className = "status error";
    statusEl.textContent = "Error: " + err.message;
    Object.values(LISTS).forEach((el) => (el.innerHTML = ""));
  } finally {
    goBtn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = queryInput.value.trim();
  if (q) runSearch(q, rerankInput.checked);
});
