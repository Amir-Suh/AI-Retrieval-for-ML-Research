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

const synthBar = document.getElementById("synth-bar");
const summarizeBtn = document.getElementById("summarize");
const graphBtn = document.getElementById("build-graph");
const synthStatus = document.getElementById("synth-status");
const synthOut = document.getElementById("synthesis");

const graphPanel = document.getElementById("graph-panel");
const graphLegend = document.getElementById("graph-legend");
const graphDetail = document.getElementById("graph-detail");
const sharedOnly = document.getElementById("shared-only");

const manualIds = document.getElementById("manual-ids");
const manualLoad = document.getElementById("manual-load");

const qaInput = document.getElementById("qa-input");
const qaAsk = document.getElementById("qa-ask");
const qaAnswer = document.getElementById("qa-answer");

let lastReranked = []; // papers available to summarize / graph
let cy = null;         // Cytoscape instance

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

    // Make the reranked top-5 available for summarization.
    lastReranked = (data.stages.reranked || []).map((h) => ({
      arxiv_id: h.arxiv_id, title: h.title, abstract: h.abstract,
    }));
    synthOut.innerHTML = "";
    synthStatus.textContent = "";
    synthBar.hidden = lastReranked.length === 0;
    graphPanel.hidden = true;
    if (cy) { cy.destroy(); cy = null; }

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

// ---- Phase 4: per-section summaries ----

function paperBlockHtml(p) {
  const blocks = (p.section_summaries || []).map((s) => `
    <div class="sec">
      <div class="sec-name">${escapeHtml(s.section)}</div>
      <div class="sec-text">${escapeHtml(s.summary)}</div>
    </div>`).join("");
  const body = blocks || `<div class="empty">No summary (${escapeHtml(p.note || "n/a")}).</div>`;
  return `
    <article class="paper">
      <div class="paper-head">
        <a href="${arxivUrl(p.arxiv_id)}" target="_blank" rel="noopener">${escapeHtml(p.title || p.arxiv_id)}</a>
        <span class="src">${escapeHtml(p.source_type || "")} · ${escapeHtml(p.arxiv_id)}</span>
      </div>
      <div class="secs">${body}</div>
    </article>`;
}

async function runSummarize() {
  if (lastReranked.length === 0) return;
  summarizeBtn.disabled = true;
  synthStatus.textContent = "Fetching full text + summarizing sections… (first run downloads papers)";
  synthOut.innerHTML = Array.from({ length: lastReranked.length })
    .map(() => `<div class="skeleton tall"></div>`).join("");

  try {
    const res = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ papers: lastReranked }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    synthOut.innerHTML = data.papers.map(paperBlockHtml).join("");
    synthStatus.textContent = `Summarized ${data.papers.length} papers.`;
  } catch (err) {
    synthStatus.textContent = "Error: " + err.message;
    synthOut.innerHTML = "";
  } finally {
    summarizeBtn.disabled = false;
  }
}

summarizeBtn.addEventListener("click", runSummarize);

// ---- Phase 5: knowledge graph ----

const TYPE_COLOR = {
  paper: "#58a6ff", dataset: "#3fb950", metric: "#bc8cff",
  method: "#d29922", model: "#f778ba", task: "#39c5cf",
};

function renderLegend(nodes) {
  const present = [...new Set(nodes.map((n) => n.data.type))];
  graphLegend.innerHTML = present
    .map((t) => `<span class="lg"><i style="background:${TYPE_COLOR[t] || "#888"}"></i>${t}</span>`)
    .join("");
}

function edgeLabel(d) {
  if (d.type === "reports-metric" && d.value) {
    return d.dataset ? `${d.value} · ${d.dataset}` : d.value;
  }
  return d.type;
}

function applySharedFilter() {
  if (!cy) return;
  const on = sharedOnly.checked;
  cy.batch(() => {
    cy.nodes().forEach((n) => {
      const d = n.data();
      const hide = on && d.type !== "paper" && !d.shared;
      n.style("display", hide ? "none" : "element");
    });
  });
}

function renderGraph(data) {
  if (cy) { cy.destroy(); cy = null; }
  renderLegend(data.nodes);

  cy = cytoscape({
    container: document.getElementById("cy"),
    elements: [...data.nodes, ...data.edges],
    style: [
      {
        selector: "node",
        style: {
          "background-color": (n) => TYPE_COLOR[n.data("type")] || "#888",
          label: "data(label)",
          color: "#e6edf3",
          "font-size": 10,
          "text-wrap": "wrap",
          "text-max-width": 110,
          "text-valign": "bottom",
          "text-margin-y": 3,
          width: (n) => (n.data("type") === "paper" ? 34 : 16 + 5 * (n.data("papers") || 1)),
          height: (n) => (n.data("type") === "paper" ? 34 : 16 + 5 * (n.data("papers") || 1)),
        },
      },
      { selector: 'node[type = "paper"]', style: { shape: "round-rectangle", "font-size": 11, "font-weight": "bold" } },
      { selector: "node[?shared]", style: { "border-width": 2, "border-color": "#e6edf3" } },
      {
        selector: "edge",
        style: {
          width: 1.2,
          "line-color": "#3a4452",
          "target-arrow-color": "#3a4452",
          "target-arrow-shape": "triangle",
          "arrow-scale": 0.7,
          "curve-style": "bezier",
          label: (e) => edgeLabel(e.data()),
          "font-size": 8,
          color: "#8b949e",
          "text-rotation": "autorotate",
        },
      },
      { selector: 'edge[type = "reports-metric"]', style: { "line-color": "#bc8cff", "target-arrow-color": "#bc8cff", "line-style": "dashed" } },
      { selector: ".faded", style: { opacity: 0.12 } },
      { selector: ".qa-hit", style: { "border-width": 4, "border-color": "#f0f6fc", "font-weight": "bold", "font-size": 12, "z-index": 99 } },
    ],
    layout: { name: "cose", animate: false, padding: 24, nodeRepulsion: 9000, idealEdgeLength: 95, nodeOverlap: 16 },
  });

  cy.on("tap", "node", (evt) => {
    const d = evt.target.data();
    if (d.type === "paper") {
      graphDetail.textContent = `${d.full || d.label} (${d.id})`;
    } else {
      graphDetail.textContent = `${d.label} · ${d.type} · referenced by ${d.papers} paper(s)`;
    }
    cy.elements().addClass("faded");
    evt.target.removeClass("faded");
    evt.target.neighborhood().removeClass("faded");
  });
  cy.on("tap", (evt) => { if (evt.target === cy) { cy.elements().removeClass("faded"); graphDetail.textContent = ""; } });

  applySharedFilter();
}

async function buildGraph() {
  if (lastReranked.length === 0) return;
  graphBtn.disabled = true;
  graphPanel.hidden = false;
  graphDetail.textContent = "";
  synthStatus.textContent = "Extracting entities + resolving across papers… (first run fetches full text)";

  try {
    const res = await fetch("/api/graph", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ papers: lastReranked }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    renderGraph(data);
    const s = data.stats;
    synthStatus.textContent = `Graph: ${s.papers} papers · ${s.entities} entities (${s.shared} shared) · ${s.edges} edges.`;
  } catch (err) {
    synthStatus.textContent = "Error: " + err.message;
    graphPanel.hidden = true;
  } finally {
    graphBtn.disabled = false;
  }
}

graphBtn.addEventListener("click", buildGraph);
sharedOnly.addEventListener("change", applySharedFilter);

// ---- Load papers by id (test without a search / empty index) ----

function loadManual() {
  const ids = manualIds.value.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean);
  if (ids.length === 0) return;
  lastReranked = ids.map((id) => ({ arxiv_id: id, title: "", abstract: "" }));
  synthOut.innerHTML = "";
  graphPanel.hidden = true;
  if (cy) { cy.destroy(); cy = null; }
  synthBar.hidden = false;
  statusEl.className = "status";
  statusEl.textContent = "";
  synthStatus.textContent =
    `Loaded ${ids.length} paper(s) by id — click “Build knowledge graph” or “Summarize”.`;
}

manualLoad.addEventListener("click", loadManual);

// ---- Graph-grounded Q&A ----

function linkifyArxiv(escaped) {
  // ids look like 2603.02810; wrap them in links to the abstract page
  return escaped.replace(/\b(\d{4}\.\d{4,5})\b/g,
    '<a href="https://arxiv.org/abs/$1" target="_blank" rel="noopener">$1</a>');
}

function renderAnswer(data) {
  const text = linkifyArxiv(escapeHtml(data.answer || ""));
  const cites = (data.cited_papers || [])
    .map((id) => `<span class="qa-chip">${escapeHtml(id)}</span>`).join("");
  const ents = (data.key_entities || [])
    .map((e) => `<span class="qa-chip ent">${escapeHtml(e)}</span>`).join("");
  qaAnswer.innerHTML =
    `<div class="qa-text">${text}</div>` +
    (cites ? `<div class="qa-meta">grounded in ${cites}</div>` : "") +
    (ents ? `<div class="qa-meta">entities ${ents}</div>` : "");
}

function highlightCited(data) {
  if (!cy) return;
  cy.elements().removeClass("faded qa-hit");
  const ids = new Set(data.cited_papers || []);
  const labels = new Set((data.key_entities || []).map((s) => s.toLowerCase()));
  const hits = cy.nodes().filter((n) => {
    const d = n.data();
    return ids.has(d.id) || labels.has((d.label || "").toLowerCase());
  });
  if (hits.length === 0) return;
  cy.elements().addClass("faded");
  hits.removeClass("faded").addClass("qa-hit");
  hits.neighborhood().removeClass("faded");
}

async function askQuestion() {
  const q = qaInput.value.trim();
  if (!q) return;
  if (lastReranked.length === 0) {
    qaAnswer.innerHTML = `<div class="qa-error">Load or search papers first.</div>`;
    return;
  }
  qaAsk.disabled = true;
  qaAnswer.innerHTML = `<div class="qa-thinking">Thinking… referencing the graph + summaries</div>`;
  try {
    const res = await fetch("/api/graph_qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, papers: lastReranked }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    renderAnswer(data);
    highlightCited(data);
  } catch (err) {
    qaAnswer.innerHTML = `<div class="qa-error">Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    qaAsk.disabled = false;
  }
}

qaAsk.addEventListener("click", askQuestion);
qaInput.addEventListener("keydown", (e) => { if (e.key === "Enter") askQuestion(); });
