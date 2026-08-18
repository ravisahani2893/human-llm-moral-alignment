// Shared across every page: model list, formatting helpers, the modal,
// and the Active Jobs dashboard (rendered on the home page, but the data
// helpers live here since normalizeAgentRun etc. need MODELS).

const API = "";

let MODELS = [];

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function fmtValence(v, band) {
  if (v === null || v === undefined) return "—";
  const cls = v > 0 ? "val-pos" : v < 0 ? "val-neg" : "";
  const bandHtml = band ? `<span class="val-band">${escapeHtml(band)}</span>` : "";
  return `<span class="${cls}">${v.toFixed(2)}</span>${bandHtml}`;
}

// Mirrors app/interpret.py's valence_label() so client-computed rows can
// show the same band labels as server-computed ones.
function valenceBand(v) {
  if (v === null || v === undefined) return null;
  if (v === 0) return "Neutral";
  const edges = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
  const labels = [
    "Extremely negative", "Strongly negative", "Moderately negative", "Slightly negative",
    "Slightly positive", "Moderately positive", "Strongly positive", "Extremely positive",
  ];
  for (let i = 0; i < edges.length - 1; i++) {
    if (v > edges[i] && v <= edges[i + 1]) return labels[i];
  }
  return v < -1 ? "Extremely negative" : "Extremely positive";
}

// Mirrors app/interpret.py's agreement_label()/error_label() so
// client-rendered metric tables can show the same band labels as
// server-computed ones, without a schema change to every MCP response.
function agreementLabel(v) {
  if (v === null || v === undefined) return null;
  const magnitude = Math.min(Math.abs(v), 1.0);
  const bands = [[0.20, "Very weak"], [0.40, "Weak"], [0.60, "Moderate"], [0.80, "Strong"], [1.01, "Very strong"]];
  for (const [cutoff, label] of bands) {
    if (magnitude < cutoff) return v < 0 ? `${label} (inverse)` : label;
  }
  return v < 0 ? "Very strong (inverse)" : "Very strong";
}

function errorLabel(v) {
  if (v === null || v === undefined) return null;
  if (v < 0.3) return "Low error";
  if (v < 0.5) return "Moderate error";
  return "High error";
}

function fmtMetric(v, labelFn, decimals = 3) {
  if (v === null || v === undefined) return "—";
  const label = labelFn(v);
  return `${v.toFixed(decimals)}${label ? ` <span class="val-band">${escapeHtml(label)}</span>` : ""}`;
}

function fmtFactors(factors) {
  if (!factors || factors.length === 0) return "";
  return factors.map((f) => `<span class="factor-tag">${escapeHtml(f)}</span>`).join("");
}

function buildModelSelect(containerId, { includeAll = true } = {}) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = "";

  if (includeAll) {
    const allChip = document.createElement("label");
    allChip.className = "model-chip all";
    allChip.innerHTML = `<input type="checkbox" data-role="all" /> All models`;
    el.appendChild(allChip);
  }

  MODELS.forEach((m) => {
    const chip = document.createElement("label");
    chip.className = "model-chip";
    chip.innerHTML = `<input type="checkbox" value="${m.id}" checked /> ${m.label}`;
    el.appendChild(chip);
  });

  if (includeAll) {
    const allInput = el.querySelector('input[data-role="all"]');
    const modelInputs = () => Array.from(el.querySelectorAll('input[type="checkbox"]:not([data-role="all"])'));

    const syncAll = () => {
      allInput.checked = modelInputs().every((i) => i.checked);
    };
    allInput.addEventListener("change", () => {
      modelInputs().forEach((i) => (i.checked = allInput.checked));
    });
    modelInputs().forEach((i) => i.addEventListener("change", syncAll));
    syncAll();
  }
}

function getSelectedModels(containerId) {
  const el = document.getElementById(containerId);
  return Array.from(el.querySelectorAll('input[type="checkbox"]:not([data-role="all"])'))
    .filter((i) => i.checked)
    .map((i) => i.value);
}

async function loadModels() {
  const res = await fetch(`${API}/api/models`);
  MODELS = await res.json();
}

// v1/v2 are early, superseded prompt iterations kept in the backend for
// historical/reproducibility reasons (existing exports still reference
// them), but aren't meaningful choices for a user starting a new run —
// hidden from every prompt-version dropdown in the UI, not removed from
// the backend itself.
const HIDDEN_PROMPT_VERSIONS = new Set(["v1", "v2"]);

async function loadPromptVersions(selectId) {
  const res = await fetch(`${API}/api/prompt-versions`);
  const versions = (await res.json()).filter((v) => !HIDDEN_PROMPT_VERSIONS.has(v));
  const select = document.getElementById(selectId);
  if (!select) return;
  select.innerHTML = versions.map((v) => `<option value="${v}"${v === "current" ? " selected" : ""}>${escapeHtml(v)}</option>`).join("");
}

function setStatus(id, text, kind) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = "status" + (kind ? ` ${kind}` : "");
}

// ---- Modal ----

function openModal(titleHtml, bodyHtml) {
  document.getElementById("modal-body").innerHTML = titleHtml + bodyHtml;
  document.getElementById("details-modal").hidden = false;
}

function closeModal() {
  document.getElementById("details-modal").hidden = true;
}

function modelReasoningBlockHtml(label, action) {
  return `
    <div class="modal-model-block">
      <h4>${escapeHtml(label)}</h4>
      <div class="modal-axis">
        <span class="axis-label">Action:</span>
        <span class="axis-score">${fmtValence(action.action_valence, action.action_band)}</span>
        <div>${escapeHtml(action.action_reasoning)}</div>
        <div>${fmtFactors(action.action_factors)}</div>
      </div>
      <div class="modal-axis">
        <span class="axis-label">Consequence:</span>
        <span class="axis-score">${fmtValence(action.consequence_valence, action.consequence_band)}</span>
        <div>${escapeHtml(action.consequence_reasoning)}</div>
        <div>${fmtFactors(action.consequence_factors)}</div>
      </div>
    </div>`;
}

function initModal() {
  const closeBtn = document.getElementById("modal-close-btn");
  const backdrop = document.getElementById("details-modal");
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (backdrop) {
    backdrop.addEventListener("click", (e) => {
      if (e.target.id === "details-modal") closeModal();
    });
  }
}

function buildBatchTableHead(tableId, models) {
  const tr = document.querySelector(`#${tableId} thead tr`);
  tr.innerHTML = "<th>ID</th><th>Scenario</th><th>Human Action</th><th>Human Consequence</th>";
  models.forEach((m) => {
    const label = MODELS.find((x) => x.id === m)?.label || m;
    tr.innerHTML += `<th>${label} Action</th><th>${label} Consequence</th>`;
  });
  tr.innerHTML += "<th>Reasoning</th>";
}

function renderBatchRows(tableId, rows, models) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.innerHTML = "";
  rows.slice(0, 200).forEach((row) => {
    const tr = document.createElement("tr");
    let html = `<td>${row.ID}</td><td class="scenario-cell">${escapeHtml(row.Scenario)}</td>`;
    html += `<td>${fmtValence(row.Human_Action)}</td><td>${fmtValence(row.Human_Consequence)}</td>`;
    models.forEach((m) => {
      const label = MODELS.find((x) => x.id === m)?.label || m;
      html += `<td>${fmtValence(row[`${label}_Action`], row[`${label}_Action_Band`])}</td>`;
      html += `<td>${fmtValence(row[`${label}_Consequence`], row[`${label}_Consequence_Band`])}</td>`;
    });
    html += `<td><button class="details-btn" data-row-id="${row.ID}">View</button></td>`;
    tr.innerHTML = html;
    tbody.appendChild(tr);
  });

  tbody.querySelectorAll(".details-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const row = rows.find((r) => String(r.ID) === btn.dataset.rowId);
      if (!row) return;
      const title = `<div class="modal-scenario">${escapeHtml(row.Scenario)}</div>`;
      const body = models
        .map((m) => {
          const label = MODELS.find((x) => x.id === m)?.label || m;
          return modelReasoningBlockHtml(label, {
            action_valence: row[`${label}_Action`],
            action_band: row[`${label}_Action_Band`],
            action_reasoning: row[`${label}_Action_Reasoning`],
            action_factors: (row[`${label}_Action_Factors`] || "").split("; ").filter(Boolean),
            consequence_valence: row[`${label}_Consequence`],
            consequence_band: row[`${label}_Consequence_Band`],
            consequence_reasoning: row[`${label}_Consequence_Reasoning`],
            consequence_factors: (row[`${label}_Consequence_Factors`] || "").split("; ").filter(Boolean),
          });
        })
        .join("");
      openModal(title, body);
    });
  });
}

// ---- Minimal, safe Markdown renderer (escapes HTML first, then applies
// a small subset of Markdown on top of the escaped text) ----

function renderMarkdown(raw) {
  const escaped = escapeHtml(raw ?? "");
  const lines = escaped.split("\n");
  const htmlParts = [];
  let listBuffer = [];
  let listType = null;

  const flushList = () => {
    if (listBuffer.length) {
      htmlParts.push(`<${listType}>${listBuffer.map((li) => `<li>${li}</li>`).join("")}</${listType}>`);
      listBuffer = [];
      listType = null;
    }
  };

  const inline = (text) =>
    text
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+?)`/g, "<code>$1</code>")
      .replace(/(^|[^*])\*(?!\*)([^*]+?)\*(?!\*)/g, "$1<em>$2</em>");

  let paragraphBuffer = [];
  const flushParagraph = () => {
    if (paragraphBuffer.length) {
      htmlParts.push(`<p>${inline(paragraphBuffer.join(" "))}</p>`);
      paragraphBuffer = [];
    }
  };

  lines.forEach((line) => {
    const headerMatch = line.match(/^(#{2,4})\s+(.*)$/);
    const ulMatch = line.match(/^\s*[*-]\s+(.*)$/);
    const olMatch = line.match(/^\s*\d+\.\s+(.*)$/);

    if (headerMatch) {
      flushParagraph();
      flushList();
      const level = headerMatch[1].length;
      htmlParts.push(`<h${level}>${inline(headerMatch[2])}</h${level}>`);
    } else if (ulMatch) {
      flushParagraph();
      if (listType !== "ul") flushList();
      listType = "ul";
      listBuffer.push(inline(ulMatch[1]));
    } else if (olMatch) {
      flushParagraph();
      if (listType !== "ol") flushList();
      listType = "ol";
      listBuffer.push(inline(olMatch[1]));
    } else if (line.trim() === "") {
      flushParagraph();
      flushList();
    } else {
      paragraphBuffer.push(line.trim());
    }
  });
  flushParagraph();
  flushList();

  return htmlParts.join("\n");
}

// ---- Active Jobs dashboard ----

function fmtRelativeTime(unixSeconds) {
  const diffSec = Math.round(Date.now() / 1000 - unixSeconds);
  if (diffSec < 5) return "just now";
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.round(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.round(diffSec / 3600)}h ago`;
  return `${Math.round(diffSec / 86400)}d ago`;
}

function modelLabel(id) {
  return MODELS.find((m) => m.id === id)?.label || id;
}

// A "running" job whose status file hasn't been touched in a while is
// almost always orphaned (its process was killed/restarted mid-run without
// a clean shutdown) rather than genuinely still working — this has happened
// repeatedly in this project (server restarts kill background export/job
// threads). Surface that distinction instead of trusting "running" blindly.
const STALL_THRESHOLD_SECONDS = 90;

function effectiveStatus(status, updatedAt) {
  if (status !== "running") return status;
  if (!updatedAt) return status;
  const staleFor = Date.now() / 1000 - updatedAt;
  return staleFor > STALL_THRESHOLD_SECONDS ? "stalled" : "running";
}

function normalizeExportJob(j) {
  return {
    type: "Export",
    models: modelLabel(j.model),
    prompt: j.prompt_version || "current",
    status: effectiveStatus(j.status, j.updated_at),
    progress: j.total ? `${j.completed} / ${j.total}` : (j.status === "running" ? "starting…" : "—"),
    created_at: j.created_at,
  };
}

function normalizeBatchJob(j) {
  const totalUnits = j.total * (j.models || []).length;
  const doneUnits = Object.values(j.completed_per_model || {}).reduce((a, b) => a + b, 0);
  return {
    type: "Batch",
    models: (j.models || []).map(modelLabel).join(", "),
    prompt: j.prompt_version || "current",
    status: effectiveStatus(j.status, j.updated_at),
    progress: totalUnits ? `${doneUnits} / ${totalUnits}` : "—",
    created_at: j.created_at,
  };
}

function normalizeAgentRun(r) {
  return {
    type: "Agent",
    models: r.instruction ? (r.instruction.length > 60 ? r.instruction.slice(0, 60) + "…" : r.instruction) : "—",
    prompt: "—",
    status: effectiveStatus(r.status, r.updated_at),
    progress: r.status === "running" ? `${(r.log || []).length} step(s)` : "—",
    created_at: r.created_at,
  };
}

async function loadActiveJobs() {
  const table = document.getElementById("active-jobs-table");
  if (!table) return;
  try {
    const [exportsRes, jobsRes, agentRes] = await Promise.all([
      fetch(`${API}/api/exports`),
      fetch(`${API}/api/jobs`),
      fetch(`${API}/api/agent/runs`),
    ]);
    const [exportsData, jobsData, agentData] = await Promise.all([exportsRes.json(), jobsRes.json(), agentRes.json()]);

    const rows = [
      ...(exportsData.exports || []).map(normalizeExportJob),
      ...(jobsData.jobs || []).map(normalizeBatchJob),
      ...(agentData.runs || []).map(normalizeAgentRun),
    ].sort((a, b) => b.created_at - a.created_at);

    const tbody = table.querySelector("tbody");
    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="hint">No jobs yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows
      .map(
        (r) => `<tr>
          <td>${escapeHtml(r.type)}</td>
          <td>${escapeHtml(r.models)}</td>
          <td>${escapeHtml(r.prompt)}</td>
          <td><span class="job-badge ${escapeHtml(r.status)}">${escapeHtml(r.status)}</span></td>
          <td>${escapeHtml(r.progress)}</td>
          <td>${fmtRelativeTime(r.created_at)}</td>
        </tr>`
      )
      .join("");
  } catch {
    // dashboard is a supplement to the per-card views; fail quietly on a transient error
  }
}

function startActiveJobsPolling() {
  if (!document.getElementById("active-jobs-table")) return;
  loadActiveJobs();
  setInterval(loadActiveJobs, 3000);
}
