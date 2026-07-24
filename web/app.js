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

// Mirrors app/interpret.py's valence_label() so client-computed rows (e.g.
// single-scenario agent mode, which returns raw MCP tool results with no
// band attached) can still show the same band labels as server-computed ones.
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

function fmtFactors(factors) {
  if (!factors || factors.length === 0) return "";
  return factors.map((f) => `<span class="factor-tag">${escapeHtml(f)}</span>`).join("");
}

function buildModelSelect(containerId, { includeAll = true } = {}) {
  const el = document.getElementById(containerId);
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
  buildModelSelect("single-model-select");
  buildModelSelect("batch-model-select");
  buildModelSelect("agent-model-select");
}

async function loadDatasetCount() {
  const res = await fetch(`${API}/api/dataset/count`);
  const data = await res.json();
  document.getElementById("dataset-count").textContent = data.count;
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

// ---- Single scenario ----

async function runSingle() {
  const scenario = document.getElementById("scenario-input").value.trim();
  const models = getSelectedModels("single-model-select");
  const btn = document.getElementById("run-single-btn");
  const tbody = document.querySelector("#single-results-table tbody");

  if (!scenario) {
    setStatus("single-status", "Please enter a scenario.", "error");
    return;
  }
  if (models.length === 0) {
    setStatus("single-status", "Select at least one model.", "error");
    return;
  }

  btn.disabled = true;
  tbody.innerHTML = "";
  setStatus("single-status", `Evaluating with ${models.length} model(s)…`);

  try {
    const res = await fetch(`${API}/api/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario, models }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }
    const data = await res.json();

    data.results.forEach((r) => {
      const tr = document.createElement("tr");
      if (r.error) {
        tr.innerHTML = `
          <td>${escapeHtml(r.label)}</td>
          <td colspan="2" class="val-err">Error</td>
          <td class="val-err">${escapeHtml(r.error)}</td>`;
      } else {
        tr.innerHTML = `
          <td>${escapeHtml(r.label)}</td>
          <td>${fmtValence(r.action_valence, r.action_band)}</td>
          <td>${fmtValence(r.consequence_valence, r.consequence_band)}</td>
          <td class="reasoning-cell">
            <span class="axis-label">Action:</span> ${escapeHtml(r.action_reasoning)}
            <div>${fmtFactors(r.action_factors)}</div>
            <span class="axis-label">Consequence:</span> ${escapeHtml(r.consequence_reasoning)}
            <div>${fmtFactors(r.consequence_factors)}</div>
          </td>`;
      }
      tbody.appendChild(tr);
    });

    setStatus("single-status", "Done.", "ok");
  } catch (e) {
    setStatus("single-status", e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

// ---- Batch / full dataset ----

let currentJobId = null;
let pollTimer = null;

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

// ---- CCC (Concordance Correlation Coefficient) metrics, paper-style ----

let PAPER_REFERENCE = null;

async function loadPaperReference() {
  const res = await fetch(`${API}/api/paper-reference`);
  PAPER_REFERENCE = await res.json();
}

function fmtCcc(v) {
  return v === null || v === undefined ? "—" : v.toFixed(3);
}

function cccCellClass(v) {
  if (v === null || v === undefined) return "";
  if (v >= 0.5) return "ccc-cell-good";
  if (v < 0.2) return "ccc-cell-bad";
  return "ccc-cell-mid";
}

function renderCccTables(containerId, report, models, options = {}) {
  const container = document.getElementById(containerId);
  if (!report || !report.models) {
    container.innerHTML = "";
    return;
  }

  const {
    title = "Concordance Correlation Coefficient (CCC) vs Human Gold Standard",
    subtitle = "Lin's CCC — the same metric the source paper reports in Table II. Unlike Pearson r, it penalizes a systematic mean/scale shift, not just weak correlation.",
    showPaperReference = true,
    showCrossModel = true,
  } = options;

  const modelLabels = models.map((m) => MODELS.find((x) => x.id === m)?.label || m);
  const refHeaders = showPaperReference
    ? "<th>Paper: Human Pairwise</th><th>Paper: Human vs EWE Gold Standard</th>"
    : "";

  let html = `<div class="ccc-block">
    <h3>${escapeHtml(title)}</h3>
    <p class="hint">${subtitle}</p>
    <table class="ccc-table">
      <thead><tr><th>Valence Type</th>${modelLabels.map((l) => `<th>${escapeHtml(l)}</th>`).join("")}${refHeaders}</tr></thead>
      <tbody>`;

  const rowLabels = { action: "Action", consequence: "Consequence", combined: "Combined*" };
  ["action", "consequence", "combined"].forEach((axis) => {
    const rowClass = axis === "combined" ? ' class="ccc-reference-row"' : "";
    html += `<tr${rowClass}><td>${rowLabels[axis]}</td>`;
    modelLabels.forEach((ml) => {
      const m = report.models[ml];
      const ccc = m ? m[axis].ccc : null;
      html += `<td class="${cccCellClass(ccc)}">${fmtCcc(ccc)}</td>`;
    });
    if (showPaperReference) {
      const ref = PAPER_REFERENCE ? PAPER_REFERENCE[axis] : null;
      html += `<td>${ref ? ref.pairwise_human.toFixed(3) : "—"}</td>`;
      html += `<td>${ref ? ref.human_vs_ewe_gold_standard.toFixed(3) : "—"}</td>`;
    }
    html += `</tr>`;
  });

  html += `</tbody></table>
    <p class="hint">*Combined: CCC computed by stacking action and consequence values together into one series${showPaperReference ? " — the paper does not report this figure" : ""}.</p>
  </div>`;

  if (showCrossModel && modelLabels.length > 1 && report.cross_model_agreement) {
    const pairs = Object.keys(report.cross_model_agreement.action || {});
    if (pairs.length) {
      html += `<div class="ccc-block">
        <h3>Cross-Model CCC</h3>
        <p class="hint">Agreement between models themselves, not vs. human — the model analog of the paper's own pairwise human-annotator CCC (0.260 action / 0.356 consequence).</p>
        <table class="ccc-table">
          <thead><tr><th>Model Pair</th><th>Action CCC</th><th>Consequence CCC</th></tr></thead>
          <tbody>`;
      pairs.forEach((pair) => {
        const a = report.cross_model_agreement.action[pair];
        const c = report.cross_model_agreement.consequence[pair];
        html += `<tr><td>${escapeHtml(pair)}</td><td class="${cccCellClass(a)}">${fmtCcc(a)}</td><td class="${cccCellClass(c)}">${fmtCcc(c)}</td></tr>`;
      });
      html += `</tbody></table></div>`;
    }
  }

  container.innerHTML = html;
}

async function fetchAndRenderCcc(metricsUrl, containerId, models) {
  try {
    const res = await fetch(metricsUrl);
    if (!res.ok) return;
    const report = await res.json();
    renderCccTables(containerId, report, models);
  } catch {
    // metrics are a supplement to the main result; fail quietly
  }
}

async function pollJob(jobId, models) {
  const res = await fetch(`${API}/api/jobs/${jobId}`);
  const job = await res.json();

  const totalUnits = job.total * models.length;
  const doneUnits = Object.values(job.completed_per_model).reduce((a, b) => a + b, 0);
  const pct = totalUnits ? Math.round((doneUnits / totalUnits) * 100) : 0;

  document.getElementById("progress-fill").style.width = `${pct}%`;
  const errTotal = Object.values(job.errors_per_model).reduce((a, b) => a + b, 0);
  document.getElementById("progress-label").textContent =
    `${doneUnits} / ${totalUnits} evaluations complete (${pct}%)` +
    (errTotal ? ` · ${errTotal} error(s)` : "");

  const resultsRes = await fetch(`${API}/api/jobs/${jobId}/results`);
  const resultsData = await resultsRes.json();
  renderBatchRows("batch-results-table", resultsData.rows, models);

  if (job.status === "running") {
    pollTimer = setTimeout(() => pollJob(jobId, models), 1500);
  } else if (job.status === "completed") {
    setStatus("batch-status", "Completed.", "ok");
    document.getElementById("run-batch-btn").disabled = false;
    document.getElementById("download-csv-btn").disabled = false;
    fetchAndRenderCcc(`${API}/api/jobs/${jobId}/metrics`, "batch-ccc-wrap", models);
  } else {
    setStatus("batch-status", job.error || "Job failed.", "error");
    document.getElementById("run-batch-btn").disabled = false;
  }
}

async function runBatch() {
  const models = getSelectedModels("batch-model-select");
  const sampleInput = document.getElementById("sample-size-input").value.trim();
  const sampleSize = sampleInput ? parseInt(sampleInput, 10) : null;

  if (models.length === 0) {
    setStatus("batch-status", "Select at least one model.", "error");
    return;
  }

  clearTimeout(pollTimer);
  document.getElementById("run-batch-btn").disabled = true;
  document.getElementById("download-csv-btn").disabled = true;
  document.getElementById("progress-wrap").hidden = false;
  document.getElementById("progress-fill").style.width = "0%";
  setStatus("batch-status", "Starting job…");
  buildBatchTableHead("batch-results-table", models);
  document.querySelector("#batch-results-table tbody").innerHTML = "";
  document.getElementById("batch-ccc-wrap").innerHTML = "";

  try {
    const res = await fetch(`${API}/api/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models, sample_size: sampleSize }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }
    const job = await res.json();
    currentJobId = job.id;
    setStatus("batch-status", "Running…");
    pollJob(currentJobId, models);
  } catch (e) {
    setStatus("batch-status", e.message, "error");
    document.getElementById("run-batch-btn").disabled = false;
  }
}

function downloadCsv() {
  if (!currentJobId) return;
  window.location.href = `${API}/api/jobs/${currentJobId}/csv`;
}

document.getElementById("run-single-btn").addEventListener("click", runSingle);
document.getElementById("run-batch-btn").addEventListener("click", runBatch);
document.getElementById("download-csv-btn").addEventListener("click", downloadCsv);
document.getElementById("modal-close-btn").addEventListener("click", closeModal);
document.getElementById("details-modal").addEventListener("click", (e) => {
  if (e.target.id === "details-modal") closeModal();
});

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

// ---- Divergence analyst agent ----

let currentAgentRunId = null;
let agentPollTimer = null;

function syncAgentModeFields() {
  const mode = document.querySelector('input[name="agent-mode"]:checked').value;
  document.getElementById("agent-scenario-field").hidden = mode !== "scenario";
  document.getElementById("agent-sample-field").hidden = mode !== "random";
  document.getElementById("agent-all-warning").hidden = mode !== "all";
}

function renderAgentLog(log) {
  const el = document.getElementById("agent-log");
  const wrap = document.getElementById("agent-log-wrap");
  if (!log || log.length === 0) return;
  wrap.hidden = false;

  const compactArgs = (args) => {
    try {
      return JSON.stringify(args);
    } catch {
      return "";
    }
  };

  const resultPreview = (text) => {
    if (!text) return "";
    const trimmed = text.length > 220 ? text.slice(0, 220) + "…" : text;
    return trimmed.replace(/\s+/g, " ");
  };

  const lines = log.map((entry) => {
    switch (entry.type) {
      case "task":
        return `<div class="agent-log-line"><span class="log-task">▶ Task:</span> ${escapeHtml(entry.text || "")}</div>`;
      case "tool_call":
        return `<div class="agent-log-line"><span class="log-tool">→ ${escapeHtml(entry.name)}</span>(${escapeHtml(compactArgs(entry.args))})</div>`;
      case "tool_result": {
        const isError = (entry.result || "").includes('"error"');
        const cls = isError ? "log-error" : "log-result";
        return `<div class="agent-log-line"><span class="${cls}">← ${escapeHtml(entry.name)}:</span> ${escapeHtml(resultPreview(entry.result))}</div>`;
      }
      case "job_progress":
        return `<div class="agent-log-line"><span class="log-progress">⏳ job ${escapeHtml(entry.job_id)}:</span> ${escapeHtml(entry.status)} ${escapeHtml(compactArgs(entry.completed_per_model))}</div>`;
      case "final_report":
        return `<div class="agent-log-line"><span class="log-final">✓ Final report produced.</span></div>`;
      case "max_turns_reached":
        return `<div class="agent-log-line"><span class="log-error">⚠ Stopped: reached max tool-call turns without a final report.</span></div>`;
      default:
        return "";
    }
  });

  el.innerHTML = lines.join("");
  el.scrollTop = el.scrollHeight;
}

function renderScenarioModeTable(run, models) {
  const row = {
    ID: run.human_reference ? run.human_reference.ID : "—",
    Scenario: run.scenario,
    Human_Action: run.human_reference ? run.human_reference.Human_Action : null,
    Human_Consequence: run.human_reference ? run.human_reference.Human_Consequence : null,
  };

  models.forEach((m) => {
    const label = MODELS.find((x) => x.id === m)?.label || m;
    const evalResult = run.scenario_evaluations.find((e) => e.model === m);
    row[`${label}_Action`] = evalResult ? evalResult.action_valence : null;
    row[`${label}_Action_Band`] = evalResult ? valenceBand(evalResult.action_valence) : null;
    row[`${label}_Action_Reasoning`] = evalResult ? evalResult.action_reasoning : "";
    row[`${label}_Action_Factors`] = evalResult ? (evalResult.action_factors || []).join("; ") : "";
    row[`${label}_Consequence`] = evalResult ? evalResult.consequence_valence : null;
    row[`${label}_Consequence_Band`] = evalResult ? valenceBand(evalResult.consequence_valence) : null;
    row[`${label}_Consequence_Reasoning`] = evalResult ? evalResult.consequence_reasoning : "";
    row[`${label}_Consequence_Factors`] = evalResult ? (evalResult.consequence_factors || []).join("; ") : "";
  });

  buildBatchTableHead("agent-results-table", models);
  renderBatchRows("agent-results-table", [row], models);

  const cccWrap = document.getElementById("agent-ccc-wrap");
  if (run.human_reference) {
    cccWrap.innerHTML = `<p class="hint">This scenario matches dataset entry ID ${run.human_reference.ID} — the Human Action/Consequence columns above are the real gold-standard labels. Correlation-based metrics (CCC, Pearson) need multiple scenarios to be statistically meaningful, so they're not shown for a single row; click "View" in the table for the full reasoning comparison.</p>`;
  } else {
    cccWrap.innerHTML = `<p class="hint">This scenario doesn't match any entry in the labeled dataset, so there's no human gold-standard label to show — the table above compares the models' outputs to each other only.</p>`;
  }
}

async function pollAgentRun(runId, models) {
  const res = await fetch(`${API}/api/agent/runs/${runId}`);
  const run = await res.json();

  renderAgentLog(run.log);

  if (run.status === "running") {
    setStatus("agent-status", "Agent is working — this can take a while, especially for larger samples…");
    agentPollTimer = setTimeout(() => pollAgentRun(runId, models), 3000);
    return;
  }

  document.getElementById("run-agent-btn").disabled = false;

  if (run.status === "completed") {
    setStatus("agent-status", "Done.", "ok");
    const reportEl = document.getElementById("agent-report");
    reportEl.hidden = false;
    reportEl.innerHTML = renderMarkdown(run.report || "(No report text returned.)");

    if (run.csv_path) {
      const resultsRes = await fetch(`${API}/api/agent/runs/${runId}/results`);
      const resultsData = await resultsRes.json();
      if (resultsData.rows && resultsData.rows.length) {
        buildBatchTableHead("agent-results-table", models);
        renderBatchRows("agent-results-table", resultsData.rows, models);
      }
      document.getElementById("download-agent-csv-btn").disabled = false;
      fetchAndRenderCcc(`${API}/api/agent/runs/${runId}/metrics`, "agent-ccc-wrap", models);
    } else if (run.mode === "scenario" && run.scenario_evaluations && run.scenario_evaluations.length) {
      renderScenarioModeTable(run, models);
    }
  } else {
    setStatus("agent-status", run.error || "Agent run failed.", "error");
  }
}

async function runAgent() {
  const mode = document.querySelector('input[name="agent-mode"]:checked').value;
  const models = getSelectedModels("agent-model-select");
  const scenario = document.getElementById("agent-scenario-input").value.trim();
  const sampleSize = parseInt(document.getElementById("agent-sample-size-input").value, 10);

  if (models.length === 0) {
    setStatus("agent-status", "Select at least one model.", "error");
    return;
  }
  if (mode === "scenario" && !scenario) {
    setStatus("agent-status", "Please enter a scenario.", "error");
    return;
  }
  if (mode === "random" && (!sampleSize || sampleSize <= 0)) {
    setStatus("agent-status", "Enter a positive sample size.", "error");
    return;
  }

  clearTimeout(agentPollTimer);
  document.getElementById("run-agent-btn").disabled = true;
  document.getElementById("download-agent-csv-btn").disabled = true;
  document.getElementById("agent-report").hidden = true;
  document.getElementById("agent-log-wrap").hidden = true;
  document.getElementById("agent-log").innerHTML = "";
  document.querySelector("#agent-results-table tbody").innerHTML = "";
  document.querySelector("#agent-results-table thead tr").innerHTML = "";
  document.getElementById("agent-ccc-wrap").innerHTML = "";
  setStatus("agent-status", "Starting agent…");

  const body = { mode, models };
  if (mode === "scenario") body.scenario = scenario;
  if (mode === "random") body.sample_size = sampleSize;

  try {
    const res = await fetch(`${API}/api/agent/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }
    const run = await res.json();
    currentAgentRunId = run.id;
    pollAgentRun(currentAgentRunId, models);
  } catch (e) {
    setStatus("agent-status", e.message, "error");
    document.getElementById("run-agent-btn").disabled = false;
  }
}

function downloadAgentCsv() {
  if (!currentAgentRunId) return;
  window.location.href = `${API}/api/agent/runs/${currentAgentRunId}/csv`;
}

document.querySelectorAll('input[name="agent-mode"]').forEach((el) => el.addEventListener("change", syncAgentModeFields));
document.getElementById("run-agent-btn").addEventListener("click", runAgent);
document.getElementById("download-agent-csv-btn").addEventListener("click", downloadAgentCsv);
syncAgentModeFields();

loadModels();
loadDatasetCount();
loadPaperReference();
