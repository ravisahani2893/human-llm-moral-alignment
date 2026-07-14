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

function buildBatchTableHead(models) {
  const tr = document.querySelector("#batch-results-table thead tr");
  tr.innerHTML = "<th>ID</th><th>Scenario</th><th>Human Action</th><th>Human Consequence</th>";
  models.forEach((m) => {
    const label = MODELS.find((x) => x.id === m)?.label || m;
    tr.innerHTML += `<th>${label} Action</th><th>${label} Consequence</th>`;
  });
  tr.innerHTML += "<th>Reasoning</th>";
}

function renderBatchRows(rows, models) {
  const tbody = document.querySelector("#batch-results-table tbody");
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
  renderBatchRows(resultsData.rows, models);

  if (job.status === "running") {
    pollTimer = setTimeout(() => pollJob(jobId, models), 1500);
  } else if (job.status === "completed") {
    setStatus("batch-status", "Completed.", "ok");
    document.getElementById("run-batch-btn").disabled = false;
    document.getElementById("download-csv-btn").disabled = false;
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
  buildBatchTableHead(models);
  document.querySelector("#batch-results-table tbody").innerHTML = "";

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

loadModels();
loadDatasetCount();
