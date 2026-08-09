let currentJobId = null;
let pollTimer = null;

async function loadDatasetCount() {
  const res = await fetch(`${API}/api/dataset/count`);
  const data = await res.json();
  document.getElementById("dataset-count").textContent = data.count;
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

loadModels().then(() => buildModelSelect("batch-model-select"));
loadDatasetCount();
initModal();
document.getElementById("run-batch-btn").addEventListener("click", runBatch);
document.getElementById("download-csv-btn").addEventListener("click", downloadCsv);
