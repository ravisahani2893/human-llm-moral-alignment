let currentExportJobId = null;
let exportPollTimer = null;

async function pollExportJob(jobId) {
  const res = await fetch(`${API}/api/exports/${jobId}`);
  const job = await res.json();

  const pct = job.total ? Math.round((job.completed / job.total) * 100) : 0;
  document.getElementById("export-progress-fill").style.width = `${pct}%`;
  document.getElementById("export-progress-label").textContent =
    job.total ? `${job.completed} / ${job.total} scenarios done (${pct}%)` : "Starting…";

  if (job.status === "running") {
    exportPollTimer = setTimeout(() => pollExportJob(jobId), 2000);
  } else if (job.status === "completed") {
    setStatus("export-status", "Completed.", "ok");
    document.getElementById("run-export-btn").disabled = false;
    document.getElementById("download-export-csv-btn").disabled = false;
  } else {
    setStatus("export-status", job.error || "Export failed.", "error");
    document.getElementById("run-export-btn").disabled = false;
  }
}

async function runExport() {
  const model = document.getElementById("export-model-select").value;
  const promptVersion = document.getElementById("export-prompt-version-select").value;
  if (!model) {
    setStatus("export-status", "Select a model.", "error");
    return;
  }

  clearTimeout(exportPollTimer);
  document.getElementById("run-export-btn").disabled = true;
  document.getElementById("download-export-csv-btn").disabled = true;
  document.getElementById("export-progress-wrap").hidden = false;
  document.getElementById("export-progress-fill").style.width = "0%";
  setStatus("export-status", "Starting export…");

  try {
    const res = await fetch(`${API}/api/exports`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, prompt_version: promptVersion }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }
    const job = await res.json();
    currentExportJobId = job.id;
    setStatus("export-status", "Running…");
    pollExportJob(currentExportJobId);
  } catch (e) {
    setStatus("export-status", e.message, "error");
    document.getElementById("run-export-btn").disabled = false;
  }
}

function downloadExportCsv() {
  if (!currentExportJobId) return;
  window.location.href = `${API}/api/exports/${currentExportJobId}/csv`;
}

loadModels().then(() => {
  const exportSelect = document.getElementById("export-model-select");
  exportSelect.innerHTML = MODELS.map((m) => `<option value="${m.id}">${escapeHtml(m.label)}</option>`).join("");
});
loadPromptVersions("export-prompt-version-select");
document.getElementById("run-export-btn").addEventListener("click", runExport);
document.getElementById("download-export-csv-btn").addEventListener("click", downloadExportCsv);
