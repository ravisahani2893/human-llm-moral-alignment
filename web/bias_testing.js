// Bias Testing — demographic variant robustness. Runs one model over
// every variant of a counterfactual perturbation dataset (data/variants/),
// then reports a paired Wilcoxon signed-rank test per variant pair, per
// axis. Deliberately separate from Human-LLM Alignment and Cross-Model
// Agreement — never reads or displays human annotation values.

let currentBiasEvalJobId = null;
let biasEvalPollTimer = null;

// ETHNICITY is hidden from the UI dropdown (scope narrowed to gender bias
// testing for this dissertation), but left in the backend/data folder —
// not deleted, so it can still be run directly via the agent or API if
// needed later.
const HIDDEN_BIAS_DATASETS = new Set(["ETHNICITY"]);

async function loadBiasDatasets() {
  const res = await fetch(`${API}/api/bias-datasets`);
  const datasets = (await res.json()).filter((d) => !HIDDEN_BIAS_DATASETS.has(d));
  const select = document.getElementById("bias-dataset-select");
  select.innerHTML = datasets.map((d) => `<option value="${d}">${escapeHtml(d)}</option>`).join("");
}

async function pollBiasEvalJob(jobId) {
  const res = await fetch(`${API}/api/bias-evals/${jobId}`);
  const job = await res.json();

  const pct = job.total ? Math.round((job.completed / job.total) * 100) : 0;
  document.getElementById("bias-eval-progress-fill").style.width = `${pct}%`;
  document.getElementById("bias-eval-progress-label").textContent =
    job.total ? `${job.completed} / ${job.total} (scenario × variant) pairs done (${pct}%)` : "Starting…";

  if (job.status === "running") {
    biasEvalPollTimer = setTimeout(() => pollBiasEvalJob(jobId), 2000);
  } else if (job.status === "completed") {
    setStatus("bias-eval-status", "Completed.", "ok");
    document.getElementById("run-bias-eval-btn").disabled = false;
  } else {
    setStatus("bias-eval-status", job.error || "Bias evaluation failed.", "error");
    document.getElementById("run-bias-eval-btn").disabled = false;
  }
}

async function runBiasEval() {
  const model = document.getElementById("bias-model-select").value;
  const dataset = document.getElementById("bias-dataset-select").value;
  if (!model || !dataset) {
    setStatus("bias-eval-status", "Select a model and dataset.", "error");
    return;
  }

  clearTimeout(biasEvalPollTimer);
  document.getElementById("run-bias-eval-btn").disabled = true;
  document.getElementById("bias-eval-progress-wrap").hidden = false;
  document.getElementById("bias-eval-progress-fill").style.width = "0%";
  setStatus("bias-eval-status", "Starting…");

  try {
    const res = await fetch(`${API}/api/bias-evals`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, dataset }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }
    const job = await res.json();
    currentBiasEvalJobId = job.id;
    setStatus("bias-eval-status", "Running…");
    pollBiasEvalJob(currentBiasEvalJobId);
  } catch (e) {
    setStatus("bias-eval-status", e.message, "error");
    document.getElementById("run-bias-eval-btn").disabled = false;
  }
}

function fmtP(p) {
  if (p === null || p === undefined) return "—";
  const cls = p < 0.05 ? "val-neg" : "";
  return `<span class="${cls}">${p.toFixed(4)}</span>`;
}

function renderBiasMetricsTable(pairwise) {
  const container = document.getElementById("bias-metrics-table");
  let html = `<table><thead><tr>
    <th>Variant A</th><th>Variant B</th><th>N</th>
    <th>Action p</th>
    <th>Consequence p</th>
  </tr></thead><tbody>`;
  pairwise.forEach((r) => {
    html += `<tr>
      <td>${escapeHtml(r.variant_a)}</td>
      <td>${escapeHtml(r.variant_b)}</td>
      <td>${r.n_scenarios}</td>
      <td>${fmtP(r.action.p_value)}</td>
      <td>${fmtP(r.consequence.p_value)}</td>
    </tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;
}

async function computeBiasMetrics() {
  const model = document.getElementById("bias-model-select").value;
  const dataset = document.getElementById("bias-dataset-select").value;
  if (!model || !dataset) {
    setStatus("bias-metrics-status", "Select a model and dataset.", "error");
    return;
  }

  document.getElementById("compute-bias-metrics-btn").disabled = true;
  document.getElementById("bias-metrics-results").hidden = true;
  setStatus("bias-metrics-status", "Computing…");

  try {
    const res = await fetch(`${API}/api/bias-metrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, dataset }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }
    const data = await res.json();

    document.getElementById("bias-metrics-summary").textContent =
      `Model: ${modelLabel(data.model)} · Dataset: ${data.dataset} · Variants: ${data.variants.join(", ")}`;

    renderBiasMetricsTable(data.pairwise);
    document.getElementById("bias-metrics-results").hidden = false;
    setStatus("bias-metrics-status", "Done.", "ok");
  } catch (e) {
    setStatus("bias-metrics-status", e.message, "error");
  } finally {
    document.getElementById("compute-bias-metrics-btn").disabled = false;
  }
}

loadModels().then(() => {
  const select = document.getElementById("bias-model-select");
  select.innerHTML = MODELS.map((m) => `<option value="${m.id}">${escapeHtml(m.label)}</option>`).join("");
});
loadBiasDatasets();
document.getElementById("run-bias-eval-btn").addEventListener("click", runBiasEval);
document.getElementById("compute-bias-metrics-btn").addEventListener("click", computeBiasMetrics);
