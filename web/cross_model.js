// Cross-Model Agreement — model-vs-model correlation only. Never displays
// or requests human annotation values; this page is deliberately kept
// separate from the Human-LLM Alignment story told elsewhere in the app.

function renderMatrix(containerId, models, matrix) {
  const container = document.getElementById(containerId);
  let html = `<table><thead><tr><th></th>`;
  models.forEach((m) => (html += `<th>${escapeHtml(modelLabel(m))}</th>`));
  html += `</tr></thead><tbody>`;
  models.forEach((rowModel) => {
    html += `<tr><th>${escapeHtml(modelLabel(rowModel))}</th>`;
    models.forEach((colModel) => {
      const v = matrix[rowModel][colModel];
      html += `<td>${fmtMetric(v, agreementLabel)}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;
}

async function runCrossModelAgreement() {
  const models = getSelectedModels("cma-model-select");
  const promptVersion = document.getElementById("cma-prompt-version-select").value;

  if (models.length < 2) {
    setStatus("cma-status", "Select at least 2 models to compare.", "error");
    return;
  }

  document.getElementById("run-cma-btn").disabled = true;
  document.getElementById("cma-results").hidden = true;
  setStatus("cma-status", "Computing…");

  try {
    const res = await fetch(`${API}/api/cross-model-agreement`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models, prompt_version: promptVersion }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }
    const data = await res.json();

    document.getElementById("cma-summary").textContent =
      `Models: ${data.models.map(modelLabel).join(", ")} · Prompt version: ${data.prompt_version} · ` +
      `Common scenarios (present in ALL selected models' predictions): ${data.n_scenarios}`;

    renderMatrix("cma-action-ccc", data.models, data.action.ccc_matrix);
    renderMatrix("cma-action-spearman", data.models, data.action.spearman_matrix);
    renderMatrix("cma-consequence-ccc", data.models, data.consequence.ccc_matrix);
    renderMatrix("cma-consequence-spearman", data.models, data.consequence.spearman_matrix);

    document.getElementById("cma-results").hidden = false;
    setStatus("cma-status", "Done.", "ok");
  } catch (e) {
    setStatus("cma-status", e.message, "error");
  } finally {
    document.getElementById("run-cma-btn").disabled = false;
  }
}

loadModels().then(() => buildModelSelect("cma-model-select"));
loadPromptVersions("cma-prompt-version-select");
document.getElementById("run-cma-btn").addEventListener("click", runCrossModelAgreement);
