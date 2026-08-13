// Human-LLM Alignment — human vs. each model's own predictions, one row
// per model per axis. Deliberately kept a separate page from
// Cross-Model Agreement (cross_model.html), which never touches human
// annotations at all.

function renderAlignmentTable(containerId, results, axisKey) {
  const container = document.getElementById(containerId);
  let html = `<table><thead><tr><th>Model</th><th>N</th><th>CCC</th><th>Pearson</th><th>Spearman</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>`;
  results.forEach((r) => {
    if (r.error) {
      html += `<tr><td>${escapeHtml(modelLabel(r.model))}</td><td colspan="6" class="hint">${escapeHtml(r.error)}</td></tr>`;
      return;
    }
    const axis = r[axisKey];
    html += `<tr>
      <td>${escapeHtml(modelLabel(r.model))}</td>
      <td>${r.n_scenarios}</td>
      <td>${fmtValence(axis.ccc)}</td>
      <td>${fmtValence(axis.pearson)}</td>
      <td>${fmtValence(axis.spearman)}</td>
      <td>${axis.mae === null || axis.mae === undefined ? "—" : axis.mae.toFixed(3)}</td>
      <td>${axis.rmse === null || axis.rmse === undefined ? "—" : axis.rmse.toFixed(3)}</td>
    </tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;
}

async function runHumanAlignment() {
  const models = getSelectedModels("ham-model-select");
  const promptVersion = document.getElementById("ham-prompt-version-select").value;

  if (models.length < 1) {
    setStatus("ham-status", "Select at least 1 model to evaluate.", "error");
    return;
  }

  document.getElementById("run-ham-btn").disabled = true;
  document.getElementById("ham-results").hidden = true;
  setStatus("ham-status", "Computing…");

  try {
    const res = await fetch(`${API}/api/alignment-metrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models, prompt_version: promptVersion }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }
    const data = await res.json();

    document.getElementById("ham-summary").textContent =
      `Models: ${data.models.map(modelLabel).join(", ")} · Prompt version: ${data.prompt_version}`;

    renderAlignmentTable("ham-action-table", data.results, "action_results");
    renderAlignmentTable("ham-consequence-table", data.results, "consequence_results");

    document.getElementById("ham-results").hidden = false;
    setStatus("ham-status", "Done.", "ok");
  } catch (e) {
    setStatus("ham-status", e.message, "error");
  } finally {
    document.getElementById("run-ham-btn").disabled = false;
  }
}

loadModels().then(() => buildModelSelect("ham-model-select"));
loadPromptVersions("ham-prompt-version-select");
document.getElementById("run-ham-btn").addEventListener("click", runHumanAlignment);
