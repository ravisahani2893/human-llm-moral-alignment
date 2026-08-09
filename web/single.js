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

loadModels().then(() => buildModelSelect("single-model-select"));
initModal();
document.getElementById("run-single-btn").addEventListener("click", runSingle);
