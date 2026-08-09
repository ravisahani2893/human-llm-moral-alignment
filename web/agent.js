let currentAgentRunId = null;
let agentPollTimer = null;

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
        return `<div class="agent-log-line"><span class="log-task">▶ Instruction:</span> ${escapeHtml(entry.text || "")}</div>`;
      case "tool_call":
        return `<div class="agent-log-line"><span class="log-tool">→ ${escapeHtml(entry.name)}</span>(${escapeHtml(compactArgs(entry.args))})</div>`;
      case "tool_result": {
        const isError = (entry.result || "").includes('"error"');
        const cls = isError ? "log-error" : "log-result";
        return `<div class="agent-log-line"><span class="${cls}">← ${escapeHtml(entry.name)}:</span> ${escapeHtml(resultPreview(entry.result))}</div>`;
      }
      case "job_progress":
        return `<div class="agent-log-line"><span class="log-progress">⏳ ${escapeHtml(entry.job_id)}:</span> ${escapeHtml(entry.status)} (${escapeHtml(String(entry.completed))}/${escapeHtml(String(entry.total))})</div>`;
      case "final_report":
        return `<div class="agent-log-line"><span class="log-final">✓ Final response produced.</span></div>`;
      case "max_turns_reached":
        return `<div class="agent-log-line"><span class="log-error">⚠ Stopped: reached max tool-call turns without a final response.</span></div>`;
      default:
        return "";
    }
  });

  el.innerHTML = lines.join("");
  el.scrollTop = el.scrollHeight;
}

async function pollAgentRun(runId) {
  const res = await fetch(`${API}/api/agent/runs/${runId}`);
  const run = await res.json();

  renderAgentLog(run.log);

  if (run.status === "running") {
    setStatus("agent-status", "Agent is working — deciding which tools to call…");
    agentPollTimer = setTimeout(() => pollAgentRun(runId), 3000);
    return;
  }

  document.getElementById("run-agent-btn").disabled = false;

  if (run.status === "completed") {
    setStatus("agent-status", "Done.", "ok");
    const reportEl = document.getElementById("agent-report");
    reportEl.hidden = false;
    reportEl.innerHTML = renderMarkdown(run.report || "(No response text returned.)");

    if (run.csv_path) {
      document.getElementById("download-agent-csv-btn").disabled = false;
    }
  } else {
    setStatus("agent-status", run.error || "Agent run failed.", "error");
  }
}

async function runAgent() {
  const instruction = document.getElementById("agent-instruction-input").value.trim();
  if (!instruction) {
    setStatus("agent-status", "Enter an instruction for the agent.", "error");
    return;
  }

  clearTimeout(agentPollTimer);
  document.getElementById("run-agent-btn").disabled = true;
  document.getElementById("download-agent-csv-btn").disabled = true;
  document.getElementById("agent-report").hidden = true;
  document.getElementById("agent-log-wrap").hidden = true;
  document.getElementById("agent-log").innerHTML = "";
  setStatus("agent-status", "Starting agent…");

  try {
    const res = await fetch(`${API}/api/agent/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }
    const run = await res.json();
    currentAgentRunId = run.id;
    pollAgentRun(currentAgentRunId);
  } catch (e) {
    setStatus("agent-status", e.message, "error");
    document.getElementById("run-agent-btn").disabled = false;
  }
}

function downloadAgentCsv() {
  if (!currentAgentRunId) return;
  window.location.href = `${API}/api/agent/runs/${currentAgentRunId}/csv`;
}

loadModels();
document.getElementById("run-agent-btn").addEventListener("click", runAgent);
document.getElementById("download-agent-csv-btn").addEventListener("click", downloadAgentCsv);
