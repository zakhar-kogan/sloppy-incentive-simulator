import { actionFrequency, comparisonBars, smallMultiples, studyChart } from "./charts.js";
import { $, esc, factsGrid, fmt, formatMetric, keyValues, markup, metricGrid, table } from "./format.js";
import { bindMechanics, mechanicsGraph, mechanicsTable } from "./mechanics.js";

const state = {
  mode: "run", packs: [], settings: {}, selectedPack: null, catalogKind: "runs",
  page: 0, pageSize: 25, selected: new Set(), activeJobs: new Set(), current: null,
  historyRows: [], llmOffset: 0,
};

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function init() {
  bind();
  $("workers").value = String(Math.min(4, navigator.hardwareConcurrency || 1));
  try {
    const [packs, settings] = await Promise.all([api("/api/packs"), api("/api/settings")]);
    state.packs = packs.packs;
    state.settings = settings.settings;
    $("pack").innerHTML = state.packs.map((pack) => `<option value="${esc(pack.id)}">${esc(pack.title)}</option>`).join("");
    $("llmBaseUrl").value = state.settings.base_url || "";
    selectPack();
    await refreshHistory();
    window.setInterval(poll, 1000);
  } catch (error) { showError(error); }
}

function bind() {
  document.querySelectorAll("[data-workspace]").forEach((button) => button.addEventListener("click", () => setWorkspace(button.dataset.workspace)));
  document.querySelectorAll(".mode").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => selectTab(button.dataset.tab)));
  document.querySelectorAll(".catalog-tab").forEach((button) => button.addEventListener("click", () => setCatalog(button.dataset.kind)));
  $("pack").addEventListener("change", selectPack);
  $("studyMode").addEventListener("change", renderObjectives);
  $("resetParams").addEventListener("click", renderParameters);
  $("start").addEventListener("click", startJob);
  $("cancel").addEventListener("click", cancelJobs);
  $("refresh").addEventListener("click", refreshHistory);
  $("compare").addEventListener("click", compareSelected);
  $("historySearch").addEventListener("input", renderHistory);
  $("historyStatus").addEventListener("change", renderHistory);
  $("previous").addEventListener("click", () => { if (state.page) { state.page -= 1; refreshHistory(); } });
  $("next").addEventListener("click", () => { state.page += 1; refreshHistory(); });
}

function setWorkspace(name) {
  document.querySelectorAll("[data-workspace]").forEach((button) => button.classList.toggle("active", button.dataset.workspace === name));
  document.querySelectorAll(".workspace").forEach((node) => node.classList.toggle("active", node.id === `${name}Workspace`));
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  document.querySelectorAll(".run-only").forEach((node) => node.classList.toggle("hidden", mode !== "run"));
  document.querySelectorAll(".study-only").forEach((node) => node.classList.toggle("hidden", mode !== "study"));
  $("start").textContent = mode === "study" ? "Run study" : "Run experiment";
  $("parameterTitle").textContent = mode === "study" ? "Search space" : "Parameters";
  updateLLMPanel(); renderParameters(); renderObjectives();
}

function setCatalog(kind) {
  state.catalogKind = kind; state.page = 0; state.selected.clear(); $("compare").disabled = true;
  document.querySelectorAll(".catalog-tab").forEach((button) => button.classList.toggle("active", button.dataset.kind === kind));
  refreshHistory();
}

function selectPack() {
  state.selectedPack = state.packs.find((pack) => pack.id === $("pack").value) || state.packs[0];
  if (!state.selectedPack) return;
  $("packDescription").textContent = state.selectedPack.description;
  $("packMeta").textContent = `${state.selectedPack.schedule.replaceAll("_", " ")} / ${state.selectedPack.steps} steps`;
  $("seeds").value = state.selectedPack.seeds.join(", "); $("studySeeds").value = state.selectedPack.seeds.join(", ");
  const llm = state.selectedPack.llm || {};
  $("llmModel").value = state.settings.model || llm.model || ""; $("llmTemperature").value = llm.temperature ?? 0;
  $("llmSystemPrompt").value = llm.system_prompt || ""; $("llmPromptPreview").textContent = llm.prompt_preview || "";
  updateLLMPanel(); renderParameters(); renderObjectives();
}

function updateLLMPanel() {
  const enabled = Boolean(state.selectedPack?.llm?.enabled);
  $("llmSettings").classList.toggle("hidden", !enabled);
  $("liveBudget").classList.toggle("hidden", !(enabled && state.mode === "study"));
  $("workers").disabled = enabled && state.mode === "study";
  if (enabled && state.mode === "study") $("workers").value = "1";
  $("llmCredentialState").textContent = state.settings.has_api_key ? `Key from ${state.settings.api_key_source}` : "Key required";
}

function parameterHeader(parameter) {
  const bounds = parameter.minimum !== null && parameter.minimum !== undefined ? `${parameter.minimum} to ${parameter.maximum}${parameter.unit ? ` ${parameter.unit}` : ""}` : parameter.type;
  return `<div class="parameter-label"><span>${esc(parameter.label)}</span><button type="button" class="help" aria-label="About ${esc(parameter.label)}" data-tooltip="${esc(`${bounds}. ${parameter.description || ""}`)}">?</button></div><p class="hint">${esc(parameter.description || "")}</p>`;
}

function renderParameters() {
  const parameters = state.selectedPack?.parameters || [];
  $("parameters").innerHTML = parameters.map((parameter) => state.mode === "run" ? experimentParameter(parameter) : studyParameter(parameter)).join("") || `<div class="muted">No guided parameters</div>`;
  document.querySelectorAll("[data-slider-for]").forEach((slider) => { const input = $(slider.dataset.sliderFor); slider.addEventListener("input", () => { input.value = slider.value; }); input.addEventListener("input", () => { slider.value = input.value; }); });
}

function experimentParameter(parameter) {
  const id = `param-${parameter.id}`;
  if (parameter.type === "boolean") return `<div class="parameter">${parameterHeader(parameter)}<label class="check"><input id="${id}" data-param="${esc(parameter.id)}" type="checkbox" ${parameter.default ? "checked" : ""}>Enabled</label></div>`;
  if (parameter.type === "choice") return `<div class="parameter">${parameterHeader(parameter)}<select id="${id}" data-param="${esc(parameter.id)}">${parameter.choices.map((choice) => `<option ${choice === parameter.default ? "selected" : ""}>${esc(choice)}</option>`).join("")}</select></div>`;
  const step = parameter.step || (parameter.type === "integer" ? 1 : 0.01);
  const slider = parameter.slider ? `<input data-slider-for="${id}" type="range" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.default}" aria-label="${esc(parameter.label)} slider">` : "";
  return `<div class="parameter">${parameterHeader(parameter)}<div class="number-control"><input id="${id}" data-param="${esc(parameter.id)}" data-type="${parameter.type}" type="number" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.default}">${parameter.unit ? `<span class="unit">${esc(parameter.unit)}</span>` : ""}</div>${slider}</div>`;
}

function studyParameter(parameter) {
  if (!parameter.optimizable) return `<div class="parameter">${parameterHeader(parameter)}<div class="muted">Fixed at ${esc(parameter.default)} for studies</div></div>`;
  const id = `study-${parameter.id}`;
  if (!["integer", "float"].includes(parameter.type)) return `<div class="parameter">${parameterHeader(parameter)}<label class="check"><input type="checkbox" data-study-param="${esc(parameter.id)}" checked>Search all allowed values</label></div>`;
  const step = parameter.step || (parameter.type === "integer" ? 1 : 0.01);
  return `<div class="parameter">${parameterHeader(parameter)}<label class="check"><input id="${id}" type="checkbox" data-study-param="${esc(parameter.id)}" checked>Optimize this parameter</label><div class="range-grid"><div><label for="${id}-min">Minimum</label><input id="${id}-min" data-range-min="${esc(parameter.id)}" data-type="${parameter.type}" type="number" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.minimum}"></div><div><label for="${id}-max">Maximum</label><input id="${id}-max" data-range-max="${esc(parameter.id)}" data-type="${parameter.type}" type="number" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.maximum}"></div></div></div>`;
}

function renderObjectives() {
  const pack = state.selectedPack; if (!pack) return;
  const defaults = $("studyMode").value === "single" ? [pack.study.single_objective] : pack.study.pareto_objectives;
  $("objectives").innerHTML = Object.entries(pack.objectives).map(([name, value]) => `<label class="check"><input type="checkbox" data-objective="${esc(name)}" ${defaults.includes(name) ? "checked" : ""}>${esc(name.replaceAll("_", " "))} <span class="muted">${esc(value.direction)}</span></label>`).join("");
}

function experimentValues() {
  const parameters = {};
  document.querySelectorAll("[data-param]").forEach((input) => { let value = input.type === "checkbox" ? input.checked : input.value; if (input.dataset.type === "integer") value = Number.parseInt(value, 10); if (input.dataset.type === "float") value = Number.parseFloat(value); if (["integer", "float"].includes(input.dataset.type) && !Number.isFinite(value)) throw new Error(`${input.id} requires a number`); parameters[input.dataset.param] = value; });
  return parameters;
}

function studySearchSpace() {
  const parameters = [...document.querySelectorAll("[data-study-param]:checked")].map((input) => input.dataset.studyParam);
  if (!parameters.length) throw new Error("Select at least one search parameter");
  const parameterRanges = {};
  parameters.forEach((id) => { const minimumInput = document.querySelector(`[data-range-min="${CSS.escape(id)}"]`); const maximumInput = document.querySelector(`[data-range-max="${CSS.escape(id)}"]`); if (!minimumInput || !maximumInput) return; const parse = minimumInput.dataset.type === "integer" ? Number.parseInt : Number.parseFloat; const minimum = parse(minimumInput.value, 10), maximum = parse(maximumInput.value, 10); if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum) throw new Error(`${id} needs a valid minimum and maximum`); parameterRanges[id] = { minimum, maximum }; });
  return { parameters, parameter_ranges: parameterRanges };
}

function parsedSeeds(id) {
  const tokens = $(id).value.split(",").map((value) => value.trim());
  if (tokens.some((value) => value && !/^[+-]?\d+$/.test(value))) throw new Error("Seeds must be comma-separated integers");
  const result = tokens.filter(Boolean).map(Number);
  if (result.some((value) => !Number.isSafeInteger(value))) throw new Error("Seeds must be safe integers");
  if (!result.length) throw new Error("At least one integer seed is required");
  return [...new Set(result)];
}

function llmPayload() {
  if (!state.selectedPack?.llm?.enabled) return { llm_mode: "none" };
  if (!$("llmModel").value.trim()) throw new Error("A live LLM model is required");
  return { llm_mode: "live", llm_base_url: $("llmBaseUrl").value.trim(), llm_api_key: $("llmApiKey").value, llm_model: $("llmModel").value.trim(), llm_temperature: Number.parseFloat($("llmTemperature").value), llm_system_prompt: $("llmSystemPrompt").value };
}

async function startJob() {
  clearError(); $("start").disabled = true;
  try {
    const llm = llmPayload(); let response;
    if (state.mode === "study") {
      const objectives = [...document.querySelectorAll("[data-objective]:checked")].map((item) => item.dataset.objective);
      const expected = $("studyMode").value === "single" ? 1 : 2;
      if (objectives.length < expected || (expected === 1 && objectives.length !== 1)) throw new Error(expected === 1 ? "Select exactly one objective" : "Select at least two objectives");
      response = await api("/api/studies", { method: "POST", body: JSON.stringify({ pack: state.selectedPack.id, ...llm, ...studySearchSpace(), mode: $("studyMode").value, objectives, trials: Number.parseInt($("trials").value, 10), seeds: parsedSeeds("studySeeds"), workers: Number.parseInt($("workers").value, 10), allow_live_llm: llm.llm_mode === "live", max_llm_calls: $("maxCalls").value || null, max_llm_cost_usd: $("maxCost").value || null }) });
      setCatalog("studies");
    } else {
      response = await api("/api/runs", { method: "POST", body: JSON.stringify({ pack: state.selectedPack.id, ...llm, seeds: parsedSeeds("seeds"), parameters: experimentValues(), retention: $("retention").value }) });
      setCatalog("runs");
    }
    (response.jobs || [response.job]).forEach((job) => state.activeJobs.add(job.id));
    $("cancel").classList.remove("hidden"); setStatus("queued"); setWorkspace("results"); await refreshHistory();
  } catch (error) { showError(error); } finally { $("start").disabled = false; }
}

async function cancelJobs() { await Promise.all([...state.activeJobs].map((id) => api(`/api/jobs/${id}/cancel`, { method: "POST", body: "{}" }))); await poll(); }

async function poll() {
  if (!state.activeJobs.size) return;
  try {
    const results = await Promise.all([...state.activeJobs].map((id) => api(`/api/jobs/${id}`))); let completed = null;
    results.forEach(({ job }) => { setStatus(job.status); renderProgress(job); if (["completed", "failed", "cancelled"].includes(job.status)) { state.activeJobs.delete(job.id); if (job.status === "completed") completed = job; if (job.error) showError(new Error(job.error)); } });
    if (!state.activeJobs.size) $("cancel").classList.add("hidden");
    await refreshHistory(); if (completed) await openArtifact(completed.kind === "run" ? "runs" : "studies", completed.id);
  } catch (_) { state.activeJobs.clear(); $("cancel").classList.add("hidden"); }
}

function renderProgress(job) {
  const progress = job.progress || {};
  if (job.kind === "study") $("liveProgress").textContent = progress.trials_planned ? `${progress.trials_completed || 0} / ${progress.trials_planned} trials` : job.status;
  else $("liveProgress").textContent = progress.steps_planned ? `${progress.steps_completed || 0} / ${progress.steps_planned} steps${progress.llm?.attempted ? ` / ${progress.llm.attempted} LLM calls` : ""}` : job.status;
}

async function refreshHistory() {
  $("history").innerHTML = `<div class="empty compact">Loading...</div>`;
  try {
    const payload = await api(`/api/${state.catalogKind}?limit=${state.pageSize}&offset=${state.page * state.pageSize}`);
    state.historyRows = payload[state.catalogKind];
    $(state.catalogKind === "runs" ? "runCount" : "studyCount").textContent = payload.total === undefined ? "" : fmt(payload.total, 0);
    $("page").textContent = String(state.page + 1); $("previous").disabled = state.page === 0; $("next").disabled = state.historyRows.length < state.pageSize; renderHistory();
  } catch (error) { $("history").innerHTML = `<div class="error">${esc(error.message)}</div>`; }
}

function renderHistory() {
  const query = $("historySearch").value.trim().toLowerCase(), status = $("historyStatus").value;
  const rows = state.historyRows.filter((row) => (!query || JSON.stringify(row).toLowerCase().includes(query)) && (!status || row.status === status));
  $("history").innerHTML = rows.map(historyRow).join("") || `<div class="empty compact">No matching ${esc(state.catalogKind)}.</div>`;
}

function historyRow(row) {
  const kind = state.catalogKind, complete = row.status === "completed", id = row.id;
  const detail = kind === "runs" ? `seed ${row.seed ?? "-"}${row.llm_calls ? ` / ${row.llm_calls} calls / ${row.estimated_llm_cost_usd === null ? "cost unavailable" : `$${Number(row.estimated_llm_cost_usd).toFixed(4)}`}` : ""}` : `${row.trial_count || 0} / ${row.requested_trials || row.trial_count || 0} trials`;
  const selectable = complete ? `<input type="checkbox" data-select="${kind}:${esc(id)}" ${state.selected.has(`${kind}:${id}`) ? "checked" : ""} aria-label="Select ${esc(id)} for comparison">` : `<span class="activity" aria-hidden="true"></span>`;
  return `<div class="history-item ${complete ? "openable" : ""}" ${complete ? `data-open="${kind}:${esc(id)}"` : ""}>${selectable}<div><div class="history-title"><span>${esc(row.pack_id || id)}</span><span class="${statusClass(row.status)}">${esc(row.status)}</span></div><div class="history-meta">${esc(detail)}<br>${esc(id)}</div>${row.error ? `<div class="error">${esc(row.error)}</div>` : ""}</div></div>`;
}

document.addEventListener("click", (event) => {
  const checkbox = event.target.closest("[data-select]"); if (checkbox) { event.stopPropagation(); checkbox.checked ? state.selected.add(checkbox.dataset.select) : state.selected.delete(checkbox.dataset.select); $("compare").disabled = state.selected.size < 2; return; }
  const rerun = event.target.closest("[data-rerun]"); if (rerun) { event.stopPropagation(); rerunTrial(rerun.dataset.rerun, Number.parseInt(rerun.dataset.trial, 10)); return; }
  const row = event.target.closest("[data-open]"); if (row) { const [kind, id] = row.dataset.open.split(":"); openArtifact(kind, id); }
});

async function openArtifact(kind, id) {
  clearError();
  try {
    const payload = await api(`/api/${kind}/${id}`); state.current = payload; $("report").href = `/api/${kind}/${id}/report`; $("report").classList.remove("hidden");
    if (kind === "studies") { payload.view.trials = await loadAllStudyTrials(id); payload.view.trialsComplete = true; payload.view.trialTotal = payload.view.trials.length; }
    render(payload.view); setWorkspace("results");
  } catch (error) { showError(error); }
}

async function loadAllStudyTrials(studyId) {
  const pageSize = 500, trials = []; let total = Infinity;
  while (trials.length < total) { const payload = await api(`/api/studies/${encodeURIComponent(studyId)}/trials?limit=${pageSize}&offset=${trials.length}`); total = payload.total; trials.push(...payload.trials.map((trial) => ({ number: trial.number, parameters: trial.parameters, objectives: trial.objective_values, feasible: trial.feasible, state: trial.state, winner: trial.number === state.current?.view.best_trial, frontier: state.current?.view.pareto_trials.includes(trial.number) }))); if (!payload.trials.length) break; }
  return trials;
}

function render(view) {
  $("resultEmpty").classList.add("hidden"); $("title").textContent = view.title; $("subtitle").textContent = view.subtitle; setStatus(view.status);
  document.querySelectorAll(".run-tab").forEach((node) => node.classList.toggle("hidden", view.kind !== "run"));
  document.querySelectorAll(".study-tab").forEach((node) => node.classList.toggle("hidden", view.kind !== "study"));
  $("resultTabs").querySelector(".llm-tab").classList.toggle("hidden", !(view.kind === "run" && view.has_llm));
  if (view.kind === "run") renderRun(view); else renderStudy(view); selectTab("overview");
}

function findings(view) { return `<div class="findings">${view.findings.map((item) => `<div class="finding"><strong>${esc(item.kind)}</strong><span>${esc(item.text)}</span></div>`).join("") || `<div class="empty compact">No deterministic findings are available.</div>`}</div>`; }

function renderRun(view) {
  $("overview").innerHTML = `<section class="result-section">${findings(view)}</section><section class="result-section"><div class="section-heading"><h3>Run facts</h3></div>${factsGrid(view.facts)}</section><section class="result-section"><div class="section-heading"><h3>Trusted objectives</h3></div>${metricGrid(view.objectives)}</section><section class="result-section"><div class="section-heading"><h3>Trusted constraints</h3></div>${table(["Metric", "Value", "Rule", "Result"], view.constraints.map((item) => [item.label, formatMetric(item.value, item.format), `${item.operator} ${formatMetric(item.threshold, item.format)}`, markup(`<span class="${item.passed ? "ok" : "bad"}">${item.passed ? "pass" : "fail"}</span>`)]), "Trusted constraints")}</section>`;
  $("charts").innerHTML = smallMultiples(view) + `<section class="result-section"><div class="section-heading"><h3>Action frequency by checkpoint</h3></div>${actionFrequency(view.checkpoints)}</section>`;
  $("mechanics").innerHTML = `<section class="result-section">${mechanicsGraph(view.mechanics)}</section><details><summary>Accessible mechanics table</summary>${mechanicsTable(view.mechanics)}</details>`; bindMechanics(view.mechanics);
  $("agents").innerHTML = table(["Agent", "Archetype", "Policy", "Reward", "Actions", "Failures", "Violations", "Enforced", "Resources"], view.agents.map((item) => [item.id, item.archetype, item.policy, formatMetric(item.reward), markup(keyValues(item.action_counts)), item.failed_decisions, item.violations, item.enforcement, markup(keyValues(item.resources))]), "Agent statistics");
  if (view.has_llm) renderLLM(view);
}

async function renderLLM(view) {
  const usage = view.llm;
  $("llm").innerHTML = `<div class="llm-summary"><div><span class="cell-label">Attempted</span><div class="cell-value">${usage.attempted}</div></div><div><span class="cell-label">Completed</span><div class="cell-value">${usage.completed}</div></div><div><span class="cell-label">Failed / malformed / invalid</span><div class="cell-value">${usage.failed} / ${usage.malformed} / ${usage.invalid}</div></div><div><span class="cell-label">Tokens</span><div class="cell-value">${fmt(usage.total_tokens, 0)}</div></div><div><span class="cell-label">Estimated cost</span><div class="cell-value">${usage.estimated_cost_usd === null ? "Unavailable" : formatMetric(usage.estimated_cost_usd, "currency")}</div></div><div><span class="cell-label">Latency p50 / p95</span><div class="cell-value">${usage.approximate_p50_ms ?? "-"} / ${usage.approximate_p95_ms ?? "-"} ms</div></div></div><section class="result-section"><div class="section-heading"><h3>Redacted calls</h3><div><button id="llmPrevious" class="icon-button" aria-label="Previous calls">&#8592;</button> <button id="llmNext" class="icon-button" aria-label="Next calls">&#8594;</button></div></div><div id="llmCalls" class="empty compact">Loading calls...</div></section>`;
  state.llmOffset = 0; await loadLLMCalls(view.id);
  $("llmPrevious").addEventListener("click", async () => { state.llmOffset = Math.max(0, state.llmOffset - 50); await loadLLMCalls(view.id); });
  $("llmNext").addEventListener("click", async () => { state.llmOffset += 50; await loadLLMCalls(view.id); });
}

async function loadLLMCalls(runId) {
  try { const payload = await api(`/api/runs/${encodeURIComponent(runId)}/llm-calls?limit=50&offset=${state.llmOffset}`); $("llmPrevious").disabled = state.llmOffset === 0; $("llmNext").disabled = state.llmOffset + payload.calls.length >= payload.total; $("llmCalls").innerHTML = table(["Step", "Agent", "Provider / model", "Status", "Tokens", "Latency", "Action", "Failure"], payload.calls.map((call) => [call.step ?? "-", call.agent_id ?? "-", `${call.provider || "unknown"} / ${call.model || "unknown"}`, call.status || (call.error ? "failed" : "completed"), call.total_tokens || 0, `${fmt(call.latency_ms || 0, 1)} ms`, call.selected_action || "-", call.failure_classification || call.error || "-"]), "LLM calls"); } catch (error) { $("llmCalls").innerHTML = `<div class="error">${esc(error.message)}</div>`; }
}

function renderStudy(view) {
  $("overview").innerHTML = `<section class="result-section">${findings(view)}</section><section class="result-section"><div class="section-heading"><h3>Study facts</h3></div>${factsGrid(view.facts)}</section>`;
  const controls = `<div class="grid-four"><div><label for="chartX">X axis</label><select id="chartX">${view.objectives.map((name) => `<option>${esc(name)}</option>`).join("")}</select></div><div><label for="chartY">Y axis</label><select id="chartY">${view.objectives.map((name, index) => `<option ${index === 1 ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></div><label class="check"><input id="feasibleOnly" type="checkbox">Feasible only</label></div><div id="studyPlot">${studyChart(view)}</div>`;
  $("charts").innerHTML = `<section class="result-section">${controls}</section>`;
  const redraw = () => { $("studyPlot").innerHTML = studyChart(view, $("chartX").value, $("chartY").value, $("feasibleOnly").checked); };
  $("chartX").addEventListener("change", redraw); $("chartY").addEventListener("change", redraw); $("feasibleOnly").addEventListener("change", redraw);
  const trialLabel = view.trialsComplete ? `Complete trial set (${view.trialTotal})` : `Summary preview (${view.trials.length} of ${view.facts.Trials})`;
  $("trials").innerHTML = `<p class="muted">${esc(trialLabel)}</p>${table(["Trial", "Parameters", "Objectives", "Feasible", "State", ""], view.trials.map((item) => [markup(`${item.winner ? "<strong>Winner</strong> " : ""}${item.frontier ? "<span class=\"ok\">Frontier</span> " : ""}${item.number}`), markup(keyValues(item.parameters)), markup(keyValues(item.objectives)), item.feasible ? "yes" : "no", item.state, markup(`<button class="secondary" data-rerun="${esc(view.id)}" data-trial="${item.number}">Rerun</button>`)]), "Study trials")}`;
  $("retained").innerHTML = view.retained_run_ids.length ? `<div class="findings">${view.retained_run_ids.map((id) => `<button class="finding text-button" data-open="runs:${esc(id)}"><strong>Run</strong><span>${esc(id)}</span></button>`).join("")}</div>` : `<div class="empty">No runs were retained for this study.</div>`;
}

async function rerunTrial(studyId, number) { clearError(); try { const payload = await api(`/api/studies/${studyId}/trials/${number}/rerun`, { method: "POST", body: JSON.stringify({ retention: "experiment" }) }); payload.jobs.forEach((job) => state.activeJobs.add(job.id)); $("cancel").classList.remove("hidden"); setCatalog("runs"); } catch (error) { showError(error); } }

async function compareSelected() {
  const loaded = await Promise.all([...state.selected].map((key) => { const [kind, id] = key.split(":"); return api(`/api/${kind}/${id}`); }));
  $("title").textContent = `${state.catalogKind === "runs" ? "Run" : "Study"} comparison`; $("subtitle").textContent = `${loaded.length} selected artifacts`; $("report").classList.add("hidden");
  if (state.catalogKind === "runs") {
    const metricIds = [...new Set(loaded.flatMap((item) => item.view.metrics.map((metric) => metric.id)))];
    const labels = Object.fromEntries(metricIds.map((id) => {
      const metric = loaded.flatMap((item) => item.view.metrics).find((item) => item.id === id);
      return [id, metric?.label || id.replaceAll("_", " ")];
    }));
    const rows = loaded.map((item) => {
      const metrics = Object.fromEntries(item.view.metrics.map((metric) => [metric.id, metric]));
      return [item.view.id, item.view.title, ...metricIds.map((id) => metrics[id] ? formatMetric(metrics[id].value, metrics[id].format) : "-")];
    });
    $("overview").innerHTML = `<section class="result-section"><div class="section-heading"><h3>Final metrics</h3></div>${table(["Run", "Pack", ...metricIds.map((id) => labels[id])], rows, "Run comparison")}</section>`;
    $("charts").innerHTML = comparisonBars(loaded.map((item) => item.view));
  } else {
    $("overview").innerHTML = table(["Study", "Pack", "Trials", "Objectives"], loaded.map((item) => [item.view.id, item.view.title, item.view.facts.Trials, item.view.objectives.join(", ")]), "Study comparison");
    $("charts").innerHTML = "";
  }
  document.querySelectorAll(".run-tab,.study-tab,.llm-tab").forEach((node) => node.classList.add("hidden")); selectTab("overview");
}

function selectTab(name) { document.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === name)); document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === name)); }
function statusClass(status) { if (status === "completed") return "ok"; if (["failed", "cancelled", "interrupted"].includes(status)) return "bad"; return "pending"; }
function setStatus(status) { $("status").textContent = status; $("status").className = `status ${statusClass(status)}`; }
function clearError() { $("formError").textContent = ""; }
function showError(error) { $("formError").textContent = error.message; }

init();
