const state = {
  mode: "run",
  packs: [],
  settings: {},
  selectedPack: null,
  catalogKind: "runs",
  page: 0,
  pageSize: 25,
  selected: new Set(),
  activeJobs: new Set(),
  current: null,
};

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
}[char]));
const fmt = (value) => Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
const markup = (html) => ({ html });

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
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
  } catch (error) {
    showError(error);
  }
}

function bind() {
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
  $("previous").addEventListener("click", () => { if (state.page) { state.page -= 1; refreshHistory(); } });
  $("next").addEventListener("click", () => { state.page += 1; refreshHistory(); });
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  document.querySelectorAll(".run-only").forEach((node) => node.classList.toggle("hidden", mode !== "run"));
  document.querySelectorAll(".study-only").forEach((node) => node.classList.toggle("hidden", mode !== "study"));
  $("start").textContent = mode === "study" ? "Run study" : "Run experiment";
  $("parameterTitle").textContent = mode === "study" ? "Search space" : "Parameters";
  updateLLMPanel();
  renderParameters();
  renderObjectives();
}

function setCatalog(kind) {
  state.catalogKind = kind;
  state.page = 0;
  state.selected.clear();
  $("compare").disabled = true;
  document.querySelectorAll(".catalog-tab").forEach((item) => item.classList.toggle("active", item.dataset.kind === kind));
  refreshHistory();
}

function selectPack() {
  state.selectedPack = state.packs.find((pack) => pack.id === $("pack").value) || state.packs[0];
  if (!state.selectedPack) return;
  $("packDescription").textContent = state.selectedPack.description;
  $("packMeta").textContent = `${state.selectedPack.schedule.replaceAll("_", " ")} / ${state.selectedPack.steps} steps`;
  $("seeds").value = state.selectedPack.seeds.join(", ");
  $("studySeeds").value = state.selectedPack.seeds.join(", ");
  const llm = state.selectedPack.llm || {};
  $("llmModel").value = state.settings.model || llm.model || "";
  $("llmTemperature").value = llm.temperature ?? 0;
  $("llmSystemPrompt").value = llm.system_prompt || "";
  $("llmPromptPreview").textContent = llm.prompt_preview || "";
  updateLLMPanel();
  renderParameters();
  renderObjectives();
}

function updateLLMPanel() {
  const enabled = Boolean(state.selectedPack?.llm?.enabled);
  $("llmSettings").classList.toggle("hidden", !enabled);
  $("liveBudget").classList.toggle("hidden", !(enabled && state.mode === "study"));
  $("workers").disabled = enabled && state.mode === "study";
  if (enabled && state.mode === "study") $("workers").value = "1";
  $("llmCredentialState").textContent = state.settings.has_api_key ? `key from ${state.settings.api_key_source}` : "key required";
}

function parameterHeader(parameter) {
  const bounds = parameter.minimum !== null && parameter.minimum !== undefined
    ? `${parameter.minimum} to ${parameter.maximum}${parameter.unit ? ` ${parameter.unit}` : ""}`
    : parameter.type;
  return `<div class="parameter-label"><span>${esc(parameter.label)}</span><button type="button" class="help" aria-label="About ${esc(parameter.label)}" data-tooltip="${esc(`${bounds}. ${parameter.description || ""}`)}">?</button></div><p class="hint">${esc(parameter.description || "")}</p>`;
}

function renderParameters() {
  const parameters = state.selectedPack?.parameters || [];
  $("parameters").innerHTML = parameters.map((parameter) => state.mode === "run"
    ? experimentParameter(parameter)
    : studyParameter(parameter)).join("") || `<div class="meta">No guided parameters</div>`;
  bindParameterPairs();
}

function experimentParameter(parameter) {
  const id = `param-${parameter.id}`;
  if (parameter.type === "boolean") {
    return `<div class="parameter">${parameterHeader(parameter)}<label class="check"><input id="${id}" data-param="${esc(parameter.id)}" type="checkbox" ${parameter.default ? "checked" : ""}>Enabled</label></div>`;
  }
  if (parameter.type === "choice") {
    return `<div class="parameter">${parameterHeader(parameter)}<select id="${id}" data-param="${esc(parameter.id)}">${parameter.choices.map((choice) => `<option ${choice === parameter.default ? "selected" : ""}>${esc(choice)}</option>`).join("")}</select></div>`;
  }
  const step = parameter.step || (parameter.type === "integer" ? 1 : 0.01);
  const slider = parameter.slider ? `<input class="range" data-slider-for="${id}" type="range" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.default}" aria-label="${esc(parameter.label)} slider">` : "";
  return `<div class="parameter">${parameterHeader(parameter)}<div class="number-control"><input id="${id}" data-param="${esc(parameter.id)}" data-type="${parameter.type}" type="number" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.default}">${parameter.unit ? `<span class="unit">${esc(parameter.unit)}</span>` : ""}</div>${slider}</div>`;
}

function studyParameter(parameter) {
  if (!parameter.optimizable) {
    return `<div class="parameter muted-parameter">${parameterHeader(parameter)}<div class="meta">Fixed at ${esc(parameter.default)} for studies</div></div>`;
  }
  const id = `study-${parameter.id}`;
  if (!["integer", "float"].includes(parameter.type)) {
    return `<div class="parameter">${parameterHeader(parameter)}<label class="check"><input type="checkbox" data-study-param="${esc(parameter.id)}" checked>Search all allowed values</label></div>`;
  }
  const step = parameter.step || (parameter.type === "integer" ? 1 : 0.01);
  return `<div class="parameter">${parameterHeader(parameter)}<label class="check"><input id="${id}" type="checkbox" data-study-param="${esc(parameter.id)}" checked>Optimize this parameter</label><div class="range-grid"><div><label for="${id}-min">Minimum</label><input id="${id}-min" data-range-min="${esc(parameter.id)}" data-type="${parameter.type}" type="number" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.minimum}"></div><div><label for="${id}-max">Maximum</label><input id="${id}-max" data-range-max="${esc(parameter.id)}" data-type="${parameter.type}" type="number" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.maximum}"></div></div></div>`;
}

function bindParameterPairs() {
  document.querySelectorAll("[data-slider-for]").forEach((slider) => {
    const input = $(slider.dataset.sliderFor);
    slider.addEventListener("input", () => { input.value = slider.value; });
    input.addEventListener("input", () => { slider.value = input.value; });
  });
}

function renderObjectives() {
  const pack = state.selectedPack;
  if (!pack) return;
  const mode = $("studyMode").value;
  const defaults = mode === "single" ? [pack.study.single_objective] : pack.study.pareto_objectives;
  $("objectives").innerHTML = Object.entries(pack.objectives).map(([name, value]) => `<label class="check"><input type="checkbox" data-objective="${esc(name)}" ${defaults.includes(name) ? "checked" : ""}>${esc(name.replaceAll("_", " "))} <span class="meta">${esc(value.direction)}</span></label>`).join("");
}

function experimentValues() {
  const parameters = {};
  document.querySelectorAll("[data-param]").forEach((input) => {
    let value = input.type === "checkbox" ? input.checked : input.value;
    if (input.dataset.type === "integer") value = Number.parseInt(value, 10);
    if (input.dataset.type === "float") value = Number.parseFloat(value);
    if ((input.dataset.type === "integer" || input.dataset.type === "float") && !Number.isFinite(value)) throw new Error(`${input.id} requires a number`);
    parameters[input.dataset.param] = value;
  });
  return parameters;
}

function studySearchSpace() {
  const parameters = [...document.querySelectorAll("[data-study-param]:checked")].map((input) => input.dataset.studyParam);
  if (!parameters.length) throw new Error("Select at least one search parameter");
  const parameterRanges = {};
  parameters.forEach((id) => {
    const minimumInput = document.querySelector(`[data-range-min="${CSS.escape(id)}"]`);
    const maximumInput = document.querySelector(`[data-range-max="${CSS.escape(id)}"]`);
    if (!minimumInput || !maximumInput) return;
    const parse = minimumInput.dataset.type === "integer" ? Number.parseInt : Number.parseFloat;
    const minimum = parse(minimumInput.value, 10);
    const maximum = parse(maximumInput.value, 10);
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum) throw new Error(`${id} needs a valid minimum and maximum`);
    parameterRanges[id] = { minimum, maximum };
  });
  return { parameters, parameter_ranges: parameterRanges };
}

function parsedSeeds(id) {
  const tokens = $(id).value.split(",").map((value) => value.trim());
  if (tokens.some((value) => value && !/^[+-]?\d+$/.test(value))) {
    throw new Error("Seeds must be comma-separated integers");
  }
  const result = tokens.filter(Boolean).map((value) => Number(value));
  if (result.some((value) => !Number.isSafeInteger(value))) {
    throw new Error("Seeds must be safe integers");
  }
  if (!result.length) throw new Error("At least one integer seed is required");
  return [...new Set(result)];
}

function llmPayload() {
  if (!state.selectedPack?.llm?.enabled) return { llm_mode: "none" };
  if (!$("llmModel").value.trim()) throw new Error("A live LLM model is required");
  return {
    llm_mode: "live",
    llm_base_url: $("llmBaseUrl").value.trim(),
    llm_api_key: $("llmApiKey").value,
    llm_model: $("llmModel").value.trim(),
    llm_temperature: Number.parseFloat($("llmTemperature").value),
    llm_system_prompt: $("llmSystemPrompt").value,
  };
}

async function startJob() {
  clearError();
  $("start").disabled = true;
  try {
    const llm = llmPayload();
    let response;
    if (state.mode === "study") {
      const search = studySearchSpace();
      const objectives = [...document.querySelectorAll("[data-objective]:checked")].map((item) => item.dataset.objective);
      const expected = $("studyMode").value === "single" ? 1 : 2;
      if (objectives.length < expected || (expected === 1 && objectives.length !== 1)) throw new Error(expected === 1 ? "Select exactly one objective" : "Select at least two objectives");
      response = await api("/api/studies", { method: "POST", body: JSON.stringify({
        pack: state.selectedPack.id,
        ...llm,
        ...search,
        mode: $("studyMode").value,
        objectives,
        trials: Number.parseInt($("trials").value, 10),
        seeds: parsedSeeds("studySeeds"),
        workers: Number.parseInt($("workers").value, 10),
        allow_live_llm: llm.llm_mode === "live",
        max_llm_calls: $("maxCalls").value || null,
        max_llm_cost_usd: $("maxCost").value || null,
      }) });
      setCatalog("studies");
    } else {
      response = await api("/api/runs", { method: "POST", body: JSON.stringify({
        pack: state.selectedPack.id,
        ...llm,
        seeds: parsedSeeds("seeds"),
        parameters: experimentValues(),
        retention: $("retention").value,
      }) });
      setCatalog("runs");
    }
    const jobs = response.jobs || [response.job];
    jobs.forEach((job) => state.activeJobs.add(job.id));
    $("cancel").classList.remove("hidden");
    setStatus("queued");
    await refreshHistory();
  } catch (error) {
    showError(error);
  } finally {
    $("start").disabled = false;
  }
}

async function cancelJobs() {
  await Promise.all([...state.activeJobs].map((id) => api(`/api/jobs/${id}/cancel`, { method: "POST", body: "{}" })));
  await poll();
}

async function poll() {
  if (!state.activeJobs.size) return;
  try {
    const results = await Promise.all([...state.activeJobs].map((id) => api(`/api/jobs/${id}`)));
    let completed = null;
    results.forEach(({ job }) => {
      setStatus(job.status);
      if (["completed", "failed", "cancelled"].includes(job.status)) {
        state.activeJobs.delete(job.id);
        if (job.status === "completed") completed = job;
        if (job.error) showError(new Error(job.error));
      }
    });
    if (!state.activeJobs.size) $("cancel").classList.add("hidden");
    await refreshHistory();
    if (completed) await openArtifact(completed.kind === "run" ? "runs" : "studies", completed.id);
  } catch (_) {
    state.activeJobs.clear();
    $("cancel").classList.add("hidden");
  }
}

async function refreshHistory() {
  $("history").innerHTML = `<div class="empty">Loading...</div>`;
  try {
    const payload = await api(`/api/${state.catalogKind}?limit=${state.pageSize}&offset=${state.page * state.pageSize}`);
    const rows = payload[state.catalogKind];
    $(state.catalogKind === "runs" ? "runCount" : "studyCount").textContent = payload.total === undefined ? "" : fmt(payload.total);
    $("page").textContent = String(state.page + 1);
    $("previous").disabled = state.page === 0;
    $("next").disabled = rows.length < state.pageSize;
    $("history").innerHTML = rows.map(historyRow).join("") || `<div class="empty">No ${esc(state.catalogKind)} yet</div>`;
  } catch (error) {
    $("history").innerHTML = `<div class="error">${esc(error.message)}</div>`;
  }
}

function historyRow(row) {
  const kind = state.catalogKind;
  const id = row.id;
  const complete = row.status === "completed";
  const title = row.pack_id || id;
  const meta = kind === "runs" ? `seed ${row.seed ?? "-"}` : `${row.trial_count || 0} / ${row.requested_trials || row.trial_count || 0} trials`;
  const selectable = complete ? `<input type="checkbox" data-select="${kind}:${esc(id)}" ${state.selected.has(`${kind}:${id}`) ? "checked" : ""} aria-label="Select ${esc(id)} for comparison">` : `<span class="activity" aria-hidden="true"></span>`;
  return `<div class="history-item ${complete ? "openable" : ""}" ${complete ? `data-open="${kind}:${esc(id)}"` : ""}>${selectable}<div><div class="history-title"><span>${esc(title)}</span><span class="${statusClass(row.status)}">${esc(row.status)}</span></div><div class="history-meta">${esc(meta)} / ${esc(id)}</div>${row.error ? `<div class="error compact">${esc(row.error)}</div>` : ""}</div></div>`;
}

document.addEventListener("click", (event) => {
  const checkbox = event.target.closest("[data-select]");
  if (checkbox) {
    event.stopPropagation();
    checkbox.checked ? state.selected.add(checkbox.dataset.select) : state.selected.delete(checkbox.dataset.select);
    $("compare").disabled = state.selected.size < 2;
    return;
  }
  const rerun = event.target.closest("[data-rerun]");
  if (rerun) {
    event.stopPropagation();
    rerunTrial(rerun.dataset.rerun, Number.parseInt(rerun.dataset.trial, 10));
    return;
  }
  const row = event.target.closest("[data-open]");
  if (row) {
    const [kind, id] = row.dataset.open.split(":");
    openArtifact(kind, id);
  }
});

async function openArtifact(kind, id) {
  clearError();
  try {
    const payload = await api(`/api/${kind}/${id}`);
    state.current = payload;
    render(payload.view);
    $("report").href = `/api/${kind}/${id}/report`;
    $("report").classList.remove("hidden");
    if (kind === "studies") {
      payload.view.trials = await loadAllStudyTrials(id);
      payload.view.trialsComplete = true;
      payload.view.trialTotal = payload.view.trials.length;
      render(payload.view);
    }
  } catch (error) {
    showError(error);
  }
}

async function loadAllStudyTrials(studyId) {
  const pageSize = 500;
  const trials = [];
  let total = Infinity;
  while (trials.length < total) {
    const payload = await api(`/api/studies/${encodeURIComponent(studyId)}/trials?limit=${pageSize}&offset=${trials.length}`);
    total = payload.total;
    trials.push(...payload.trials.map((trial) => ({
      number: trial.number,
      parameters: trial.parameters,
      objectives: trial.objective_values,
      feasible: trial.feasible,
      state: trial.state,
    })));
    if (!payload.trials.length) break;
  }
  return trials;
}

function render(view) {
  $("empty").classList.add("hidden");
  $("title").textContent = view.title;
  $("subtitle").textContent = view.subtitle;
  setStatus(view.status);
  $("facts").innerHTML = section(view.kind === "run" ? "Run facts" : "Study facts", `<div class="fact-grid">${Object.entries(view.facts).map(([label, value]) => cell(label, value, "fact")).join("")}</div>`);
  if (view.kind === "run") renderRun(view); else renderStudy(view);
}

function renderRun(view) {
  $("metrics").innerHTML = section("Trusted objectives", metricGrid(view.objectives)) + section("Online metrics", metricGrid(view.metrics));
  $("chart").innerHTML = section("Metrics over time", lineChart(view.checkpoints));
  const max = Math.max(1, ...Object.values(view.actions));
  $("actions").innerHTML = section("Action mix", Object.entries(view.actions).map(([name, count]) => `<div class="bar-row"><span>${esc(name)}</span><div class="track"><div class="fill" style="width:${100 * count / max}%"></div></div><b>${count}</b></div>`).join("") || `<p class="hint">No actions retained.</p>`);
  $("constraints").innerHTML = section("Trusted constraints", table(["Metric", "Value", "Rule", "Result"], view.constraints.map((item) => [item.metric, fmt(item.value), `${item.operator} ${fmt(item.threshold)}`, markup(`<span class="${item.passed ? "ok" : "bad"}">${item.passed ? "pass" : "fail"}</span>`)])));
  $("agentTable").innerHTML = table(["Agent", "Archetype", "Role", "Policy", "State", "Resources"], view.agents.map((item) => [item.id, item.archetype, item.role, item.policy, item.state, markup(keyValues(item.resources))]));
  $("trialTable").innerHTML = `<div class="empty">This artifact is a single run.</div>`;
  selectTab("summary");
}

function renderStudy(view) {
  $("metrics").innerHTML = section("Objectives", `<div class="metric-grid">${view.objectives.map((name) => cell("Objective", name.replaceAll("_", " "), "metric")).join("")}</div>`);
  $("chart").innerHTML = section("Objective space", studyChart(view));
  $("actions").innerHTML = "";
  $("constraints").innerHTML = "";
  $("agentTable").innerHTML = `<div class="empty">Open a retained run to inspect agents.</div>`;
  const trialLabel = view.trialsComplete
    ? `Complete trial set (${view.trialTotal})`
    : `Summary preview (${view.trials.length} of ${view.facts.Trials})`;
  $("trialTable").innerHTML = `<p class="meta">${esc(trialLabel)}</p>` + table(["Trial", "Parameters", "Objectives", "Feasible", "State", ""], view.trials.map((item) => [item.number, markup(keyValues(item.parameters)), markup(keyValues(item.objectives)), markup(`<span class="${item.feasible ? "ok" : "bad"}">${item.feasible ? "yes" : "no"}</span>`), item.state, markup(`<button class="secondary small" data-rerun="${esc(view.id)}" data-trial="${item.number}">Rerun</button>`)]));
  selectTab("trials");
}

async function rerunTrial(studyId, number) {
  clearError();
  try {
    const payload = await api(`/api/studies/${studyId}/trials/${number}/rerun`, { method: "POST", body: JSON.stringify({ retention: "experiment" }) });
    payload.jobs.forEach((job) => state.activeJobs.add(job.id));
    $("cancel").classList.remove("hidden");
    setCatalog("runs");
  } catch (error) {
    showError(error);
  }
}

async function compareSelected() {
  const loaded = await Promise.all([...state.selected].map(async (key) => {
    const [kind, id] = key.split(":");
    return api(`/api/${kind}/${id}`);
  }));
  $("title").textContent = `${state.catalogKind === "runs" ? "Run" : "Study"} comparison`;
  $("subtitle").textContent = `${loaded.length} selected artifacts`;
  $("facts").innerHTML = "";
  $("actions").innerHTML = "";
  $("constraints").innerHTML = "";
  if (state.catalogKind === "runs") {
    const metricNames = [...new Set(loaded.flatMap((item) => item.view.metrics.map((metric) => metric.id)))];
    const rows = loaded.map((item) => {
      const values = Object.fromEntries(item.view.metrics.map((metric) => [metric.id, metric.value]));
      return [item.view.id, item.view.title, ...metricNames.map((name) => values[name] === undefined ? "" : fmt(values[name]))];
    });
    $("metrics").innerHTML = section("Final metrics", table(["Run", "Pack", ...metricNames], rows));
    $("chart").innerHTML = section("Selected runs", comparisonChart(loaded.map((item) => item.view)));
  } else {
    $("metrics").innerHTML = section("Studies", table(["Study", "Pack", "Trials", "Objectives"], loaded.map((item) => [item.view.id, item.view.title, item.view.facts.Trials, item.view.objectives.join(", ")])));
    $("chart").innerHTML = "";
  }
  $("agentTable").innerHTML = "";
  $("trialTable").innerHTML = "";
  $("report").classList.add("hidden");
  selectTab("summary");
}

function metricGrid(items) {
  return `<div class="metric-grid">${items.map((item) => cell(item.label, fmt(item.value), "metric")).join("")}</div>`;
}

function lineChart(points) {
  if (!points.length) return `<div class="empty">No checkpoints retained for this profile.</div>`;
  const names = [...new Set(points.flatMap((point) => Object.keys(point.values)))].slice(0, 5);
  const values = points.flatMap((point) => names.map((name) => point.values[name]).filter(Number.isFinite));
  if (!values.length) return `<div class="empty">No numeric checkpoint metrics.</div>`;
  const width = 900, height = 280, pad = 42;
  const minStep = Math.min(...points.map((point) => point.step));
  const maxStep = Math.max(...points.map((point) => point.step));
  const low = Math.min(...values, 0), high = Math.max(...values, 1);
  const x = (step) => pad + (step - minStep) * (width - 2 * pad) / Math.max(1, maxStep - minStep);
  const y = (value) => height - pad - (value - low) * (height - 2 * pad) / Math.max(1e-9, high - low);
  const colors = ["#2764a7", "#1d7a55", "#c44b38", "#a36b16", "#7354a3"];
  const series = names.map((name, index) => `<polyline fill="none" stroke="${colors[index]}" stroke-width="2" points="${points.filter((point) => Number.isFinite(point.values[name])).map((point) => `${x(point.step)},${y(point.values[name])}`).join(" ")}"></polyline>`).join("");
  const legend = names.map((name, index) => `<span><i style="background:${colors[index]}"></i>${esc(name.replaceAll("_", " "))}</span>`).join("");
  return `<div class="chart-wrap"><div class="legend">${legend}</div><svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Metrics over experiment steps"><line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line><line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>${series}<text x="${pad}" y="${height - 12}">${minStep}</text><text text-anchor="end" x="${width - pad}" y="${height - 12}">${maxStep} steps</text><text x="6" y="${pad}">${fmt(high)}</text><text x="6" y="${height - pad}">${fmt(low)}</text></svg></div>`;
}

function studyChart(view) {
  const names = view.objectives.slice(0, 2);
  const trials = view.trials.filter((trial) => names.every((name) => Number.isFinite(trial.objectives[name])));
  if (!trials.length || !names.length) return `<div class="empty">No completed objective values.</div>`;
  const xName = names[0], yName = names[1] || names[0];
  const width = 900, height = 280, pad = 42;
  const xs = trials.map((trial) => trial.objectives[xName]), ys = trials.map((trial) => trial.objectives[yName]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const x = (value) => pad + (value - minX) * (width - 2 * pad) / Math.max(1e-9, maxX - minX);
  const y = (value) => height - pad - (value - minY) * (height - 2 * pad) / Math.max(1e-9, maxY - minY);
  const dots = trials.map((trial) => `<circle fill="${trial.feasible ? "#1d7a55" : "#c44b38"}" cx="${x(trial.objectives[xName])}" cy="${y(trial.objectives[yName])}" r="6"><title>Trial ${trial.number}: ${fmt(trial.objectives[xName])}, ${fmt(trial.objectives[yName])}</title></circle>`).join("");
  return `<div class="chart-wrap"><div class="legend"><span><i class="feasible"></i>feasible</span><span><i class="infeasible"></i>infeasible</span></div><svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Study objective plot"><line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line><line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>${dots}<text text-anchor="end" x="${width - pad}" y="${height - 12}">${esc(xName.replaceAll("_", " "))}</text><text x="6" y="18">${esc(yName.replaceAll("_", " "))}</text></svg></div>`;
}

function comparisonChart(views) {
  const names = [...new Set(views.flatMap((view) => view.objectives.map((item) => item.id)))].slice(0, 4);
  if (!names.length) return `<div class="empty">No shared objectives.</div>`;
  const rows = views.map((view) => ({ id: view.id, values: Object.fromEntries(view.objectives.map((item) => [item.id, item.value])) }));
  return `<div class="comparison-bars">${rows.map((row) => `<div><strong>${esc(row.id)}</strong>${names.map((name) => `<span><small>${esc(name.replaceAll("_", " "))}</small><b>${row.values[name] === undefined ? "-" : fmt(row.values[name])}</b></span>`).join("")}</div>`).join("")}</div>`;
}

function keyValues(value) {
  const entries = Object.entries(value || {});
  return entries.length ? `<dl class="key-values">${entries.map(([key, item]) => `<dt>${esc(key.replaceAll("_", " "))}</dt><dd>${esc(typeof item === "number" ? fmt(item) : item)}</dd>`).join("")}</dl>` : `<span class="meta">None</span>`;
}

function section(title, body) {
  return `<section class="report-section"><h3>${esc(title)}</h3>${body}</section>`;
}

function cell(label, value, kind) {
  return `<div class="${kind}-cell"><div class="cell-label">${esc(label)}</div><div class="cell-value">${esc(value)}</div></div>`;
}

function table(headers, rows) {
  const cellValue = (value) => value && typeof value === "object" && "html" in value ? value.html : esc(value);
  return `<div class="table-wrap"><table><thead><tr>${headers.map((header) => `<th>${esc(header)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((value) => `<td>${cellValue(value)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function selectTab(name) {
  document.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === name));
}

function statusClass(status) {
  if (status === "completed") return "ok";
  if (["failed", "cancelled", "interrupted"].includes(status)) return "bad";
  return "pending";
}

function setStatus(status) {
  $("status").textContent = status;
  $("status").className = `status ${statusClass(status)}`;
}

function clearError() {
  $("formError").textContent = "";
}

function showError(error) {
  $("formError").textContent = error.message;
}

init();
