import { actionFrequency, comparisonBars, comparisonOverlays, cumulativeBest, parameterEffects, proxyOutcomeChart, smallMultiples, studyChart } from "./charts.js?v=0.5.0-3";
import { $, esc, factsGrid, fmt, formatMetric, keyValues, markup, metricGrid, table } from "./format.js?v=0.5.0-3";
import { bindMechanics, causalFlowGraph, mechanicsTable, stateMachineGraph } from "./mechanics.js?v=0.5.0-3";

const REQUIRED_CAPABILITIES = ["causal_mechanics", "population_overrides", "population_templates", "policy_templates", "quick_values", "runtime_handshake"];
const LLM_DEFAULTS_KEY = "icframe.llm-defaults.v1";
const LLM_API_KEY = "icframe.llm-api-key.v1";

const state = {
  mode: "run", packs: [], settings: {}, profiles: { execution: {}, llm: {} }, selectedPack: null, catalogKind: "runs",
  page: 0, pageSize: 25, selected: new Set(), activeJobs: new Set(), current: null,
  historyRows: [], llmOffset: 0, populationDraft: [], polling: false, currentLiveJob: null,
  currentTab: "overview", connectionInitialized: false,
};

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  const payload = await response.json();
  if (!response.ok) { const error = new Error(payload.error || `HTTP ${response.status}`); error.endpoint = path; error.status = response.status; throw error; }
  return payload;
}

async function init() {
  bind();
  $("workers").value = String(Math.min(4, navigator.hardwareConcurrency || 1));
  try {
    const runtimePayload = await api("/api/runtime");
    const runtime = runtimePayload.runtime || {}, missing = REQUIRED_CAPABILITIES.filter((capability) => !(runtime.capabilities || []).includes(capability));
    if (runtime.ui_api_version !== "1" || missing.length) { const error = new Error(`The running backend is incompatible with these UI assets${missing.length ? ` (missing: ${missing.join(", ")})` : ""}.`); error.endpoint = "/api/runtime"; throw error; }
    const [packs, settings, profiles] = await Promise.all([api("/api/packs"), api("/api/settings"), api("/api/profiles")]);
    state.packs = packs.packs;
    state.settings = settings.settings;
    state.profiles = profiles.profiles;
    $("pack").innerHTML = state.packs.map((pack) => `<option value="${esc(pack.id)}">${esc(pack.title)}</option>`).join("");
    $("executionProfile").innerHTML = Object.entries(state.profiles.execution).map(([name, profile]) => `<option value="${esc(name)}">${esc(name)} · ${esc(profile.type.replaceAll("_", " "))}</option>`).join("");
    $("llmProfile").innerHTML = `<option value="">Browser / runtime settings</option>${Object.keys(state.profiles.llm).map((name) => `<option value="${esc(name)}">${esc(name)}</option>`).join("")}`;
    initializeLLMConnection();
    selectPack();
    await refreshHistory();
    window.setInterval(poll, 1000);
  } catch (error) { showStartupError(error); }
}

function bind() {
  document.querySelectorAll("[data-workspace]").forEach((button) => button.addEventListener("click", () => setWorkspace(button.dataset.workspace)));
  document.querySelectorAll(".mode").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => selectTab(button.dataset.tab)));
  document.querySelectorAll(".catalog-tab").forEach((button) => button.addEventListener("click", () => setCatalog(button.dataset.kind)));
  $("pack").addEventListener("change", selectPack);
  $("executionProfile").addEventListener("change", updateProfileStatus);
  $("llmProfile").addEventListener("change", selectLLMProfile);
  $("studyPreset").addEventListener("change", applyStudyPreset);
  $("planner").addEventListener("change", () => { $("trials").disabled = $("planner").value === "matrix"; });
  $("studyMode").addEventListener("change", renderObjectives);
  $("resetParams").addEventListener("click", renderParameters);
  $("resetPopulation").addEventListener("click", resetPopulation);
  $("addPopulation").addEventListener("click", addPopulation);
  $("checkModels").addEventListener("click", checkModels);
  $("applyLLMDefaults").addEventListener("click", applyLLMDefaultsToGroups);
  $("clearLLMDefaults").addEventListener("click", clearLLMDefaults);
  ["llmBaseUrl", "llmModel", "llmTemperature", "llmSystemPrompt"].forEach((id) => $(id).addEventListener("input", persistLLMDefaults));
  $("llmApiKey").addEventListener("input", () => { sessionStorage.setItem(LLM_API_KEY, $("llmApiKey").value); updateCredentialState(); });
  $("retryStartup").addEventListener("click", () => window.location.reload());
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
  updateLLMPanel(); renderParameters(); renderObjectives(); renderPopulation();
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
  $("llmPromptPreview").textContent = llm.prompt_preview || "";
  resetPopulation(); updateLLMPanel(); renderParameters(); renderObjectives();
  renderStudyPresets(); updateProfileStatus();
}

function renderStudyPresets() {
  const presets = state.selectedPack?.study?.presets || [];
  $("studyPreset").innerHTML = `<option value="">Custom study</option>${presets.map((preset) => `<option value="${esc(preset.id)}">${esc(preset.label)}</option>`).join("")}`;
}

function applyStudyPreset() {
  const preset = (state.selectedPack?.study?.presets || []).find((item) => item.id === $("studyPreset").value);
  if (!preset) return;
  $("planner").value = preset.planner;
  $("plannerSeed").value = preset.planner_seed ?? 0;
  $("studyMode").value = preset.objectives.length === 1 ? "single" : "pareto";
  $("studySeeds").value = preset.seeds.join(", ");
  const trialCount = preset.planner === "matrix" ? Object.values(preset.parameter_matrix || {}).reduce((total, values) => total * values.length, 1) : preset.trials;
  $("trials").value = String(trialCount);
  $("trials").disabled = preset.planner === "matrix";
  renderObjectives(); renderParameters();
  const presetParameters = preset.parameters?.length ? preset.parameters : Object.keys(preset.parameter_matrix || {});
  if (preset.exclude_archetypes?.length) {
    state.populationDraft = state.populationDraft.filter((item) => !preset.exclude_archetypes.includes(item.archetype_id));
    renderPopulation(); updateLLMPanel();
  }
  document.querySelectorAll("[data-study-param]").forEach((input) => { input.checked = presetParameters.includes(input.dataset.studyParam); });
  document.querySelectorAll("[data-objective]").forEach((input) => { input.checked = preset.objectives.includes(input.dataset.objective); });
}

function updateProfileStatus() {
  const name = $("executionProfile").value || "local", profile = state.profiles.execution[name];
  if (!profile) { $("executionStatus").textContent = "Unknown profile"; return; }
  $("executionStatus").textContent = profile.type === "local" ? `${profile.workers} local workers · ${profile.status}` : `${profile.platform} / ${profile.preset} · ${profile.max_in_flight} shards in flight · ${profile.status}`;
}

function selectLLMProfile() {
  const profile = state.profiles.llm[$("llmProfile").value];
  if (!profile) { updateCredentialState(); return; }
  $("llmBaseUrl").value = profile.base_url;
  if (profile.model) $("llmModel").value = profile.model;
  $("llmApiKey").value = "";
  updateCredentialState();
}

function updateLLMPanel() {
  const enabled = state.populationDraft.some((item) => item.policy === "llm_policy");
  $("llmSettings").classList.toggle("hidden", !enabled);
  $("liveBudget").classList.toggle("hidden", !(enabled && state.mode === "study"));
  $("workers").disabled = enabled && state.mode === "study";
  if (enabled && state.mode === "study") $("workers").value = "1";
  updateCredentialState();
}

function storedLLMDefaults() {
  try { return JSON.parse(localStorage.getItem(LLM_DEFAULTS_KEY) || "{}"); } catch (_) { return {}; }
}

function initializeLLMConnection() {
  if (state.connectionInitialized) return;
  const stored = storedLLMDefaults(), pack = state.packs.find((item) => item.llm?.enabled)?.llm || {};
  $("llmBaseUrl").value = stored.base_url ?? state.settings.base_url ?? "";
  $("llmModel").value = stored.model ?? state.settings.model ?? pack.model ?? "";
  $("llmTemperature").value = stored.temperature ?? pack.temperature ?? 0;
  $("llmSystemPrompt").value = stored.system_prompt ?? pack.system_prompt ?? "";
  $("llmApiKey").value = sessionStorage.getItem(LLM_API_KEY) || "";
  state.connectionInitialized = true; updateCredentialState();
}

function currentLLMDefaults() {
  return { model: $("llmModel").value.trim(), temperature: Number.parseFloat($("llmTemperature").value) || 0, system_prompt: $("llmSystemPrompt").value };
}

function persistLLMDefaults() {
  localStorage.setItem(LLM_DEFAULTS_KEY, JSON.stringify({ base_url: $("llmBaseUrl").value.trim(), ...currentLLMDefaults() }));
}

function clearLLMDefaults() {
  localStorage.removeItem(LLM_DEFAULTS_KEY); state.connectionInitialized = false; initializeLLMConnection(); $("modelCheckState").textContent = "Saved non-secret defaults cleared";
}

function updateCredentialState() {
  const profile = state.profiles.llm[$("llmProfile").value];
  $("llmCredentialState").textContent = profile ? (profile.has_api_key ? `Key available from ${profile.api_key_env}` : `Set ${profile.api_key_env}`) : $("llmApiKey").value ? "Key from browser session" : state.settings.has_api_key ? `Key from ${state.settings.api_key_source}` : "Key required";
}

function applyLLMDefaultsToGroups() {
  syncPopulation(); const defaults = currentLLMDefaults(); let changed = 0;
  state.populationDraft.forEach((item) => { if (item.policy !== "llm_policy") return; item.llm = { ...(item.llm || {}), provider: item.llm?.provider || "litellm", require_json: true, ...defaults }; changed += 1; });
  renderPopulation(); $("modelCheckState").textContent = `Applied defaults to ${changed} LLM group${changed === 1 ? "" : "s"}`;
}

async function checkModels() {
  const button = $("checkModels"); button.disabled = true; $("modelCheckState").textContent = "Checking...";
  try {
    const payload = await api("/api/models", { method: "POST", body: JSON.stringify({ llm_profile: $("llmProfile").value || null, base_url: $("llmBaseUrl").value.trim(), api_key: $("llmApiKey").value }) });
    $("llmModels").innerHTML = payload.models.map((model) => `<option value="${esc(model)}"></option>`).join("");
    const requested = $("llmModel").value.trim();
    if (requested && !payload.models.includes(requested)) throw new Error(`Model ${requested} was not returned by this endpoint`);
    $("modelCheckState").textContent = `${payload.models.length} models available`;
  } catch (error) { $("modelCheckState").textContent = error.message; } finally { button.disabled = false; }
}

function parameterHeader(parameter) {
  const bounds = parameter.minimum !== null && parameter.minimum !== undefined ? `${parameter.minimum} to ${parameter.maximum}${parameter.unit ? ` ${parameter.unit}` : ""}` : parameter.type;
  return `<div class="parameter-label"><span>${esc(parameter.label)}</span><button type="button" class="help" aria-label="About ${esc(parameter.label)}" data-tooltip="${esc(`${bounds}. ${parameter.description || ""}`)}">?</button></div><p class="hint">${esc(parameter.description || "")}</p>`;
}

function renderParameters() {
  const parameters = state.selectedPack?.parameters || [];
  $("parameters").innerHTML = parameters.map((parameter) => state.mode === "run" ? experimentParameter(parameter) : studyParameter(parameter)).join("") || `<div class="muted">No guided parameters</div>`;
  document.querySelectorAll("[data-slider-for]").forEach((slider) => { const input = $(slider.dataset.sliderFor); slider.addEventListener("input", () => { input.value = slider.value; }); input.addEventListener("input", () => { slider.value = input.value; }); });
  document.querySelectorAll("[data-quick-for]").forEach((button) => button.addEventListener("click", () => { const input = $(button.dataset.quickFor); input.value = button.dataset.quickValue; input.dispatchEvent(new Event("input")); }));
}

function experimentParameter(parameter) {
  const id = `param-${parameter.id}`;
  if (parameter.type === "boolean") return `<div class="parameter">${parameterHeader(parameter)}<label class="check"><input id="${id}" data-param="${esc(parameter.id)}" type="checkbox" ${parameter.default ? "checked" : ""}>Enabled</label></div>`;
  if (parameter.type === "choice") return `<div class="parameter">${parameterHeader(parameter)}<select id="${id}" data-param="${esc(parameter.id)}">${parameter.choices.map((choice) => `<option ${choice === parameter.default ? "selected" : ""}>${esc(choice)}</option>`).join("")}</select></div>`;
  const step = parameter.step || (parameter.type === "integer" ? 1 : 0.01);
  const slider = parameter.slider ? `<input data-slider-for="${id}" type="range" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.default}" aria-label="${esc(parameter.label)} slider">` : "";
  const quick = (parameter.quick_values || []).length ? `<div class="quick-values">${parameter.quick_values.map((item) => `<button type="button" class="secondary" data-quick-for="${id}" data-quick-value="${item.value}">${esc(item.label)}</button>`).join("")}</div>` : "";
  const target = parameter.target ? `<p class="hint">Changes ${esc(parameter.target.entity)}${parameter.target.entity_id ? ` ${esc(parameter.target.entity_id)}` : ""}: ${esc(parameter.target.field.join("."))}</p>` : "";
  return `<div class="parameter">${parameterHeader(parameter)}${target}<div class="number-control"><input id="${id}" data-param="${esc(parameter.id)}" data-type="${parameter.type}" type="number" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.default}">${parameter.unit ? `<span class="unit">${esc(parameter.unit)}</span>` : ""}</div>${slider}${quick}</div>`;
}

function studyParameter(parameter) {
  if (!parameter.optimizable) return `<div class="parameter">${parameterHeader(parameter)}<div class="muted">Fixed at ${esc(parameter.default)} for studies</div></div>`;
  const id = `study-${parameter.id}`;
  if (!["integer", "float"].includes(parameter.type)) return `<div class="parameter">${parameterHeader(parameter)}<label class="check"><input type="checkbox" data-study-param="${esc(parameter.id)}" checked>Search all allowed values</label></div>`;
  const step = parameter.step || (parameter.type === "integer" ? 1 : 0.01);
  return `<div class="parameter">${parameterHeader(parameter)}<label class="check"><input id="${id}" type="checkbox" data-study-param="${esc(parameter.id)}" checked>Optimize this parameter</label><div class="range-grid"><div><label for="${id}-min">Minimum</label><input id="${id}-min" data-range-min="${esc(parameter.id)}" data-type="${parameter.type}" type="number" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.minimum}"></div><div><label for="${id}-max">Maximum</label><input id="${id}-max" data-range-max="${esc(parameter.id)}" data-type="${parameter.type}" type="number" min="${parameter.minimum}" max="${parameter.maximum}" step="${step}" value="${parameter.maximum}"></div></div></div>`;
}

function resetPopulation() {
  const composition = state.selectedPack?.composition;
  if (!composition) return;
  state.populationDraft = composition.population.map((entry) => ({ archetype_id: entry.archetype, count: entry.count, ...structuredClone(composition.archetypes[entry.archetype]) }));
  renderPopulation(); updateLLMPanel();
}

function uniqueArchetypeId(base = "agent") {
  const used = new Set(state.populationDraft.map((item) => item.archetype_id)); let index = 1, candidate = base;
  while (used.has(candidate)) candidate = `${base}_${index++}`;
  return candidate;
}

function addPopulation() {
  renderPopulationTemplates(); $("populationTemplateChooser").classList.toggle("hidden");
}

function renderPopulationTemplates() {
  const templates = state.selectedPack?.population_templates || [];
  $("populationTemplateChooser").innerHTML = templates.map((item) => `<button type="button" class="template-option" data-pop-template="${esc(item.id)}"><strong>${esc(item.label)}</strong><span>${esc(item.description)}</span><small>${esc(item.policy.replaceAll("_", " "))}</small></button>`).join("");
  $("populationTemplateChooser").querySelectorAll("[data-pop-template]").forEach((button) => button.addEventListener("click", () => addPopulationFromTemplate(button.dataset.popTemplate)));
}

function addPopulationFromTemplate(templateId) {
  const source = state.selectedPack.population_templates.find((item) => item.id === templateId); if (!source) return;
  const item = structuredClone(source); delete item.id; delete item.label; delete item.description;
  item.archetype_id = uniqueArchetypeId(item.archetype_id || "agent");
  if (item.policy === "llm_policy") item.llm = { ...(item.llm || {}), ...currentLLMDefaults() };
  state.populationDraft.push(item); $("populationTemplateChooser").classList.add("hidden"); renderPopulation(); updateLLMPanel();
}

function policyTemplate(policy) { return state.selectedPack.policy_templates.find((item) => item.policy === policy); }
function populationOptions(values, selected) { return values.map((value) => `<option value="${esc(value)}" ${value === selected ? "selected" : ""}>${esc(value.replaceAll("_", " "))}</option>`).join(""); }

function renderPopulation() {
  const channels = state.selectedPack?.composition?.outcome_channels || [];
  $("populationEditor").innerHTML = state.populationDraft.map((item, index) => {
    const template = policyTemplate(item.policy) || { fields: [] };
    const policyFields = template.fields.map((field) => `<div><label>${esc(field.id.replaceAll("_", " "))}</label><input data-pop-config="${esc(field.id)}" type="number" min="${field.minimum ?? ""}" max="${field.maximum ?? ""}" step="${field.step}" value="${item.policy_config?.[field.id] ?? field.default}"></div>`).join("");
    const preferences = item.policy === "deterministic" ? `<div><label>Preferred actions</label><input data-pop-preferences value="${esc((item.policy_config?.preferences || []).join(", "))}" placeholder="action_a, action_b"></div>` : "";
    const llm = item.policy === "llm_policy" ? `<div class="grid-two population-llm"><div><label>Model</label><input data-pop-llm="model" value="${esc(item.llm?.model || state.selectedPack.llm?.model || "")}"></div><div><label>Temperature</label><input data-pop-llm="temperature" type="number" min="0" max="2" step="0.1" value="${item.llm?.temperature ?? 0}"></div><div class="full"><label>System prompt</label><textarea data-pop-llm="system_prompt" rows="3">${esc(item.llm?.system_prompt || "")}</textarea></div></div>` : "";
    const weights = channels.map((channel) => `<label class="weight-row"><span>${esc(channel)}</span><input data-pop-weight="${esc(channel)}" type="number" step="0.1" value="${item.scalarizer?.[channel] ?? 0}"></label>`).join("");
    const searchFields = [{ id: "count", minimum: 1, maximum: Math.max(10, item.count * 4), step: 1 }, ...template.fields.map((field) => ({ id: `policy_config.${field.id}`, minimum: field.minimum ?? 0, maximum: field.maximum ?? 10, step: field.step })), ...(item.policy === "llm_policy" ? [{ id: "llm.temperature", minimum: 0, maximum: 2, step: 0.1 }] : [])];
    const searchRows = state.mode === "study" ? searchFields.map((field) => `<div class="composition-range"><label class="check"><input type="checkbox" data-comp-opt="composition.${esc(item.archetype_id)}.${esc(field.id)}">Optimize ${esc(field.id.replaceAll("_", " "))}</label><div class="range-grid"><input data-comp-min type="number" step="${field.step}" value="${field.minimum}"><input data-comp-max type="number" step="${field.step}" value="${field.maximum}"></div></div>`).join("") : "";
    return `<article class="population-group" data-pop-index="${index}"><header><strong>${esc(item.archetype_id)}</strong><div class="inline-actions"><button type="button" class="icon-button" data-pop-move="-1" aria-label="Move group up">&#8593;</button><button type="button" class="icon-button" data-pop-move="1" aria-label="Move group down">&#8595;</button><button type="button" class="text-button" data-pop-duplicate>Duplicate</button><button type="button" class="text-button bad" data-pop-remove>Remove</button></div></header><div class="grid-four"><div><label>Archetype ID</label><input data-pop-field="archetype_id" value="${esc(item.archetype_id)}"></div><div><label>Count</label><input data-pop-field="count" type="number" min="1" value="${item.count}"></div><div><label>Policy</label><select data-pop-field="policy">${state.selectedPack.policy_templates.map((value) => `<option value="${value.policy}" ${value.policy === item.policy ? "selected" : ""}>${esc(value.label)}</option>`).join("")}</select></div><div><label>Role</label><input data-pop-field="role" value="${esc(item.role)}"></div><div><label>Visibility</label><select data-pop-field="visibility_profile">${populationOptions(state.selectedPack.composition.visibility_profiles, item.visibility_profile)}</select></div>${policyFields}${preferences}</div>${llm}<details><summary>Reward weights and initial resources</summary><div class="weight-grid">${weights}</div><label>Initial resources (JSON object)</label><textarea data-pop-resources rows="2">${esc(JSON.stringify(item.initial_resources || {}))}</textarea></details>${searchRows ? `<details><summary>Study search fields</summary><div class="composition-search">${searchRows}</div></details>` : ""}</article>`;
  }).join("");
  $("populationEditor").querySelectorAll("[data-pop-field=policy]").forEach((select) => select.addEventListener("change", () => { syncPopulation(); const item = state.populationDraft[Number(select.closest("[data-pop-index]").dataset.popIndex)]; item.policy = select.value; item.policy_config = Object.fromEntries((policyTemplate(select.value)?.fields || []).map((field) => [field.id, field.default])); item.llm = select.value === "llm_policy" ? { provider: "litellm", require_json: true, ...currentLLMDefaults() } : null; renderPopulation(); updateLLMPanel(); }));
  $("populationEditor").querySelectorAll("[data-pop-remove]").forEach((button) => button.addEventListener("click", () => { syncPopulation(); state.populationDraft.splice(Number(button.closest("[data-pop-index]").dataset.popIndex), 1); renderPopulation(); updateLLMPanel(); }));
  $("populationEditor").querySelectorAll("[data-pop-duplicate]").forEach((button) => button.addEventListener("click", () => { syncPopulation(); const index = Number(button.closest("[data-pop-index]").dataset.popIndex); const copy = structuredClone(state.populationDraft[index]); copy.archetype_id = uniqueArchetypeId(`${copy.archetype_id}_copy`); state.populationDraft.splice(index + 1, 0, copy); renderPopulation(); }));
  $("populationEditor").querySelectorAll("[data-pop-move]").forEach((button) => button.addEventListener("click", () => { syncPopulation(); const index = Number(button.closest("[data-pop-index]").dataset.popIndex), next = index + Number(button.dataset.popMove); if (next < 0 || next >= state.populationDraft.length) return; [state.populationDraft[index], state.populationDraft[next]] = [state.populationDraft[next], state.populationDraft[index]]; renderPopulation(); }));
  renderPopulationWarnings();
}

function renderPopulationWarnings() {
  const details = state.selectedPack?.composition?.visibility_profile_details || {}, warnings = [];
  state.populationDraft.forEach((item) => { if (!Object.values(item.scalarizer || {}).some((value) => Number(value) !== 0)) warnings.push(`${item.archetype_id} has no reward weights`); if (item.policy === "llm_policy" && !item.llm?.model?.trim()) warnings.push(`${item.archetype_id} has no model`); if (item.policy === "llm_policy" && !item.llm?.system_prompt?.trim()) warnings.push(`${item.archetype_id} has no system prompt`); if (item.policy === "llm_policy" && details[item.visibility_profile]?.prompts === false) warnings.push(`${item.archetype_id} cannot receive prompts under ${item.visibility_profile}`); });
  $("populationWarnings").classList.toggle("hidden", !warnings.length); $("populationWarnings").innerHTML = warnings.length ? `<strong>Review configuration</strong><ul>${warnings.map((value) => `<li>${esc(value)}</li>`).join("")}</ul>` : "";
}

function syncPopulation() {
  document.querySelectorAll("[data-pop-index]").forEach((row) => {
    const draft = state.populationDraft[Number(row.dataset.popIndex)];
    row.querySelectorAll("[data-pop-field]").forEach((input) => { draft[input.dataset.popField] = input.dataset.popField === "count" ? Number.parseInt(input.value, 10) : input.value.trim(); });
    draft.policy_config = {};
    row.querySelectorAll("[data-pop-config]").forEach((input) => { draft.policy_config[input.dataset.popConfig] = Number.parseFloat(input.value); });
    const preferences = row.querySelector("[data-pop-preferences]"); if (preferences) draft.policy_config.preferences = preferences.value.split(",").map((value) => value.trim()).filter(Boolean);
    draft.scalarizer = {}; row.querySelectorAll("[data-pop-weight]").forEach((input) => { const value = Number.parseFloat(input.value); if (value) draft.scalarizer[input.dataset.popWeight] = value; });
    try { draft.initial_resources = JSON.parse(row.querySelector("[data-pop-resources]").value || "{}"); } catch (_) { throw new Error(`${draft.archetype_id} initial resources must be a JSON object`); }
    if (draft.policy === "llm_policy") { draft.llm = { provider: "litellm", require_json: true }; row.querySelectorAll("[data-pop-llm]").forEach((input) => { draft.llm[input.dataset.popLlm] = input.dataset.popLlm === "temperature" ? Number.parseFloat(input.value) : input.value; }); } else draft.llm = null;
  });
}

function populationPayload() { syncPopulation(); if (!state.populationDraft.length) throw new Error("At least one population group is required"); return structuredClone(state.populationDraft); }

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
  const parameterRanges = {};
  parameters.forEach((id) => { const minimumInput = document.querySelector(`[data-range-min="${CSS.escape(id)}"]`); const maximumInput = document.querySelector(`[data-range-max="${CSS.escape(id)}"]`); if (!minimumInput || !maximumInput) return; const parse = minimumInput.dataset.type === "integer" ? Number.parseInt : Number.parseFloat; const minimum = parse(minimumInput.value, 10), maximum = parse(maximumInput.value, 10); if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum) throw new Error(`${id} needs a valid minimum and maximum`); parameterRanges[id] = { minimum, maximum }; });
  return { parameters, parameter_ranges: parameterRanges };
}

function compositionSearchSpace() {
  const parameters = [], composition_parameter_ranges = {};
  document.querySelectorAll("[data-comp-opt]:checked").forEach((input) => {
    const row = input.closest(".composition-range"), minimum = Number.parseFloat(row.querySelector("[data-comp-min]").value), maximum = Number.parseFloat(row.querySelector("[data-comp-max]").value);
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum) throw new Error(`${input.dataset.compOpt} needs a valid search range`);
    parameters.push(input.dataset.compOpt); composition_parameter_ranges[input.dataset.compOpt] = { minimum, maximum };
  });
  return { parameters, composition_parameter_ranges };
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
  const hasLLM = state.populationDraft.some((item) => item.policy === "llm_policy");
  if (!hasLLM) return { llm_mode: "none" };
  return { llm_mode: "live", llm_profile: $("llmProfile").value || null, llm_base_url: $("llmBaseUrl").value.trim(), llm_api_key: $("llmApiKey").value };
}

async function startJob() {
  clearError(); $("start").disabled = true;
  try {
    const llm = llmPayload(); let response;
    if (state.mode === "study") {
      const objectives = [...document.querySelectorAll("[data-objective]:checked")].map((item) => item.dataset.objective);
      const expected = $("studyMode").value === "single" ? 1 : 2;
      if (objectives.length < expected || (expected === 1 && objectives.length !== 1)) throw new Error(expected === 1 ? "Select exactly one objective" : "Select at least two objectives");
      const manifestSearch = studySearchSpace(), compositionSearch = compositionSearchSpace(), preset = (state.selectedPack.study.presets || []).find((item) => item.id === $("studyPreset").value), parameters = preset ? (preset.parameters?.length ? preset.parameters : Object.keys(preset.parameter_matrix || {})) : [...manifestSearch.parameters, ...compositionSearch.parameters];
      if (!parameters.length) throw new Error("Select at least one search parameter");
      if ($("planner").value === "matrix" && !preset) throw new Error("Select a named matrix preset or use seeded random planning");
      response = await api("/api/studies", { method: "POST", body: JSON.stringify({ pack: state.selectedPack.id, ...llm, execution_profile: $("executionProfile").value, planner: preset?.planner || $("planner").value, planner_seed: Number.parseInt($("plannerSeed").value, 10), parameter_matrix: preset?.parameter_matrix || {}, parameters, parameter_ranges: manifestSearch.parameter_ranges, composition_parameter_ranges: compositionSearch.composition_parameter_ranges, population_overrides: populationPayload(), mode: $("studyMode").value, objectives, trials: Number.parseInt($("trials").value, 10), seeds: parsedSeeds("studySeeds"), workers: Number.parseInt($("workers").value, 10), allow_live_llm: llm.llm_mode === "live", max_llm_calls: $("maxCalls").value || null, max_llm_cost_usd: $("maxCost").value || null }) });
      setCatalog("studies");
    } else {
      response = await api("/api/runs", { method: "POST", body: JSON.stringify({ pack: state.selectedPack.id, ...llm, execution_profile: $("executionProfile").value, population_overrides: populationPayload(), seeds: parsedSeeds("seeds"), parameters: experimentValues(), retention: $("retention").value }) });
      setCatalog("runs");
    }
    const launched = response.jobs || [response.job]; launched.forEach((job) => state.activeJobs.add(job.id));
    $("cancel").classList.remove("hidden"); setStatus("queued"); setWorkspace("results"); await refreshHistory();
    if (launched.length === 1) await openArtifact("jobs", launched[0].id);
  } catch (error) { showError(error); } finally { $("start").disabled = false; }
}

async function cancelJobs() { await Promise.all([...state.activeJobs].map((id) => api(`/api/jobs/${id}/cancel`, { method: "POST", body: "{}" }))); await poll(); }

async function poll() {
  if (!state.activeJobs.size || state.polling) return;
  state.polling = true;
  try {
    const results = await Promise.all([...state.activeJobs].map((id) => api(`/api/jobs/${id}`))); let completed = null;
    results.forEach(({ job }) => { setStatus(job.status); renderProgress(job); if (state.currentLiveJob === job.id) renderLiveJob(job); if (["completed", "failed", "cancelled"].includes(job.status)) { state.activeJobs.delete(job.id); if (job.status === "completed" && state.currentLiveJob === job.id) completed = job; if (job.error) showError(new Error(job.error)); } });
    if (!state.activeJobs.size) $("cancel").classList.add("hidden");
    await refreshHistory({ background: true }); if (completed) await openArtifact(completed.kind === "run" ? "runs" : "studies", completed.id, true);
  } catch (_) { state.activeJobs.clear(); $("cancel").classList.add("hidden"); } finally { state.polling = false; }
}

function renderProgress(job) {
  const progress = job.progress || {};
  if (progress.cancel_requested) { $("liveProgress").textContent = "Cancellation requested · waiting for active shards"; return; }
  if (job.kind === "study") $("liveProgress").textContent = progress.trials_planned ? `${progress.trials_completed || 0} / ${progress.trials_planned} trials · ${progress.shard_count || 1} shards${progress.remote_job_ids?.length ? ` · ${progress.remote_job_ids.length} remote jobs` : ""}${progress.artifact_import_state ? ` · ${progress.artifact_import_state}` : ""}` : job.status;
  else $("liveProgress").textContent = progress.steps_planned ? `${progress.steps_completed || 0} / ${progress.steps_planned} steps${progress.llm?.attempted ? ` / ${progress.llm.attempted} LLM calls` : ""}` : `${job.status}${progress.backend_profile ? ` · ${progress.backend_profile}` : ""}${progress.artifact_import_state ? ` · ${progress.artifact_import_state}` : ""}`;
}

async function refreshHistory({ background = false } = {}) {
  if (!background && !state.historyRows.length) $("history").innerHTML = `<div class="empty compact">Loading...</div>`;
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
  const html = rows.map(historyRow).join("") || `<div class="empty compact">No matching ${esc(state.catalogKind)}.</div>`;
  const container = $("history"), scrollTop = container.scrollTop;
  const template = document.createElement("template"); template.innerHTML = html;
  const incoming = [...template.content.children], existing = new Map([...container.children].map((node) => [node.dataset.rowKey || "", node]));
  incoming.forEach((node, index) => { const current = existing.get(node.dataset.rowKey || ""); if (current && current.outerHTML === node.outerHTML) { if (container.children[index] !== current) container.insertBefore(current, container.children[index] || null); existing.delete(node.dataset.rowKey || ""); } else if (current) { current.replaceWith(node); existing.delete(node.dataset.rowKey || ""); } else container.insertBefore(node, container.children[index] || null); });
  existing.forEach((node) => node.remove()); container.scrollTop = scrollTop;
}

function historyRow(row) {
  const kind = state.catalogKind, complete = row.status === "completed", id = row.id, job = Boolean(row.job);
  const detail = kind === "runs" ? `seed ${row.seed ?? "-"}${row.llm_calls ? ` / ${row.llm_calls} calls / ${row.estimated_llm_cost_usd === null ? "cost unavailable" : `$${Number(row.estimated_llm_cost_usd).toFixed(4)}`}` : ""}` : `${row.trial_count || 0} / ${row.requested_trials || row.trial_count || 0} trials`;
  const selectable = complete ? `<input type="checkbox" data-select="${kind}:${esc(id)}" ${state.selected.has(`${kind}:${id}`) ? "checked" : ""} aria-label="Select ${esc(id)} for comparison">` : `<span class="activity" aria-hidden="true"></span>`;
  const openKind = job ? "jobs" : kind;
  return `<div class="history-item openable ${state.currentLiveJob === id ? "selected-row" : ""}" data-row-key="${kind}:${esc(id)}" data-open="${openKind}:${esc(id)}" tabindex="0" role="button">${selectable}<div><div class="history-title"><span>${esc(row.pack_id || id)}</span><span class="${statusClass(row.status)}">${esc(row.status)}</span></div><div class="history-meta">${esc(detail)}<br>${esc(id)}</div>${row.error ? `<div class="error">${esc(row.error)}</div>` : ""}</div></div>`;
}

document.addEventListener("click", (event) => {
  const studyParameters = event.target.closest("[data-study-parameters]"); if (studyParameters) { prepareStudy(studyParameters.dataset.studyParameters.split(",").filter(Boolean)); return; }
  const applyWinner = event.target.closest("[data-apply-winner]"); if (applyWinner) { applyWinnerToSetup(); return; }
  const evidence = event.target.closest("[data-evidence]"); if (evidence) { openEvidence(evidence.dataset.evidence); return; }
  const checkbox = event.target.closest("[data-select]"); if (checkbox) { event.stopPropagation(); checkbox.checked ? state.selected.add(checkbox.dataset.select) : state.selected.delete(checkbox.dataset.select); $("compare").disabled = state.selected.size < 2; return; }
  const rerun = event.target.closest("[data-rerun]"); if (rerun) { event.stopPropagation(); rerunTrial(rerun.dataset.rerun, Number.parseInt(rerun.dataset.trial, 10)); return; }
  const row = event.target.closest("[data-open]"); if (row) { const [kind, id] = row.dataset.open.split(":"); openArtifact(kind, id); }
});
document.addEventListener("keydown", (event) => { const row = event.target.closest("[data-open]"); if (row && ["Enter", " "].includes(event.key)) { event.preventDefault(); row.click(); } });

async function openArtifact(kind, id, preserveTab = false) {
  clearError();
  try {
    if (kind === "jobs") { const payload = await api(`/api/jobs/${id}`); state.currentLiveJob = id; state.current = payload; renderLiveJob(payload.job); setWorkspace("results"); return; }
    const payload = await api(`/api/${kind}/${id}`); state.current = payload; $("report").href = `/api/${kind}/${id}/report`; $("report").classList.remove("hidden");
    if (kind === "studies") { payload.view.trials = await loadAllStudyTrials(id); payload.view.trialsComplete = true; payload.view.trialTotal = payload.view.trials.length; }
    state.currentLiveJob = null; render(payload.view, preserveTab); setWorkspace("results");
  } catch (error) { showError(error); }
}

function renderLiveJob(job) {
  state.currentLiveJob = job.id; $("resultEmpty").classList.add("hidden"); $("report").classList.add("hidden");
  $("title").textContent = job.request.pack || job.id; $("subtitle").textContent = `${job.kind} ${job.id} / live`; setStatus(job.status);
  const progress = job.progress || {}, planned = job.kind === "study" ? progress.trials_planned : progress.steps_planned, completed = job.kind === "study" ? progress.trials_completed : progress.steps_completed;
  const percent = planned ? Math.min(100, 100 * (completed || 0) / planned) : 0;
  const errors = progress.recent_errors || [];
  $("overview").innerHTML = `<section class="result-section"><div class="section-heading"><div><span class="eyebrow">Live progress</span><h3>${completed || 0} of ${planned || "?"} ${job.kind === "study" ? "trials" : "steps"}</h3></div><strong>${fmt(percent, 0)}%</strong></div><div class="progress-track"><i style="width:${percent}%"></i></div></section><section class="result-section"><div class="section-heading"><h3>Current metrics</h3></div>${metricGrid(Object.entries(progress.metrics || {}).map(([id, value]) => ({ id, label: id.replaceAll("_", " "), value, format: "number", description: "Latest retained checkpoint" })))}</section>${errors.length ? `<section class="result-section"><div class="section-heading"><h3>Recent failures</h3></div><ul class="error-list">${errors.map((error) => `<li>${esc(error)}</li>`).join("")}</ul></section>` : ""}`;
  $("charts").innerHTML = `<div class="empty">Charts become available as a persisted artifact completes.</div>`; $("mechanics").innerHTML = ""; $("agents").innerHTML = ""; $("trials").innerHTML = ""; $("retained").innerHTML = "";
  document.querySelectorAll(".run-tab,.study-tab").forEach((node) => node.classList.add("hidden"));
  const hasLLM = job.kind === "run" && (progress.llm?.attempted || (job.request.population_overrides || []).some((item) => item.policy === "llm_policy"));
  $("resultTabs").querySelector(".llm-tab").classList.toggle("hidden", !hasLLM);
  if (hasLLM) renderLiveLLM(job); if (![...document.querySelectorAll(".tab:not(.hidden)")].some((node) => node.dataset.tab === state.currentTab)) selectTab("overview");
}

async function renderLiveLLM(job) {
  const usage = job.progress?.llm || {};
  $("llm").innerHTML = `<div class="llm-summary"><div><span class="cell-label">Attempted</span><div class="cell-value">${usage.attempted || 0}</div></div><div><span class="cell-label">Failed</span><div class="cell-value">${usage.failed || 0}</div></div><div><span class="cell-label">Tokens</span><div class="cell-value">${fmt(usage.total_tokens || 0, 0)}</div></div><div><span class="cell-label">Estimated cost</span><div class="cell-value">${usage.cost === null ? "Unavailable" : formatMetric(usage.cost || 0, "currency")}</div></div><div><span class="cell-label">Latency p50 / p95</span><div class="cell-value">${fmt(usage.latency_p50_ms || 0, 1)} / ${fmt(usage.latency_p95_ms || 0, 1)} ms</div></div></div><p class="hint">Only provider-returned rationale or reasoning summaries can be shown. Hidden chain-of-thought is unavailable.</p><section class="result-section"><div class="section-heading"><h3>Live redacted calls</h3></div><div id="llmCalls" class="empty compact">Loading calls...</div></section>`;
  await loadLLMCalls(job.id, true);
}

async function loadAllStudyTrials(studyId) {
  const pageSize = 500, trials = []; let total = Infinity;
  while (trials.length < total) { const payload = await api(`/api/studies/${encodeURIComponent(studyId)}/trials?limit=${pageSize}&offset=${trials.length}`); total = payload.total; trials.push(...payload.trials.map((trial) => { const names = [...new Set((trial.seeds || []).flatMap((seed) => Object.keys(seed.metrics || {})))], metrics = Object.fromEntries(names.map((name) => { const values = trial.seeds.map((seed) => seed.metrics?.[name]).filter(Number.isFinite); return [name, values.reduce((sum, value) => sum + value, 0) / values.length]; })); return { number: trial.number, parameters: trial.parameters, objectives: trial.objective_values, metrics, feasible: trial.feasible, state: trial.state, winner: trial.number === state.current?.view.best_trial, frontier: state.current?.view.pareto_trials.includes(trial.number) }; })); if (!payload.trials.length) break; }
  return trials;
}

function render(view, preserveTab = false) {
  $("resultEmpty").classList.add("hidden"); $("title").textContent = view.title; $("subtitle").textContent = view.subtitle; setStatus(view.status);
  document.querySelectorAll(".run-tab").forEach((node) => node.classList.toggle("hidden", view.kind !== "run"));
  document.querySelectorAll(".study-tab").forEach((node) => node.classList.toggle("hidden", view.kind !== "study"));
  $("resultTabs").querySelector(".llm-tab").classList.toggle("hidden", !(view.kind === "run" && view.has_llm));
  if (view.kind === "run") renderRun(view); else renderStudy(view); selectTab(preserveTab ? state.currentTab : "overview");
}

function findings(view) { return `<div class="findings">${view.findings.map((item) => `<div class="finding"><strong>${esc(item.kind)}</strong><div><span>${esc(item.text)}</span>${item.evidence?.length ? `<div class="evidence-links">${item.evidence.map((evidence) => `<button type="button" class="text-button" data-evidence="${esc(evidence)}">${esc(evidence.replace(":", " "))}</button>`).join("")}</div>` : ""}</div></div>`).join("") || `<div class="empty compact">No deterministic findings are available.</div>`}</div>`; }

function renderRun(view) {
  const next = view.optimizable_parameters.length ? `<div class="next-experiment"><p>This run describes one observed trajectory and does not identify a causal optimum. Test the declared parameters across seeds before choosing values.</p><button type="button" class="primary" data-study-parameters="${esc(view.optimizable_parameters.join(","))}">Study these parameters</button></div>` : `<p class="muted">This pack declares no optimizable parameters.</p>`;
  $("overview").innerHTML = `<section class="result-section"><div class="section-heading"><div><span class="eyebrow">What happened</span><h3>Deterministic findings</h3></div></div>${findings(view)}</section><section class="result-section"><div class="section-heading"><div><span class="eyebrow">Evidence</span><h3>Run facts</h3></div></div>${factsGrid(view.facts)}</section><section class="result-section" data-evidence-target="metrics"><div class="section-heading"><h3>Trusted objectives</h3></div>${metricGrid(view.objectives)}</section><section class="result-section" data-evidence-target="constraints"><div class="section-heading"><h3>Trusted constraints</h3></div>${table(["Metric", "Value", "Rule", "Result"], view.constraints.map((item) => [item.label, formatMetric(item.value, item.format), `${item.operator} ${formatMetric(item.threshold, item.format)}`, markup(`<span class="${item.passed ? "ok" : "bad"}">${item.passed ? "pass" : "fail"}</span>`)]), "Trusted constraints")}</section><section class="result-section"><div class="section-heading"><div><span class="eyebrow">Next experiment</span><h3>Test, do not infer</h3></div></div>${next}</section>`;
  $("charts").innerHTML = smallMultiples(view) + `<section class="result-section"><div class="section-heading"><h3>Action frequency by checkpoint</h3></div>${actionFrequency(view.checkpoints)}</section>`;
  renderMechanics(view.mechanics);
  $("agents").innerHTML = table(["Agent", "Archetype", "Policy", "Reward", "Actions", "Failures", "Violations", "Enforced", "Resources"], view.agents.map((item) => [item.id, item.archetype, item.policy, formatMetric(item.reward), markup(keyValues(item.action_counts)), item.failed_decisions, item.violations, item.enforcement, markup(keyValues(item.resources))]), "Agent statistics");
  if (view.has_llm) renderLLM(view);
}

function renderMechanics(mechanics) {
  const initial = mechanics.causal_flow ? "causal" : "state";
  $("mechanics").innerHTML = `<div class="mechanics-switch" role="group" aria-label="Mechanics representation">${mechanics.causal_flow ? `<button type="button" data-mechanics-view="causal">Causal flow</button>` : ""}<button type="button" data-mechanics-view="state">State machine</button><button type="button" data-mechanics-view="table">Table</button></div><section id="mechanicsView" class="result-section"></section>`;
  const show = (mode) => { document.querySelectorAll("[data-mechanics-view]").forEach((button) => button.classList.toggle("active", button.dataset.mechanicsView === mode)); $("mechanicsView").innerHTML = mode === "causal" ? causalFlowGraph(mechanics) : mode === "state" ? stateMachineGraph(mechanics) : mechanicsTable(mechanics); bindMechanics(mechanics); };
  document.querySelectorAll("[data-mechanics-view]").forEach((button) => button.addEventListener("click", () => show(button.dataset.mechanicsView))); show(initial);
}

async function renderLLM(view) {
  const usage = view.llm;
  $("llm").innerHTML = `<div class="llm-summary"><div><span class="cell-label">Attempted</span><div class="cell-value">${usage.attempted}</div></div><div><span class="cell-label">Completed</span><div class="cell-value">${usage.completed}</div></div><div><span class="cell-label">Failed / malformed / invalid</span><div class="cell-value">${usage.failed} / ${usage.malformed} / ${usage.invalid}</div></div><div><span class="cell-label">Tokens</span><div class="cell-value">${fmt(usage.total_tokens, 0)}</div></div><div><span class="cell-label">Estimated cost</span><div class="cell-value">${usage.estimated_cost_usd === null ? "Unavailable" : formatMetric(usage.estimated_cost_usd, "currency")}</div></div><div><span class="cell-label">Latency p50 / p95</span><div class="cell-value">${usage.approximate_p50_ms ?? "-"} / ${usage.approximate_p95_ms ?? "-"} ms</div></div></div><p class="hint">Only provider-returned rationale or reasoning summaries can be shown. Hidden chain-of-thought is unavailable.</p><section class="result-section"><div class="section-heading"><h3>Redacted calls</h3><div><button id="llmPrevious" class="icon-button" aria-label="Previous calls">&#8592;</button> <button id="llmNext" class="icon-button" aria-label="Next calls">&#8594;</button></div></div><div id="llmCalls" class="empty compact">Loading calls...</div></section>`;
  state.llmOffset = 0; await loadLLMCalls(view.id);
  $("llmPrevious").addEventListener("click", async () => { state.llmOffset = Math.max(0, state.llmOffset - 50); await loadLLMCalls(view.id); });
  $("llmNext").addEventListener("click", async () => { state.llmOffset += 50; await loadLLMCalls(view.id); });
}

async function loadLLMCalls(runId, live = false) {
  try { const payload = await api(`/api/${live ? "jobs" : "runs"}/${encodeURIComponent(runId)}/llm-calls?limit=50&offset=${state.llmOffset}`); if ($("llmPrevious")) $("llmPrevious").disabled = state.llmOffset === 0; if ($("llmNext")) $("llmNext").disabled = state.llmOffset + payload.calls.length >= payload.total; $("llmCalls").innerHTML = table(["Step", "Agent", "Provider / model", "Status", "Tokens", "Latency", "Action", "Details"], payload.calls.map((call, index) => [call.step ?? "-", call.agent_id ?? "-", `${call.provider || "unknown"} / ${call.model || "unknown"}`, call.status || (call.error ? "failed" : "completed"), call.total_tokens || 0, `${fmt(call.latency_ms || 0, 1)} ms`, call.selected_action || "-", markup(`<details class="call-details"><summary>Inspect</summary>${call.failure_classification || call.error ? `<p class="error">${esc(call.failure_classification || call.error)}</p>` : ""}${call.rationale ? `<p><strong>Returned rationale</strong><br>${esc(call.rationale)}</p>` : ""}${call.reasoning_summary ? `<p><strong>Provider reasoning summary</strong><br>${esc(call.reasoning_summary)}</p>` : ""}${call.prompt ? `<pre>${esc(call.prompt)}</pre>` : ""}${call.content ? `<pre>${esc(call.content)}</pre>` : ""}${call.trace_id ? `<code>trace ${esc(call.trace_id)} / span ${esc(call.span_id || "-")}</code>` : ""}${!call.rationale && !call.reasoning_summary && !call.prompt && !call.content && !call.error ? `<p class="muted">No captured content for this redaction profile.</p>` : ""}</details>`)]), "LLM calls"); } catch (error) { $("llmCalls").innerHTML = `<div class="error">${esc(error.message)}</div>`; }
}

function renderStudy(view) {
  const insights = (view.parameter_insights || []).map((item) => `<div class="finding"><strong>${esc(item.parameter)}</strong><span>${esc(item.text)}${item.winner_value !== null ? ` Winner: ${esc(item.winner_value)}.` : ""}</span></div>`).join("");
  $("overview").innerHTML = `<section class="result-section"><div class="section-heading"><div><span class="eyebrow">What happened</span><h3>Study findings</h3></div></div>${findings(view)}</section><section class="result-section"><div class="section-heading"><div><span class="eyebrow">Evidence</span><h3>Observed parameter ranges</h3></div></div><div class="findings">${insights || `<p class="muted">No numeric sensitivity summary is available.</p>`}</div></section><section class="result-section"><div class="section-heading"><div><span class="eyebrow">Next experiment</span><h3>Use the selected winner</h3></div></div><p>Winner values are selected within this tested search space. Pareto frontier points represent alternatives, not a universal ranking.</p>${Object.keys(view.winner_parameters || {}).length ? `<button type="button" class="primary" data-apply-winner>Apply winner to Setup</button>` : ""}</section><section class="result-section"><div class="section-heading"><h3>Study facts</h3></div>${factsGrid(view.facts)}</section>`;
  const controls = `<div class="grid-four"><div><label for="chartX">X axis</label><select id="chartX">${view.objectives.map((name) => `<option>${esc(name)}</option>`).join("")}</select></div><div><label for="chartY">Y axis</label><select id="chartY">${view.objectives.map((name, index) => `<option ${index === 1 ? "selected" : ""}>${esc(name)}</option>`).join("")}</select></div><label class="check"><input id="feasibleOnly" type="checkbox">Feasible only</label></div><div id="studyPlot">${studyChart(view)}</div>`;
  $("charts").innerHTML = `<section class="result-section"><div class="section-heading"><div><span class="eyebrow">Incentive alignment</span><h3>Proxy versus trusted outcome</h3></div></div>${proxyOutcomeChart(view)}</section><section class="result-section">${controls}</section><section class="result-section"><div class="section-heading"><div><span class="eyebrow">Sensitivity</span><h3>Parameter effects</h3></div></div>${parameterEffects(view)}</section><section class="result-section"><div class="section-heading"><div><span class="eyebrow">Search progress</span><h3>Cumulative best</h3></div></div>${cumulativeBest(view)}</section>`;
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
    $("charts").innerHTML = comparisonBars(loaded.map((item) => item.view)) + comparisonOverlays(loaded.map((item) => item.view));
  } else {
    $("overview").innerHTML = table(["Study", "Pack", "Trials", "Objectives"], loaded.map((item) => [item.view.id, item.view.title, item.view.facts.Trials, item.view.objectives.join(", ")]), "Study comparison");
    $("charts").innerHTML = "";
  }
  document.querySelectorAll(".run-tab,.study-tab,.llm-tab").forEach((node) => node.classList.add("hidden")); selectTab("overview");
}

function selectResultPack() {
  const packId = state.current?.summary?.pack_id || state.current?.job?.request?.pack;
  if (packId && state.packs.some((pack) => pack.id === packId) && $("pack").value !== packId) { $("pack").value = packId; selectPack(); }
}

function prepareStudy(parameters) {
  selectResultPack(); setMode("study");
  document.querySelectorAll("[data-study-param]").forEach((input) => { input.checked = parameters.includes(input.dataset.studyParam); });
  setWorkspace("setup"); document.querySelector("#parameters")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function applyWinnerToSetup() {
  const values = state.current?.view?.winner_parameters || {}, composition = state.current?.view?.composition || []; selectResultPack();
  if (composition.length) state.populationDraft = structuredClone(composition);
  setMode("run");
  Object.entries(values).forEach(([id, value]) => { const input = document.querySelector(`[data-param="${CSS.escape(id)}"]`); if (!input) return; if (input.type === "checkbox") input.checked = Boolean(value); else input.value = value; input.dispatchEvent(new Event("input")); });
  setWorkspace("setup"); document.querySelector("#parameters")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function openEvidence(evidence) {
  if (evidence.startsWith("transition:")) { selectTab("mechanics"); const node = document.querySelector(`[data-mechanic="${CSS.escape(evidence.slice(11))}"]`); node?.focus(); node?.dispatchEvent(new Event("click")); return; }
  if (evidence.startsWith("agent")) { selectTab("agents"); return; }
  if (evidence.startsWith("trial")) { selectTab("trials"); return; }
  if (evidence.startsWith("metric:")) { selectTab("charts"); return; }
  selectTab("overview"); document.querySelector(`[data-evidence-target="${evidence.startsWith("constraint:") ? "constraints" : "metrics"}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
}

function selectTab(name) { state.currentTab = name; document.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === name)); document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === name)); }
function statusClass(status) { if (status === "completed") return "ok"; if (["failed", "cancelled", "interrupted"].includes(status)) return "bad"; return "pending"; }
function setStatus(status) { $("status").textContent = status; $("status").className = `status ${statusClass(status)}`; }
function clearError() { $("formError").textContent = ""; }
function showError(error) { $("formError").textContent = error.message; }
function showStartupError(error) {
  document.querySelectorAll(".workspace").forEach((node) => node.classList.remove("active"));
  $("startupError").classList.remove("hidden");
  const endpoint = error.endpoint ? ` ${error.endpoint} failed.` : "";
  $("startupErrorMessage").textContent = `${endpoint} ${error.message} The server may still be running older Python code while serving newer frontend files.`.trim();
  setStatus("restart required");
}

init();
