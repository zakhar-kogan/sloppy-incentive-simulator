import { esc, table } from "./format.js";

const transitionEvidence = (values) => (values || []).filter((value) => value.startsWith("transition:")).map((value) => value.slice(11));

export function causalFlowGraph(mechanics) {
  const flow = mechanics.causal_flow;
  if (!flow) return `<div class="empty">This pack does not declare an explanatory causal flow.</div>`;
  const width = Math.max(760, flow.stages.length * 240), nodeWidth = 176, nodeHeight = 64, top = 72;
  const byStage = Object.fromEntries(flow.stages.map((stage) => [stage.id, flow.nodes.filter((node) => node.stage === stage.id)]));
  const maxRows = Math.max(...Object.values(byStage).map((nodes) => nodes.length), 1), height = top + maxRows * 112 + 40;
  const transitionCounts = Object.fromEntries(mechanics.transitions.map((item) => [item.id, item.frequency]));
  const positions = {};
  flow.stages.forEach((stage, stageIndex) => { const nodes = byStage[stage.id], x = 30 + stageIndex * ((width - 60) / flow.stages.length); nodes.forEach((node, row) => { positions[node.id] = { x, y: top + row * 112 }; }); });
  const edges = flow.edges.map((edge) => { const source = positions[edge.source], target = positions[edge.target]; if (!source || !target) return ""; const x1 = source.x + nodeWidth, y1 = source.y + nodeHeight / 2, x2 = target.x, y2 = target.y + nodeHeight / 2, midpoint = (x1 + x2) / 2; return `<g class="flow-edge"><path d="M${x1} ${y1} C${midpoint} ${y1},${midpoint} ${y2},${x2} ${y2}" marker-end="url(#arrow)"/>${edge.label ? `<text x="${midpoint}" y="${Math.min(y1, y2) - 7}" text-anchor="middle">${esc(edge.label)}</text>` : ""}</g>`; }).join("");
  const nodes = flow.nodes.map((node) => { const position = positions[node.id], frequency = transitionEvidence(node.evidence).reduce((total, id) => total + (transitionCounts[id] || 0), 0); return `<g class="flow-node kind-${esc(node.kind)}" data-flow-node="${esc(node.id)}" tabindex="0" role="button" aria-label="Inspect ${esc(node.label)}"><rect x="${position.x}" y="${position.y}" width="${nodeWidth}" height="${nodeHeight}"/><text class="flow-label" x="${position.x + nodeWidth / 2}" y="${position.y + 27}" text-anchor="middle">${esc(node.label)}</text><text class="node-meta" x="${position.x + nodeWidth / 2}" y="${position.y + 47}" text-anchor="middle">${frequency ? `${frequency} run events` : esc(node.kind)}</text></g>`; }).join("");
  const stages = flow.stages.map((stage, index) => `<text class="stage-label" x="${30 + index * ((width - 60) / flow.stages.length) + nodeWidth / 2}" y="28" text-anchor="middle">${esc(stage.label)}</text>`).join("");
  return `<div class="mechanics-layout"><div class="mechanics-canvas"><p class="mechanics-note">Explanatory projection of declared mechanics, not execution order.</p><svg class="causal-flow" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(flow.title)}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>${stages}${edges}${nodes}</svg></div><aside id="mechanicsInspector" class="inspector"><span class="eyebrow">Causal flow</span><h3>${esc(flow.title)}</h3><p>${esc(flow.description)}</p></aside></div>`;
}

export function stateMachineGraph(mechanics) {
  if (!mechanics.transitions.length) return `<div class="empty">Mechanics are unavailable for this legacy artifact.</div>`;
  const width = 840, height = Math.max(360, mechanics.states.length * 150), centerX = width / 2, stateWidth = 150, stateHeight = 58;
  const positions = Object.fromEntries(mechanics.states.map((state, index) => [state, { x: centerX - stateWidth / 2, y: 70 + index * 150 }]));
  const states = mechanics.states.map((state) => { const position = positions[state]; return `<g class="state-node"><rect x="${position.x}" y="${position.y}" width="${stateWidth}" height="${stateHeight}"/><text x="${centerX}" y="${position.y + 35}" text-anchor="middle">${esc(state)}</text></g>`; }).join("");
  const transitions = mechanics.transitions.map((item, index) => { const from = positions[item.from_state], to = positions[item.to_state], self = item.from_state === item.to_state; if (self) { const side = index % 2 ? 1 : -1, lane = 90 + Math.floor(index / 2) * 45, x = side > 0 ? from.x + stateWidth : from.x, labelX = side > 0 ? x + lane : x - lane; return `<g class="state-edge" data-mechanic="${esc(item.id)}" tabindex="0" role="button" aria-label="Inspect ${esc(item.label)}"><path d="M${x} ${from.y + 18} C${labelX} ${from.y - 25},${labelX} ${from.y + stateHeight + 25},${x} ${from.y + stateHeight - 18}" marker-end="url(#state-arrow)"/><text x="${labelX}" y="${from.y + stateHeight / 2}" text-anchor="middle">${esc(item.label)} · ${item.frequency}</text></g>`; } const x1 = centerX, y1 = from.y + stateHeight, x2 = centerX, y2 = to.y, offset = (index % 3 - 1) * 110; return `<g class="state-edge" data-mechanic="${esc(item.id)}" tabindex="0" role="button" aria-label="Inspect ${esc(item.label)}"><path d="M${x1} ${y1} C${x1 + offset} ${(y1 + y2) / 2},${x2 + offset} ${(y1 + y2) / 2},${x2} ${y2}" marker-end="url(#state-arrow)"/><text x="${x1 + offset}" y="${(y1 + y2) / 2 - 7}" text-anchor="middle">${esc(item.label)} · ${item.frequency}</text></g>`; }).join("");
  return `<div class="mechanics-layout"><div class="mechanics-canvas"><p class="mechanics-note">Exact executable states and action-labelled transitions.</p><svg class="state-machine" viewBox="0 0 ${width} ${height}" role="img" aria-label="Executable state machine"><defs><marker id="state-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker></defs>${transitions}${states}</svg></div><aside id="mechanicsInspector" class="inspector"><span class="eyebrow">State machine</span><h3>Select a transition</h3><p>This view is the exact executable state machine persisted with the run.</p></aside></div>`;
}

export const mechanicsGraph = stateMachineGraph;

export function mechanicsTable(mechanics) {
  return table(["Transition", "From", "To", "Effects", "Enforcement", "Run events"], mechanics.transitions.map((item) => [item.label, item.from_state, item.to_state, item.effects.join("; ") || "None", item.enforcement.join("; ") || "None", item.frequency]), "Mechanics table");
}

export function bindMechanics(mechanics) {
  const transitions = Object.fromEntries(mechanics.transitions.map((item) => [item.id, item]));
  const flowNodes = Object.fromEntries((mechanics.causal_flow?.nodes || []).map((item) => [item.id, item]));
  const bind = (selector, open) => document.querySelectorAll(selector).forEach((node) => { const activate = () => open(node); node.addEventListener("click", activate); node.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); activate(); } }); });
  bind("[data-mechanic]", (node) => { const item = transitions[node.dataset.mechanic], inspector = document.getElementById("mechanicsInspector"); if (!item || !inspector) return; inspector.innerHTML = `<span class="eyebrow">${esc(item.from_state)} to ${esc(item.to_state)}</span><h3>${esc(item.label)}</h3><p>${esc(item.tags.join(", ") || "No tags")}</p><h4>Effects</h4><ul>${(item.effects.length ? item.effects : ["No declared effects"]).map((value) => `<li>${esc(value)}</li>`).join("")}</ul><h4>Enforcement</h4><ul>${(item.enforcement.length ? item.enforcement : ["None"]).map((value) => `<li>${esc(value)}</li>`).join("")}</ul>`; });
  bind("[data-flow-node]", (node) => { const item = flowNodes[node.dataset.flowNode], inspector = document.getElementById("mechanicsInspector"); if (!item || !inspector) return; inspector.innerHTML = `<span class="eyebrow">${esc(item.kind)}</span><h3>${esc(item.label)}</h3><p>${esc(item.description || "Declared causal mechanism")}</p><h4>Evidence</h4><ul>${item.evidence.map((value) => `<li>${esc(value)}</li>`).join("")}</ul>`; });
}
