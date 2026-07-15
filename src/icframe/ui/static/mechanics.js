import { esc, table } from "./format.js";

export function mechanicsGraph(mechanics) {
  if (!mechanics.transitions.length) return `<div class="empty">Mechanics are unavailable for this legacy artifact.</div>`;
  const width = 760;
  const rowHeight = 104;
  const height = Math.max(220, mechanics.transitions.length * rowHeight + 50);
  const maximum = Math.max(1, ...mechanics.transitions.map((item) => item.frequency));
  const rows = mechanics.transitions.map((item, index) => {
    const y = 40 + index * rowHeight;
    const opacity = 0.25 + 0.75 * item.frequency / maximum;
    const detail = [...item.effects, ...item.enforcement].join("; ") || "No declared effects";
    return `<g class="mechanic-row" data-mechanic="${esc(item.id)}" tabindex="0" role="button" aria-label="Inspect ${esc(item.label)}"><path d="M125 ${y + 24} H245"/><path d="M425 ${y + 24} H535"/><rect x="15" y="${y}" width="110" height="48"/><text x="70" y="${y + 29}" text-anchor="middle">${esc(item.from_state)}</text><rect class="action-node" style="--heat:${opacity}" x="245" y="${y - 5}" width="180" height="58"/><text x="335" y="${y + 20}" text-anchor="middle">${esc(item.label)}</text><text class="node-meta" x="335" y="${y + 40}" text-anchor="middle">${item.frequency} run events</text><rect x="535" y="${y}" width="210" height="48"/><text x="547" y="${y + 20}">${esc(item.to_state)}</text><text class="node-meta" x="547" y="${y + 39}">${esc(detail.slice(0, 30))}</text></g>`;
  }).join("");
  return `<div class="mechanics-layout"><div class="mechanics-canvas"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="State, action, and outcome mechanics">${rows}</svg></div><aside id="mechanicsInspector" class="inspector"><span class="eyebrow">Inspector</span><h3>Select a transition</h3><p>Use the graph or the accessible table to inspect declared effects and enforcement.</p></aside></div>`;
}

export function mechanicsTable(mechanics) {
  return table(
    ["Transition", "From", "To", "Effects", "Enforcement", "Run events"],
    mechanics.transitions.map((item) => [item.label, item.from_state, item.to_state, item.effects.join("; ") || "None", item.enforcement.join("; ") || "None", item.frequency]),
    "Mechanics table",
  );
}

export function bindMechanics(mechanics) {
  const byId = Object.fromEntries(mechanics.transitions.map((item) => [item.id, item]));
  document.querySelectorAll("[data-mechanic]").forEach((node) => {
    const open = () => {
      const item = byId[node.dataset.mechanic];
      const inspector = document.getElementById("mechanicsInspector");
      if (!item || !inspector) return;
      inspector.innerHTML = `<span class="eyebrow">${esc(item.from_state)} to ${esc(item.to_state)}</span><h3>${esc(item.label)}</h3><p>${esc(item.tags.join(", ") || "No tags")}</p><h4>Effects</h4><ul>${(item.effects.length ? item.effects : ["No declared effects"]).map((value) => `<li>${esc(value)}</li>`).join("")}</ul><h4>Enforcement</h4><ul>${(item.enforcement.length ? item.enforcement : ["None"]).map((value) => `<li>${esc(value)}</li>`).join("")}</ul>`;
    };
    node.addEventListener("click", open);
    node.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); open(); } });
  });
}
