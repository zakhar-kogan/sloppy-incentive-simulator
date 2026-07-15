export const $ = (id) => document.getElementById(id);

export const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
}[char]));

export const fmt = (value, maximumFractionDigits = 4) => Number(value).toLocaleString(
  undefined,
  { maximumFractionDigits },
);

export function formatMetric(value, format = "number") {
  if (value === null || value === undefined) return "Unavailable";
  if (format === "percent") return Number(value).toLocaleString(undefined, { style: "percent", maximumFractionDigits: 1 });
  if (format === "currency") return Number(value).toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 4 });
  if (format === "integer") return fmt(value, 0);
  if (format === "duration") return `${fmt(value, 2)}s`;
  return fmt(value);
}

export const markup = (html) => ({ html });

export function keyValues(value) {
  const entries = Object.entries(value || {});
  return entries.length
    ? `<dl class="key-values">${entries.map(([key, item]) => `<dt>${esc(key.replaceAll("_", " "))}</dt><dd>${esc(typeof item === "number" ? fmt(item) : item)}</dd>`).join("")}</dl>`
    : `<span class="muted">None</span>`;
}

export function table(headers, rows, label = "Data table") {
  const cellValue = (value) => value && typeof value === "object" && "html" in value ? value.html : esc(value);
  return `<div class="table-wrap"><table aria-label="${esc(label)}"><thead><tr>${headers.map((header) => `<th>${esc(header)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((value) => `<td>${cellValue(value)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

export function metricGrid(items) {
  return `<div class="metric-grid">${items.map((item) => `<div class="metric-cell"><div class="cell-label">${esc(item.label)}${item.cumulative ? " / cumulative" : ""}</div><div class="cell-value">${esc(formatMetric(item.value, item.format))}</div>${item.description ? `<p>${esc(item.description)}</p>` : ""}${item.formula ? `<code>${esc(item.formula)}</code>` : ""}</div>`).join("")}</div>`;
}

export function factsGrid(facts) {
  return `<div class="fact-grid">${Object.entries(facts).map(([label, value]) => `<div class="fact-cell"><div class="cell-label">${esc(label)}</div><div class="cell-value">${esc(value)}</div></div>`).join("")}</div>`;
}
