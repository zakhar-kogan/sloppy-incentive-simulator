import { esc, fmt, formatMetric } from "./format.js";

function linePlot(points, metric) {
  const values = points.map((point) => point.values[metric.id]).filter(Number.isFinite);
  if (!values.length) return `<div class="empty compact">No retained values.</div>`;
  const width = 520, height = 190, px = 38, py = 24;
  const steps = points.map((point) => point.step);
  const minStep = Math.min(...steps), maxStep = Math.max(...steps);
  const low = Math.min(...values), high = Math.max(...values);
  const x = (step) => px + (step - minStep) * (width - 2 * px) / Math.max(1, maxStep - minStep);
  const y = (value) => height - py - (value - low) * (height - 2 * py) / Math.max(1e-9, high - low);
  const path = points.filter((point) => Number.isFinite(point.values[metric.id])).map((point) => `${x(point.step)},${y(point.values[metric.id])}`).join(" ");
  return `<svg class="mini-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(metric.label)} over steps"><line class="axis" x1="${px}" y1="${height - py}" x2="${width - px}" y2="${height - py}"></line><polyline points="${path}"></polyline><text x="${px}" y="${height - 5}">${minStep}</text><text text-anchor="end" x="${width - px}" y="${height - 5}">${maxStep} steps</text><text x="4" y="${py}">${esc(formatMetric(high, metric.format))}</text><text x="4" y="${height - py}">${esc(formatMetric(low, metric.format))}</text></svg>`;
}

export function smallMultiples(view) {
  if (!view.checkpoints.length) return `<div class="empty">No checkpoints retained for this profile.</div>`;
  const byId = Object.fromEntries(view.metrics.map((metric) => [metric.id, metric]));
  return view.chart_groups.map((group) => `<section class="result-section"><div class="section-heading"><h3>${esc(group.label)}</h3></div><div class="small-multiples">${group.metrics.filter((id) => byId[id]).map((id) => { const metric = byId[id]; return `<article class="plot"><header><strong>${esc(metric.label)}</strong><span>${esc(formatMetric(metric.value, metric.format))}</span></header>${linePlot(view.checkpoints, metric)}</article>`; }).join("")}</div></section>`).join("");
}

export function actionFrequency(points) {
  if (points.length < 2) return `<div class="empty compact">More checkpoints are needed for action frequency.</div>`;
  const names = [...new Set(points.flatMap((point) => Object.keys(point.action_counts || {})))];
  const rows = [];
  for (let index = 1; index < points.length; index += 1) {
    const before = points[index - 1], after = points[index];
    const total = names.reduce((sum, name) => sum + Math.max(0, (after.action_counts[name] || 0) - (before.action_counts[name] || 0)), 0);
    rows.push({ step: after.step, total, values: Object.fromEntries(names.map((name) => [name, Math.max(0, (after.action_counts[name] || 0) - (before.action_counts[name] || 0))])) });
  }
  return `<div class="frequency-list">${rows.map((row) => `<div><span>Step ${row.step}</span><div class="frequency-bar">${names.map((name, index) => { const value = row.values[name]; return value ? `<i class="series-${index % 6}" style="width:${100 * value / Math.max(1, row.total)}%" title="${esc(name)}: ${value}"></i>` : ""; }).join("")}</div><small>${row.total} actions</small></div>`).join("")}</div>`;
}

export function studyChart(view, xName = view.objectives[0], yName = view.objectives[1] || view.objectives[0], feasibleOnly = false) {
  const trials = view.trials.filter((trial) => (!feasibleOnly || trial.feasible) && Number.isFinite(trial.objectives[xName]) && Number.isFinite(trial.objectives[yName]));
  if (!trials.length) return `<div class="empty">No trials match this chart selection.</div>`;
  const width = 760, height = 340, pad = 48;
  const xs = trials.map((trial) => trial.objectives[xName]), ys = trials.map((trial) => trial.objectives[yName]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const x = (value) => pad + (value - minX) * (width - 2 * pad) / Math.max(1e-9, maxX - minX);
  const y = (value) => height - pad - (value - minY) * (height - 2 * pad) / Math.max(1e-9, maxY - minY);
  const dots = trials.map((trial) => `<circle class="trial-dot ${trial.feasible ? "feasible" : "infeasible"} ${trial.winner ? "winner" : ""} ${trial.frontier ? "frontier" : ""}" cx="${x(trial.objectives[xName])}" cy="${y(trial.objectives[yName])}" r="${trial.winner ? 8 : 5}" tabindex="0"><title>Trial ${trial.number}: ${fmt(trial.objectives[xName])}, ${fmt(trial.objectives[yName])}</title></circle>`).join("");
  return `<svg class="study-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Study objective plot"><line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line><line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>${dots}<text text-anchor="end" x="${width - pad}" y="${height - 10}">${esc(xName.replaceAll("_", " "))}</text><text x="8" y="20">${esc(yName.replaceAll("_", " "))}</text></svg>`;
}

export function comparisonBars(views) {
  const names = [...new Set(views.flatMap((view) => view.objectives.map((item) => item.id)))].slice(0, 5);
  return `<div class="comparison-grid">${views.map((view) => `<article><strong>${esc(view.id)}</strong>${names.map((name) => { const metric = view.objectives.find((item) => item.id === name); return `<span><small>${esc(metric?.label || name)}</small><b>${metric ? esc(formatMetric(metric.value, metric.format)) : "-"}</b></span>`; }).join("")}</article>`).join("")}</div>`;
}
