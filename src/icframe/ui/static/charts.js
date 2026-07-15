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

export function proxyOutcomeChart(view) {
  const declared = view.visualizations?.find((item) => view.trials.some((trial) => Number.isFinite(trial.metrics?.[item.x_metric]) && Number.isFinite(trial.metrics?.[item.y_metric])));
  const presentations = Object.fromEntries((view.metric_presentations || []).map((item) => [item.id, item]));
  const proxy = (view.metric_presentations || []).find((item) => item.category === "proxy");
  const outcome = (view.metric_presentations || []).find((item) => item.category === "outcome");
  const xName = declared?.x_metric || proxy?.id, yName = declared?.y_metric || outcome?.id;
  if (!xName || !yName) return `<div class="empty">This domain does not declare a proxy/outcome metric pair.</div>`;
  const trials = view.trials.filter((trial) => Number.isFinite(trial.metrics?.[xName]) && Number.isFinite(trial.metrics?.[yName]));
  if (!trials.length) return `<div class="empty">No complete trial metrics are available for this proxy/outcome view.</div>`;
  const width = 760, height = 340, pad = 52, xs = trials.map((trial) => trial.metrics[xName]), ys = trials.map((trial) => trial.metrics[yName]);
  const domain = (values) => { const low = Math.min(...values), high = Math.max(...values), margin = Math.max((high - low) * 0.08, Math.abs(high || 1) * 0.02); return [low - margin, high + margin]; };
  const [minX, maxX] = domain(xs), [minY, maxY] = domain(ys);
  const x = (value) => pad + (value - minX) * (width - 2 * pad) / Math.max(1e-9, maxX - minX), y = (value) => height - pad - (value - minY) * (height - 2 * pad) / Math.max(1e-9, maxY - minY);
  const dots = trials.map((trial) => `<circle class="trial-dot ${trial.feasible ? "feasible" : "infeasible"} ${trial.winner ? "winner" : ""}" cx="${x(trial.metrics[xName])}" cy="${y(trial.metrics[yName])}" r="${trial.winner ? 8 : 5}" tabindex="0"><title>Trial ${trial.number}: ${presentations[xName]?.label || xName} ${fmt(trial.metrics[xName])}; ${presentations[yName]?.label || yName} ${fmt(trial.metrics[yName])}</title></circle>`).join("");
  const axisLabel = (name) => `${presentations[name]?.label || name}${presentations[name]?.unit ? ` (${presentations[name].unit})` : ""}`;
  return `<svg class="study-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Proxy incentive versus trusted outcome"><line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line><line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>${dots}<text text-anchor="end" x="${width - pad}" y="${height - 12}">${esc(axisLabel(xName))}</text><text x="8" y="20">${esc(axisLabel(yName))}</text></svg>`;
}

export function parameterEffects(view, objective = view.objectives[0]) {
  const rows = (view.parameter_insights || []).map((insight) => {
    const groups = new Map();
    view.trials.filter((trial) => trial.feasible && Number.isFinite(trial.objectives[objective]) && Number.isFinite(trial.parameters[insight.parameter])).forEach((trial) => { const key = trial.parameters[insight.parameter]; groups.set(key, [...(groups.get(key) || []), trial.objectives[objective]]); });
    const points = [...groups].sort((a, b) => Number(a[0]) - Number(b[0])).map(([value, values]) => ({ value: Number(value), result: values.reduce((sum, item) => sum + item, 0) / values.length }));
    if (points.length < 2) return "";
    const width = 340, height = 170, pad = 34, xs = points.map((point) => point.value), ys = points.map((point) => point.result), minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys), x = (value) => pad + (value - minX) * (width - 2 * pad) / Math.max(1e-9, maxX - minX), y = (value) => height - pad - (value - minY) * (height - 2 * pad) / Math.max(1e-9, maxY - minY);
    return `<article class="plot"><header><strong>${esc(insight.parameter.replaceAll("_", " "))}</strong><span>${esc(objective.replaceAll("_", " "))}</span></header><svg class="mini-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Mean ${esc(objective)} by ${esc(insight.parameter)}"><line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line><polyline points="${points.map((point) => `${x(point.value)},${y(point.result)}`).join(" ")}"></polyline>${points.map((point) => `<circle class="trial-dot feasible" cx="${x(point.value)}" cy="${y(point.result)}" r="4"><title>${point.value}: ${fmt(point.result)}</title></circle>`).join("")}</svg></article>`;
  }).filter(Boolean);
  return rows.length ? `<div class="small-multiples">${rows.join("")}</div>` : `<div class="empty">At least two numeric parameter levels are needed.</div>`;
}

export function cumulativeBest(view, objective = view.objectives[0]) {
  const presentation = (view.objective_presentations || []).find((item) => item.id === objective), minimize = presentation?.desired_direction === "minimize";
  const trials = [...view.trials].sort((a, b) => a.number - b.number).filter((trial) => trial.feasible && Number.isFinite(trial.objectives[objective]));
  if (!trials.length) return `<div class="empty">No feasible objective values are available.</div>`;
  let best = trials[0].objectives[objective]; const points = trials.map((trial) => { best = minimize ? Math.min(best, trial.objectives[objective]) : Math.max(best, trial.objectives[objective]); return { number: trial.number, value: best }; });
  const width = 760, height = 230, pad = 42, values = points.map((point) => point.value), low = Math.min(...values), high = Math.max(...values), x = (value) => pad + value * (width - 2 * pad) / Math.max(1, points.length - 1), y = (value) => height - pad - (value - low) * (height - 2 * pad) / Math.max(1e-9, high - low);
  return `<svg class="study-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Cumulative best ${esc(objective)}"><line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line><polyline points="${points.map((point, index) => `${x(index)},${y(point.value)}`).join(" ")}"></polyline><text x="${pad}" y="20">Cumulative best · ${esc(objective.replaceAll("_", " "))}</text></svg>`;
}

export function comparisonBars(views) {
  const names = [...new Set(views.flatMap((view) => view.objectives.map((item) => item.id)))].slice(0, 5);
  return `<div class="comparison-scales">${names.map((name) => { const rows = views.map((view) => ({ view, metric: view.objectives.find((item) => item.id === name) })).filter((row) => row.metric); const values = rows.map((row) => row.metric.value), low = Math.min(0, ...values), high = Math.max(...values); return `<article><header><strong>${esc(rows[0]?.metric.label || name)}</strong><span>Shared scale ${esc(formatMetric(low, rows[0]?.metric.format))} to ${esc(formatMetric(high, rows[0]?.metric.format))}</span></header>${rows.map((row, index) => `<div class="comparison-row"><small>${esc(row.view.id)}</small><i class="series-${index % 6}" style="width:${100 * (row.metric.value - low) / Math.max(1e-9, high - low)}%"></i><b>${esc(formatMetric(row.metric.value, row.metric.format))}</b></div>`).join("")}</article>`; }).join("")}</div>`;
}

export function comparisonOverlays(views) {
  const common = views[0]?.metrics.filter((metric) => views.every((view) => view.metrics.some((item) => item.id === metric.id && item.unit === metric.unit))) || [];
  const selected = common.filter((metric) => views.some((view) => view.checkpoints.some((point) => Number.isFinite(point.values[metric.id])))).slice(0, 4);
  if (!selected.length) return "";
  return `<section class="result-section"><div class="section-heading"><div><span class="eyebrow">Shared domains</span><h3>Run overlays</h3></div></div><div class="small-multiples">${selected.map((metric) => { const series = views.map((view) => ({ view, points: view.checkpoints.filter((point) => Number.isFinite(point.values[metric.id])) })).filter((item) => item.points.length); const all = series.flatMap((item) => item.points), values = all.map((point) => point.values[metric.id]), steps = all.map((point) => point.step), low = Math.min(...values), high = Math.max(...values), minStep = Math.min(...steps), maxStep = Math.max(...steps), width = 360, height = 180, pad = 34, x = (value) => pad + (value - minStep) * (width - 2 * pad) / Math.max(1, maxStep - minStep), y = (value) => height - pad - (value - low) * (height - 2 * pad) / Math.max(1e-9, high - low); return `<article class="plot"><header><strong>${esc(metric.label)}</strong><span>${esc(metric.unit || metric.format)}</span></header><svg class="mini-chart overlay-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Compared ${esc(metric.label)} on a shared scale"><line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line>${series.map((item, index) => `<polyline class="overlay-${index % 6}" points="${item.points.map((point) => `${x(point.step)},${y(point.values[metric.id])}`).join(" ")}"><title>${esc(item.view.id)}</title></polyline>`).join("")}</svg></article>`; }).join("")}</div></section>`;
}
