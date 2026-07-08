from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from math import cos, pi, sin
from pathlib import Path

from icframe.domain.evaluation import EvaluationResult
from icframe.domain.mutations import OptimizationResult
from icframe.domain.provenance import RunProvenance
from icframe.domain.reporting import ExperimentSummary
from icframe.domain.state import SimulationTrace
from icframe.pipelines import build_experiment_summary

_COLOR_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _load_summary(artifact_dir: Path) -> tuple[ExperimentSummary, RunProvenance | None]:
    summary_path = artifact_dir / "summary.json"
    provenance_path = artifact_dir / "provenance.json"
    provenance = None
    if provenance_path.exists():
        provenance = RunProvenance.model_validate_json(provenance_path.read_text())
    if summary_path.exists():
        return ExperimentSummary.model_validate_json(summary_path.read_text()), provenance

    if provenance is None:
        raise FileNotFoundError(f"{artifact_dir} is missing summary.json and provenance.json")

    trace = SimulationTrace.model_validate_json((artifact_dir / "trace.json").read_text())
    evaluation = EvaluationResult.model_validate_json(
        (artifact_dir / "evaluation.json").read_text()
    )
    optimization = None
    optimization_path = artifact_dir / "optimization.json"
    if optimization_path.exists():
        optimization = OptimizationResult.model_validate_json(optimization_path.read_text())
    summary = build_experiment_summary(provenance.run_id, trace, evaluation, optimization)
    return summary, provenance


def _metric_cards(summary: ExperimentSummary) -> str:
    cards = [
        ("Visible score", _fmt(summary.visible_score)),
        ("Trusted score", _fmt(summary.trusted_score)),
        ("Score gap", _fmt(summary.score_gap)),
        ("Total contributions", _fmt(summary.metrics.total_contributions, 1)),
        ("Total payoff", _fmt(summary.metrics.total_payoff)),
        ("Gini", _fmt(summary.metrics.gini)),
        ("Throughput", str(summary.metrics.throughput)),
        ("Reciprocity", _fmt(summary.metrics.graph.reciprocity)),
        ("Collusion index", _fmt(summary.metrics.graph.collusion_index)),
    ]
    return "".join(
        "<div class='card'>"
        f"<div class='label'>{escape(label)}</div>"
        f"<div class='value'>{escape(value)}</div>"
        "</div>"
        for label, value in cards
    )


def _bar_chart(title: str, items: list[tuple[str, float]], color: str = "#4C78A8") -> str:
    width = 520
    height = 240
    padding_left = 46
    padding_bottom = 42
    chart_width = width - padding_left - 20
    chart_height = height - 30 - padding_bottom
    max_value = max((value for _, value in items), default=1.0) or 1.0
    bar_width = chart_width / max(len(items), 1)
    bars: list[str] = []
    for index, (label, value) in enumerate(items):
        scaled = chart_height * (value / max_value)
        x = padding_left + index * bar_width + bar_width * 0.15
        y = 20 + chart_height - scaled
        bars.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width * 0.7:.1f}' height='{scaled:.1f}' fill='{color}' rx='4' />"
        )
        bars.append(
            f"<text x='{x + bar_width * 0.35:.1f}' y='{height - 18}' text-anchor='middle' class='axis'>{escape(label)}</text>"
        )
        bars.append(
            f"<text x='{x + bar_width * 0.35:.1f}' y='{max(y - 6, 12):.1f}' text-anchor='middle' class='value-small'>{_fmt(value)}</text>"
        )
    return (
        f"<section class='chart'><h3>{escape(title)}</h3>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}'>"
        f"<line x1='{padding_left}' y1='20' x2='{padding_left}' y2='{20 + chart_height}' class='axis-line' />"
        f"<line x1='{padding_left}' y1='{20 + chart_height}' x2='{padding_left + chart_width}' y2='{20 + chart_height}' class='axis-line' />"
        + "".join(bars)
        + "</svg></section>"
    )


def _line_chart(title: str, points: list[tuple[str, int, float]], y_label: str) -> str:
    width = 620
    height = 260
    pad_left = 48
    pad_bottom = 36
    chart_width = width - pad_left - 20
    chart_height = height - 24 - pad_bottom
    max_step = max((step for _, step, _ in points), default=1)
    max_value = max((value for _, _, value in points), default=1.0) or 1.0
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for name, step, value in points:
        grouped[name].append((step, value))

    series_markup: list[str] = []
    legend_markup: list[str] = []
    for index, name in enumerate(sorted(grouped)):
        color = _COLOR_PALETTE[index % len(_COLOR_PALETTE)]
        ordered = sorted(grouped[name])
        coords = []
        for step, value in ordered:
            x = pad_left + (chart_width * step / max(max_step, 1))
            y = 20 + chart_height - (chart_height * value / max_value)
            coords.append((x, y))
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        series_markup.append(
            f"<polyline points='{polyline}' fill='none' stroke='{color}' stroke-width='2.5' />"
        )
        series_markup.extend(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='{color}' />" for x, y in coords
        )
        legend_y = 18 + index * 18
        legend_markup.append(
            f"<rect x='{width - 150}' y='{legend_y - 10}' width='12' height='12' fill='{color}' rx='2' />"
            f"<text x='{width - 132}' y='{legend_y}' class='axis'>{escape(name)}</text>"
        )

    x_ticks = "".join(
        f"<text x='{pad_left + chart_width * step / max(max_step, 1):.1f}' y='{height - 12}' text-anchor='middle' class='axis'>{step}</text>"
        for step in range(max_step + 1)
    )
    return (
        f"<section class='chart'><h3>{escape(title)}</h3>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}'>"
        f"<text x='12' y='18' class='axis'>{escape(y_label)}</text>"
        f"<line x1='{pad_left}' y1='20' x2='{pad_left}' y2='{20 + chart_height}' class='axis-line' />"
        f"<line x1='{pad_left}' y1='{20 + chart_height}' x2='{pad_left + chart_width}' y2='{20 + chart_height}' class='axis-line' />"
        + "".join(series_markup)
        + x_ticks
        + "".join(legend_markup)
        + "</svg></section>"
    )


def _network_chart(summary: ExperimentSummary) -> str:
    names = [agent.name for agent in summary.agent_outcomes]
    if not names:
        return ""
    width = 520
    height = 320
    cx = width / 2
    cy = height / 2
    radius = 110
    positions = {
        name: (
            cx + radius * cos((2 * pi * index) / len(names) - pi / 2),
            cy + radius * sin((2 * pi * index) / len(names) - pi / 2),
        )
        for index, name in enumerate(names)
    }
    max_weight = max((edge.weight for edge in summary.graph_edges), default=1.0) or 1.0
    edge_markup = []
    for edge in summary.graph_edges:
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        stroke_width = 1.5 + 4 * (edge.weight / max_weight)
        edge_markup.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#7A8BA3' stroke-width='{stroke_width:.1f}' marker-end='url(#arrow)' opacity='0.85' />"
        )
        edge_markup.append(
            f"<text x='{(x1 + x2) / 2:.1f}' y='{(y1 + y2) / 2 - 6:.1f}' text-anchor='middle' class='value-small'>{_fmt(edge.weight, 1)}</text>"
        )
    node_markup = []
    for index, name in enumerate(names):
        x, y = positions[name]
        color = _COLOR_PALETTE[index % len(_COLOR_PALETTE)]
        node_markup.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='22' fill='{color}' />")
        node_markup.append(
            f"<text x='{x:.1f}' y='{y + 4:.1f}' text-anchor='middle' class='node-label'>{escape(name)}</text>"
        )
    return (
        "<section class='chart'><h3>Interaction graph</h3>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Interaction graph'>"
        "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'><path d='M 0 0 L 10 5 L 0 10 z' fill='#7A8BA3' /></marker></defs>"
        + "".join(edge_markup)
        + "".join(node_markup)
        + "</svg></section>"
    )


def _table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    header_markup = "".join(f"<th>{escape(header)}</th>" for header in headers)
    row_markup = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        f"<section class='table-section'><h3>{escape(title)}</h3>"
        f"<table><thead><tr>{header_markup}</tr></thead><tbody>{row_markup}</tbody></table></section>"
    )


def render_html_report(summary: ExperimentSummary, provenance: RunProvenance | None = None) -> str:
    diagnostics = summary.diagnostics.notes or ["none"]
    event_chart = _bar_chart(
        "Event counts",
        [(key, float(value)) for key, value in summary.event_counts.items()],
        color="#54A24B",
    )
    agent_balance_chart = _bar_chart(
        "Final balances",
        [(agent.name, agent.balance) for agent in summary.agent_outcomes],
        color="#4C78A8",
    )
    agent_payoff_chart = _bar_chart(
        "Final payoffs",
        [(agent.name, agent.payoff) for agent in summary.agent_outcomes],
        color="#F58518",
    )
    balance_line = _line_chart(
        "Balance trajectory",
        [(point.name, point.step, point.balance) for point in summary.agent_series],
        "balance",
    )
    payoff_line = _line_chart(
        "Payoff trajectory",
        [(point.name, point.step, point.payoff) for point in summary.agent_series],
        "payoff",
    )
    network_chart = _network_chart(summary)
    step_rows = [
        [
            str(step.step),
            _fmt(step.total_balance),
            _fmt(step.total_payoff),
            json.dumps(step.event_counts, sort_keys=True),
        ]
        for step in summary.step_summaries
    ]
    agent_rows = [
        [
            agent.name,
            _fmt(agent.balance),
            _fmt(agent.payoff),
            str(agent.contributions),
            str(agent.sent_messages),
            agent.last_action or "",
        ]
        for agent in summary.agent_outcomes
    ]
    graph_rows = [
        [edge.source, edge.target, _fmt(edge.weight, 1), str(edge.event_count)]
        for edge in summary.graph_edges
    ]
    best_params = json.dumps(summary.best_params, sort_keys=True) if summary.best_params else "{}"
    created_at = provenance.created_at.isoformat() if provenance is not None else "UNCONFIRMED"
    return f"""
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <title>ICFRAME report — {escape(summary.run_id)}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f5f7fb; color: #1f2937; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ margin: 8px 0 0; }}
    .hero, .panel, .table-section {{ background: white; border-radius: 16px; box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08); padding: 20px 22px; margin-bottom: 22px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 18px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px; background: #fbfdff; }}
    .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #6b7280; margin-bottom: 6px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }}
    .axis {{ fill: #6b7280; font-size: 11px; }}
    .axis-line {{ stroke: #cbd5e1; stroke-width: 1; }}
    .value-small {{ fill: #374151; font-size: 11px; }}
    .node-label {{ fill: white; font-size: 12px; font-weight: 700; }}
    .pill {{ display: inline-block; margin: 4px 8px 0 0; padding: 6px 10px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
    th {{ color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    code {{ background: #f3f4f6; border-radius: 6px; padding: 2px 6px; }}
  </style>
</head>
<body>
  <main>
    <section class='hero'>
      <h1>ICFRAME experiment report</h1>
      <p><strong>Run:</strong> <code>{escape(summary.run_id)}</code></p>
      <p><strong>Scenario:</strong> {escape(summary.scenario_name)} | <strong>Seed:</strong> {summary.seed} | <strong>Created:</strong> {escape(created_at)}</p>
      <p><strong>Best params:</strong> <code>{escape(best_params)}</code></p>
      <div>{"".join(f"<span class='pill'>{escape(note)}</span>" for note in diagnostics)}</div>
      <div class='cards'>{_metric_cards(summary)}</div>
    </section>
    <div class='grid-2'>
      {event_chart}
      {agent_balance_chart}
      {agent_payoff_chart}
      {network_chart}
      {balance_line}
      {payoff_line}
    </div>
    {_table("Per-agent outcomes", ["Agent", "Balance", "Payoff", "Contrib.", "Sent msgs", "Last action"], agent_rows)}
    {_table("Step summaries", ["Step", "Total balance", "Total payoff", "Event counts"], step_rows)}
    {_table("Graph edges", ["Source", "Target", "Weight", "Events"], graph_rows or [["—", "—", "0.0", "0"]])}
  </main>
</body>
</html>
""".strip()


def write_html_report(artifact_dir: str | Path, output_path: str | Path | None = None) -> Path:
    artifact_path = Path(artifact_dir)
    summary, provenance = _load_summary(artifact_path)
    report_path = Path(output_path) if output_path is not None else artifact_path / "report.html"
    report_path.write_text(render_html_report(summary, provenance))
    return report_path
