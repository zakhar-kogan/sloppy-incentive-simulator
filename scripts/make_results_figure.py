"""Render the README results figure from a goodhart_audit study's trials.jsonl.

Dependency-free SVG output, matching the palette used by icframe.reports.html.

Usage:
    uv run python scripts/make_results_figure.py <trials.jsonl> <out.svg>
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

INK = "#17201e"
MUTED = "#66706d"
LINE = "#d3dad5"
SERIES = ["#245d8f", "#176b4d", "#b8860b", "#b33b2e"]

PANEL_W = 380
PANEL_H = 250
PAD_L = 62
PAD_T = 52


def load(trials_path: Path) -> dict[int, list[tuple[float, float, float]]]:
    """Return {proxy_agents: [(audit_probability, mean exploit_rate, mean governance_cost)]}."""
    by_agents: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for raw in trials_path.read_text().splitlines():
        if not raw.strip():
            continue
        trial = json.loads(raw)
        seeds = trial["seeds"]
        if not seeds:
            continue
        params = trial["parameters"]

        def mean(key: str, seeds: list = seeds) -> float:
            return statistics.mean(s["metrics"][key] for s in seeds)

        by_agents[int(params["proxy_agents"])].append(
            (float(params["audit_probability"]), mean("exploit_rate"), mean("governance_cost"))
        )
    for points in by_agents.values():
        points.sort()
    return dict(sorted(by_agents.items()))


def panel(
    x0: int,
    data: dict[int, list[tuple[float, float, float]]],
    value_index: int,
    title: str,
    y_max: float,
    y_label: str,
    y_fmt: str,
) -> list[str]:
    """Emit one line-chart panel; value_index selects the metric tuple slot."""
    out = [
        f'<text x="{x0}" y="{PAD_T - 22}" fill="{INK}" font-size="14" '
        f'font-weight="700">{title}</text>',
        f'<text x="{x0}" y="{PAD_T - 6}" fill="{MUTED}" font-size="11">{y_label}</text>',
    ]
    bottom = PAD_T + PANEL_H
    right = x0 + PANEL_W

    for step in range(5):
        y = bottom - step * PANEL_H / 4
        value = y_max * step / 4
        out.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="{LINE}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" fill="{MUTED}" font-size="10" '
            f'text-anchor="end">{value:{y_fmt}}</text>'
        )

    for tick in (0.0, 0.1, 0.3, 0.6, 1.0):
        x = x0 + tick * PANEL_W
        out.append(
            f'<text x="{x:.1f}" y="{bottom + 18}" fill="{MUTED}" font-size="10" '
            f'text-anchor="middle">{tick:g}</text>'
        )
    out.append(
        f'<text x="{x0 + PANEL_W / 2:.1f}" y="{bottom + 38}" fill="{MUTED}" font-size="11" '
        f'text-anchor="middle">audit probability</text>'
    )

    for index, (agents, points) in enumerate(data.items()):
        colour = SERIES[index % len(SERIES)]
        coords = [
            (x0 + audit * PANEL_W, bottom - min(row[value_index], y_max) / y_max * PANEL_H)
            for row in points
            for audit in (row[0],)
        ]
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords)
        )
        out.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.2"/>')
        for x, y in coords:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colour}"/>')
        end_x, end_y = coords[-1]
        out.append(
            f'<text x="{end_x + 6:.1f}" y="{end_y + 4:.1f}" fill="{colour}" font-size="10" '
            f'font-weight="700">{agents}</text>'
        )
    return out


def render(data: dict[int, list[tuple[float, float, float]]]) -> str:
    width = PAD_L + PANEL_W * 2 + 150
    height = PAD_T + PANEL_H + 90
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Avenir Next,Avenir,Segoe UI,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    parts += panel(PAD_L, data, 1, "Exploitation is flat", 1.0, "exploit rate", ".2f")
    parts += panel(
        PAD_L + PANEL_W + 100, data, 2, "Audit cost is not", 120.0, "governance cost", ".0f"
    )
    parts.append(
        f'<text x="{PAD_L}" y="{PAD_T + PANEL_H + 72}" fill="{MUTED}" font-size="11">'
        "line labels = proxy agents; mean of 5 seeds (19, 23, 41, 101, 777)</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    trials_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    data = load(trials_path)
    if not data:
        print(f"no trials found in {trials_path}")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(data))
    print(f"wrote {out_path} ({len(data)} series)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
