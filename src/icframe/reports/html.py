from __future__ import annotations

import json
from html import escape
from pathlib import Path

from icframe.domain.run import RunSummary, StudySummary

from .view_models import run_view_model, study_view_model


def render_html_report(
    value: RunSummary | StudySummary,
) -> str:
    view = run_view_model(value) if isinstance(value, RunSummary) else study_view_model(value)
    payload = json.dumps(view.model_dump(mode="json"), separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{escape(view.title)} / ICFRAME</title>
  <style>
    :root {{ color-scheme: light; --ink:#17202a; --muted:#66707c; --line:#dce1e6;
      --paper:#f5f6f4; --surface:#fff; --green:#1b7355; --red:#b44234; --blue:#2d61a8; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink);
      font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif; }}
    header {{ padding:28px max(24px,calc((100vw - 1180px)/2)); background:var(--surface);
      border-bottom:1px solid var(--line); }}
    h1 {{ margin:3px 0 4px; font-size:28px; letter-spacing:0; }} h2 {{ font-size:16px; margin:0 0 14px; }}
    .eyebrow,.muted {{ color:var(--muted); }} .eyebrow {{ font-size:11px; font-weight:700; text-transform:uppercase; }}
    .status {{ float:right; padding:4px 8px; border:1px solid var(--line); border-radius:4px; font-weight:650; }}
    main {{ max-width:1180px; margin:auto; padding:24px; }} section {{ margin-bottom:28px; }}
    .facts,.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); border:1px solid var(--line); background:var(--surface); }}
    .fact,.metric {{ min-height:82px; padding:14px; border-right:1px solid var(--line); }}
    .fact:last-child,.metric:last-child {{ border-right:0; }} .label {{ color:var(--muted); font-size:12px; }}
    .value {{ margin-top:6px; font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }}
    table {{ width:100%; border-collapse:collapse; background:var(--surface); border:1px solid var(--line); }}
    th,td {{ padding:10px 12px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top; }}
    th {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
    .ok {{ color:var(--green); }} .bad {{ color:var(--red); }} .bar {{ display:flex; align-items:center; gap:10px; margin:8px 0; }}
    .bar span {{ min-width:150px; }} .track {{ flex:1; height:8px; background:#e8ebee; }} .fill {{ height:100%; background:var(--blue); }}
    .chart {{ min-height:260px; padding:12px; border:1px solid var(--line); background:var(--surface); }}
    .chart svg {{ width:100%; height:240px; overflow:visible; }} .axis {{ stroke:#aab2bb; stroke-width:1; }}
    .series {{ fill:none; stroke-width:2; }} .dot {{ stroke:var(--surface); stroke-width:1.5; }}
    .kv {{ display:grid; grid-template-columns:minmax(90px,auto) 1fr; gap:2px 8px; }}
    .kv span:nth-child(odd) {{ color:var(--muted); }}
    @media(max-width:640px) {{ main {{ padding:16px; }} .facts,.metrics {{ grid-template-columns:1fr 1fr; }}
      .fact,.metric {{ border-bottom:1px solid var(--line); }} .bar span {{ min-width:100px; }} }}
  </style>
</head>
<body>
  <header><span id="status" class="status"></span><div class="eyebrow">ICFRAME report</div>
    <h1 id="title"></h1><div id="subtitle" class="muted"></div></header>
  <main id="report"></main>
  <script id="view-model" type="application/json">{payload}</script>
  <script>
    const v=JSON.parse(document.getElementById('view-model').textContent);
    const esc=x=>String(x??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));
    const fmt=x=>Number(x).toLocaleString(undefined,{{maximumFractionDigits:4}});
    const section=(title,body)=>`<section><h2>${{esc(title)}}</h2>${{body}}</section>`;
    const cells=items=>`<div class="metrics">${{items.map(x=>`<div class="metric"><div class="label">${{esc(x.label)}}</div><div class="value">${{fmt(x.value)}}</div></div>`).join('')}}</div>`;
    const facts=()=>`<div class="facts">${{Object.entries(v.facts).map(([k,x])=>`<div class="fact"><div class="label">${{esc(k)}}</div><div class="value">${{esc(x)}}</div></div>`).join('')}}</div>`;
    const kv=value=>`<span class="kv">${{Object.entries(value||{{}}).map(([k,x])=>`<span>${{esc(k.replaceAll('_',' '))}}</span><b>${{esc(typeof x==='number'?fmt(x):x)}}</b>`).join('')}}</span>`;
    const colors=['#2d61a8','#1b7355','#b44234','#8b5a2b','#6b4ca5'];
    const lineChart=points=>{{
      if(!points.length)return '<p class="muted">No checkpoints retained.</p>';
      const names=[...new Set(points.flatMap(p=>Object.keys(p.values)))].slice(0,5), W=900,H=230,P=34;
      const vals=points.flatMap(p=>names.map(n=>p.values[n]).filter(Number.isFinite));
      const lo=Math.min(...vals,0),hi=Math.max(...vals,1),x=i=>P+i*(W-2*P)/Math.max(1,points.length-1),y=n=>H-P-(n-lo)*(H-2*P)/Math.max(1e-9,hi-lo);
      const series=names.map((n,j)=>`<polyline class="series" stroke="${{colors[j]}}" points="${{points.filter(p=>Number.isFinite(p.values[n])).map((p,i)=>`${{x(i)}},${{y(p.values[n])}}`).join(' ')}}"/>`).join('');
      const legend=names.map((n,j)=>`<span style="color:${{colors[j]}}">${{esc(n.replaceAll('_',' '))}}</span>`).join(' &nbsp; ');
      return `<div class="chart"><div class="label">${{legend}}</div><svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="Metrics over steps"><line class="axis" x1="${{P}}" y1="${{H-P}}" x2="${{W-P}}" y2="${{H-P}}"/><line class="axis" x1="${{P}}" y1="${{P}}" x2="${{P}}" y2="${{H-P}}"/>${{series}}</svg></div>`;
    }};
    const objectiveChart=()=>{{
      const names=v.objectives.slice(0,2); if(!v.trials.length||!names.length)return '<p class="muted">No completed trials.</p>';
      const W=900,H=230,P=34,xn=names[0],yn=names[1]||names[0],xs=v.trials.map(t=>t.objectives[xn]),ys=v.trials.map(t=>t.objectives[yn]);
      const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),x=n=>P+(n-minx)*(W-2*P)/Math.max(1e-9,maxx-minx),y=n=>H-P-(n-miny)*(H-2*P)/Math.max(1e-9,maxy-miny);
      const dots=v.trials.map(t=>`<circle class="dot" fill="${{t.feasible?'#1b7355':'#b44234'}}" cx="${{x(t.objectives[xn])}}" cy="${{y(t.objectives[yn])}}" r="5"><title>Trial ${{t.number}}</title></circle>`).join('');
      return `<div class="chart"><div class="label">${{esc(xn)}} vs ${{esc(yn)}}</div><svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="Study objectives"><line class="axis" x1="${{P}}" y1="${{H-P}}" x2="${{W-P}}" y2="${{H-P}}"/><line class="axis" x1="${{P}}" y1="${{P}}" x2="${{P}}" y2="${{H-P}}"/>${{dots}}</svg></div>`;
    }};
    document.getElementById('title').textContent=v.title; document.getElementById('subtitle').textContent=v.subtitle;
    document.getElementById('status').textContent=v.status;
    let html=section('Summary',facts());
    if(v.kind==='run'){{
      html+=section('Trusted objectives',cells(v.objectives)); html+=section('Metrics',cells(v.metrics));
      html+=section('Metrics over time',lineChart(v.checkpoints));
      const max=Math.max(1,...Object.values(v.actions));
      html+=section('Actions',Object.entries(v.actions).map(([k,x])=>`<div class="bar"><span>${{esc(k)}}</span><div class="track"><div class="fill" style="width:${{100*x/max}}%"></div></div><b>${{x}}</b></div>`).join('')||'<p class="muted">No actions retained.</p>');
      html+=section('Constraints',`<table><thead><tr><th>Metric</th><th>Value</th><th>Rule</th><th>Result</th></tr></thead><tbody>${{v.constraints.map(x=>`<tr><td>${{esc(x.metric)}}</td><td>${{fmt(x.value)}}</td><td>${{esc(x.operator)}} ${{fmt(x.threshold)}}</td><td class="${{x.passed?'ok':'bad'}}">${{x.passed?'pass':'fail'}}</td></tr>`).join('')}}</tbody></table>`);
      html+=section('Agents',`<table><thead><tr><th>Agent</th><th>Role</th><th>Policy</th><th>State</th><th>Resources</th></tr></thead><tbody>${{v.agents.map(x=>`<tr><td>${{esc(x.id)}}</td><td>${{esc(x.role)}}</td><td>${{esc(x.policy)}}</td><td>${{esc(x.state)}}</td><td>${{kv(x.resources)}}</td></tr>`).join('')}}</tbody></table>`);
    }}else{{
      html+=section('Objective space',objectiveChart());
      html+=section('Trials',`<table><thead><tr><th>Trial</th><th>Parameters</th><th>Objectives</th><th>Feasible</th></tr></thead><tbody>${{v.trials.map(x=>`<tr><td>${{x.number}}</td><td>${{kv(x.parameters)}}</td><td>${{kv(x.objectives)}}</td><td class="${{x.feasible?'ok':'bad'}}">${{x.feasible?'yes':'no'}}</td></tr>`).join('')}}</tbody></table>`);
    }}
    document.getElementById('report').innerHTML=html;
  </script>
</body></html>"""


def write_html_report(
    artifact: str | Path,
    output: str | Path | None = None,
) -> Path:
    source = Path(artifact)
    summary_path = source / "summary.json" if source.is_dir() else source
    raw = summary_path.read_text()
    payload = json.loads(raw)
    if "run_id" in payload:
        summary: RunSummary | StudySummary = RunSummary.model_validate_json(raw)
    elif "study_id" in payload:
        summary = StudySummary.model_validate_json(raw)
    else:
        raise ValueError(f"{summary_path} is not a v0.4 run or study summary")
    destination = Path(output) if output is not None else summary_path.parent / "report.html"
    destination.write_text(render_html_report(summary), encoding="utf-8")
    return destination
