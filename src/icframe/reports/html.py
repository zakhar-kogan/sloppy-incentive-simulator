from __future__ import annotations

import json
from html import escape
from pathlib import Path

from icframe.domain.incentive_spec import DomainPackManifest, IncentiveSpec
from icframe.domain.run import RunSummary, StudySummary

from .view_models import run_view_model, study_view_model


def render_html_report(
    value: RunSummary | StudySummary,
    *,
    manifest: DomainPackManifest | None = None,
    spec: IncentiveSpec | None = None,
) -> str:
    view = (
        run_view_model(value, manifest, spec)
        if isinstance(value, RunSummary)
        else study_view_model(value, manifest, spec)
    )
    payload = json.dumps(view.model_dump(mode="json"), separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{escape(view.title)} / ICFRAME</title>
  <style>
    :root {{ color-scheme:light; --ink:#17201e; --muted:#66706d; --line:#d3dad5;
      --paper:#f3f5f2; --surface:#fff; --green:#176b4d; --red:#b33b2e; --blue:#245d8f; }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:var(--paper); color:var(--ink);
      font:14px/1.45 "Avenir Next",Avenir,"Segoe UI",sans-serif }}
    header {{ padding:24px max(20px,calc((100vw - 1180px)/2)); border-bottom:1px solid var(--line);
      background:var(--surface) }} h1 {{ margin:3px 0; font-size:26px; letter-spacing:0 }}
    h2 {{ margin:0 0 12px; font-size:15px }} .eyebrow,.muted,.label {{ color:var(--muted) }}
    .eyebrow {{ font-size:10px; font-weight:800; text-transform:uppercase }}
    .status {{ float:right; padding:4px 8px; border:1px solid var(--line); font-weight:700 }}
    main {{ max-width:1180px; margin:auto; padding:24px }} section {{ margin-bottom:30px }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
      border:1px solid var(--line); background:var(--surface) }}
    .cell {{ min-height:86px; padding:14px; border-right:1px solid var(--line) }}
    .value {{ margin-top:6px; font-size:21px; font-weight:700 }} .description {{ font-size:11px; color:var(--muted) }}
    .findings {{ border-top:1px solid var(--line) }} .finding {{ display:grid; grid-template-columns:110px 1fr;
      gap:16px; padding:12px 0; border-bottom:1px solid var(--line) }} .finding b {{ font-size:10px; text-transform:uppercase }}
    .plots {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:10px }}
    .plot {{ padding:12px; border:1px solid var(--line); background:var(--surface) }}
    svg {{ width:100%; height:190px }} .axis {{ stroke:#aeb9b2 }} polyline {{ fill:none; stroke:var(--green); stroke-width:2.5 }}
    svg text {{ fill:var(--muted); font-size:10px }} table {{ width:100%; border-collapse:collapse; background:var(--surface) }}
    th,td {{ padding:9px 11px; border:1px solid var(--line); text-align:left; vertical-align:top }}
    th {{ color:var(--muted); font-size:10px; text-transform:uppercase }} .ok {{ color:var(--green) }} .bad {{ color:var(--red) }}
    code {{ display:block; margin-top:6px; color:var(--blue); font-size:10px }}
    @media(max-width:620px) {{ main {{ padding:15px }} .grid {{ grid-template-columns:1fr 1fr }}
      .finding {{ grid-template-columns:1fr; gap:3px }} .plots {{ grid-template-columns:1fr }} }}
  </style>
</head>
<body>
  <header><span id="status" class="status"></span><div class="eyebrow">ICFRAME report</div>
    <h1 id="title"></h1><div id="subtitle" class="muted"></div></header>
  <main id="report"></main>
  <script id="view-model" type="application/json">{payload}</script>
  <script>
    const v=JSON.parse(document.getElementById('view-model').textContent);
    const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    const fmt=(x,f='number')=>x==null?'Unavailable':f==='percent'?Number(x).toLocaleString(undefined,{{style:'percent',maximumFractionDigits:1}}):f==='currency'?Number(x).toLocaleString(undefined,{{style:'currency',currency:'USD',maximumFractionDigits:4}}):Number(x).toLocaleString(undefined,{{maximumFractionDigits:f==='integer'?0:4}});
    const section=(title,body)=>`<section><h2>${{esc(title)}}</h2>${{body}}</section>`;
    const grid=items=>`<div class="grid">${{items.map(x=>`<div class="cell"><div class="label">${{esc(x.label)}}${{x.cumulative?' / cumulative':''}}</div><div class="value">${{fmt(x.value,x.format)}}</div>${{x.description?`<p class="description">${{esc(x.description)}}</p>`:''}}${{x.formula?`<code>${{esc(x.formula)}}</code>`:''}}</div>`).join('')}}</div>`;
    const facts=()=>`<div class="grid">${{Object.entries(v.facts).map(([k,x])=>`<div class="cell"><div class="label">${{esc(k)}}</div><div class="value">${{esc(x)}}</div></div>`).join('')}}</div>`;
    const findings=()=>`<div class="findings">${{v.findings.map(x=>`<div class="finding"><b>${{esc(x.kind)}}</b><span>${{esc(x.text)}}</span></div>`).join('')}}</div>`;
    const table=(heads,rows)=>`<div style="overflow:auto"><table><thead><tr>${{heads.map(x=>`<th>${{esc(x)}}</th>`).join('')}}</tr></thead><tbody>${{rows.map(r=>`<tr>${{r.map(x=>`<td>${{x&&x.html?x.html:esc(x)}}</td>`).join('')}}</tr>`).join('')}}</tbody></table></div>`;
    const plot=(metric)=>{{ const pts=v.checkpoints.filter(p=>Number.isFinite(p.values[metric.id])); if(!pts.length)return ''; const W=520,H=180,P=34,vals=pts.map(p=>p.values[metric.id]),lo=Math.min(...vals),hi=Math.max(...vals),minS=Math.min(...pts.map(p=>p.step)),maxS=Math.max(...pts.map(p=>p.step)),x=s=>P+(s-minS)*(W-2*P)/Math.max(1,maxS-minS),y=n=>H-P-(n-lo)*(H-2*P)/Math.max(1e-9,hi-lo); return `<article class="plot"><b>${{esc(metric.label)}}</b><svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{esc(metric.label)}} over steps"><line class="axis" x1="${{P}}" y1="${{H-P}}" x2="${{W-P}}" y2="${{H-P}}"/><polyline points="${{pts.map(p=>`${{x(p.step)}},${{y(p.values[metric.id])}}`).join(' ')}}"/><text x="${{P}}" y="${{H-5}}">${{minS}}</text><text text-anchor="end" x="${{W-P}}" y="${{H-5}}">${{maxS}} steps</text></svg></article>`; }};
    document.getElementById('title').textContent=v.title; document.getElementById('subtitle').textContent=v.subtitle; document.getElementById('status').textContent=v.status;
    let html=section('Interpretation',findings())+section(v.kind==='run'?'Run facts':'Study facts',facts());
    if(v.kind==='run'){{
      html+=section('Trusted objectives',grid(v.objectives)); html+=section('Metrics',grid(v.metrics));
      html+=section('Metrics over time',`<div class="plots">${{v.metrics.map(plot).join('')}}</div>`);
      html+=section('Trusted constraints',table(['Metric','Value','Rule','Result'],v.constraints.map(x=>[x.label,fmt(x.value,x.format),`${{x.operator}} ${{fmt(x.threshold,x.format)}}`,{{html:`<span class="${{x.passed?'ok':'bad'}}">${{x.passed?'pass':'fail'}}</span>`}}])));
      if(v.mechanics.causal_flow){{const f=v.mechanics.causal_flow,stages=Object.fromEntries(f.stages.map(x=>[x.id,x.label]));html+=section('Causal flow (explanatory)',`<p class="muted">${{esc(f.description)}} This is not execution order.</p>`+table(['Stage','Mechanism','Kind','Evidence'],f.nodes.map(x=>[stages[x.stage]||x.stage,x.label,x.kind,x.evidence.join('; ')])));}}
      html+=section('Executable state machine',table(['Transition','From','To','Effects','Enforcement','Run events'],v.mechanics.transitions.map(x=>[x.label,x.from_state,x.to_state,x.effects.join('; ')||'None',x.enforcement.join('; ')||'None',x.frequency])));
      html+=section('Agents',table(['Agent','Archetype','Policy','Reward','Failures','Violations','Enforced'],v.agents.map(x=>[x.id,x.archetype,x.policy,fmt(x.reward),x.failed_decisions,x.violations,x.enforcement])));
      if(v.has_llm)html+=section('LLM usage',grid([{{label:'Attempts',value:v.llm.attempted,format:'integer'}},{{label:'Tokens',value:v.llm.total_tokens,format:'integer'}},{{label:'Estimated cost',value:v.llm.estimated_cost_usd,format:'currency'}},{{label:'Approximate p95 latency',value:v.llm.approximate_p95_ms,format:'integer'}}]));
    }}else{{
      if(v.visualizations.length){{const p=v.visualizations[0];html+=section('Proxy versus trusted outcome',table(['Trial',p.x_metric,p.y_metric,p.color_metric||'Feasible'],v.trials.map(x=>[x.number,fmt(x.metrics[p.x_metric]),fmt(x.metrics[p.y_metric]),p.color_metric?fmt(x.metrics[p.color_metric]):(x.feasible?'yes':'no')])));}}
      html+=section('Trials',table(['Trial','Parameters','Objectives','Feasible','State'],v.trials.map(x=>[x.number,JSON.stringify(x.parameters),JSON.stringify(x.objectives),x.feasible?'yes':'no',x.state])));
      html+=section('Retained runs',v.retained_run_ids.length?`<ul>${{v.retained_run_ids.map(x=>`<li>${{esc(x)}}</li>`).join('')}}</ul>`:'<p class="muted">None retained.</p>');
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
    spec_path = summary_path.parent / "spec.json"
    manifest_path = summary_path.parent / "domain_pack_manifest.json"
    if not manifest_path.exists():
        manifest_path = summary_path.parent / "pack.json"
    spec = IncentiveSpec.model_validate_json(spec_path.read_text()) if spec_path.exists() else None
    manifest = (
        DomainPackManifest.model_validate_json(manifest_path.read_text())
        if manifest_path.exists()
        else None
    )
    destination = Path(output) if output is not None else summary_path.parent / "report.html"
    destination.write_text(
        render_html_report(summary, manifest=manifest, spec=spec), encoding="utf-8"
    )
    return destination
