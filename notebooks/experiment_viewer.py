import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd

    from icframe.domain.evaluation import EvaluationResult
    from icframe.domain.mutations import OptimizationResult
    from icframe.domain.provenance import RunProvenance
    from icframe.domain.reporting import ExperimentSummary
    from icframe.domain.state import SimulationTrace
    from icframe.pipelines import build_experiment_summary

    return (
        EvaluationResult,
        ExperimentSummary,
        OptimizationResult,
        Path,
        RunProvenance,
        SimulationTrace,
        build_experiment_summary,
        json,
        mo,
        nx,
        pd,
        plt,
    )


@app.cell
def _(Path):
    artifact_root = Path(".artifacts")
    summary_paths = sorted(artifact_root.glob("**/summary.json")) if artifact_root.exists() else []
    run_options = {path.parent.name: str(path.parent) for path in summary_paths}
    default_run = next(iter(run_options.values()), ".artifacts/demo")
    return artifact_root, default_run, run_options


@app.cell(hide_code=True)
def _(artifact_root, default_run, mo, run_options):
    run_selector = mo.ui.dropdown(
        options=run_options or {"example": default_run},
        value=next(iter(run_options.values()), default_run),
        label="Saved run",
    )
    path_input = mo.ui.text(
        label="Artifact directory",
        value=default_run,
        full_width=True,
    )
    intro = mo.md(
        f"""
        # ICFRAME experiment viewer

        Load a persisted run from `{artifact_root}` or type a different
        artifact directory. The viewer expects `summary.json` when
        available and falls back to raw artifacts for older runs.
        """
    )
    return intro, path_input, run_selector


@app.cell
def _(Path, path_input, run_selector):
    selected_dir = Path(path_input.value or run_selector.value)
    if run_selector.value and path_input.value == ".artifacts/demo":
        selected_dir = Path(run_selector.value)
    return (selected_dir,)


@app.cell
def _(
    EvaluationResult,
    ExperimentSummary,
    OptimizationResult,
    RunProvenance,
    SimulationTrace,
    build_experiment_summary,
    json,
    mo,
    selected_dir,
):
    summary_path = selected_dir / "summary.json"
    if summary_path.exists():
        summary = ExperimentSummary.model_validate_json(summary_path.read_text())
        status = mo.md(f"Loaded summary from `{summary_path}`")
    else:
        provenance_path = selected_dir / "provenance.json"
        trace_path = selected_dir / "trace.json"
        evaluation_path = selected_dir / "evaluation.json"
        if provenance_path.exists() and trace_path.exists() and evaluation_path.exists():
            provenance = RunProvenance.model_validate_json(provenance_path.read_text())
            trace = SimulationTrace.model_validate_json(trace_path.read_text())
            evaluation = EvaluationResult.model_validate_json(evaluation_path.read_text())
            optimization = None
            optimization_path = selected_dir / "optimization.json"
            if optimization_path.exists():
                optimization = OptimizationResult.model_validate_json(optimization_path.read_text())
            summary = build_experiment_summary(
                provenance.run_id,
                trace,
                evaluation,
                optimization,
            )
            status = mo.md(
                f"Loaded raw artifacts from `{selected_dir}` and derived an in-memory summary."
            )
        else:
            summary = None
            status = mo.md(
                "No readable artifacts found in "
                f"`{selected_dir}`. Run `icframe optimize ... --output-dir "
                f"{selected_dir}` first."
            )
    return status, summary


@app.cell(hide_code=True)
def _(mo, summary):
    if summary is None:
        headline = mo.md("No summary available yet.")
    else:
        diagnostics = ", ".join(summary.diagnostics.notes) or "none"
        headline = mo.md(
            f"""
            ## Headline metrics

            - Run: `{summary.run_id}`
            - Scenario: `{summary.scenario_name}`
            - Seed: `{summary.seed}`
            - Visible score: `{summary.visible_score:.3f}`
            - Trusted score: `{summary.trusted_score:.3f}`
            - Score gap: `{summary.score_gap:.3f}`
            - Total contributions: `{summary.metrics.total_contributions:.2f}`
            - Gini: `{summary.metrics.gini:.3f}`
            - Throughput: `{summary.metrics.throughput}`
            - Reciprocity: `{summary.metrics.graph.reciprocity:.3f}`
            - Collusion index: `{summary.metrics.graph.collusion_index:.3f}`
            - Diagnostics: `{diagnostics}`
            """
        )
    return (headline,)


@app.cell
def _(pd, summary):
    if summary is None:
        agents_df = pd.DataFrame()
        event_counts_df = pd.DataFrame()
        step_df = pd.DataFrame()
        edges_df = pd.DataFrame()
    else:
        agents_df = pd.DataFrame(
            [agent.model_dump(mode="json") for agent in summary.agent_outcomes]
        )
        event_counts_df = pd.DataFrame(
            [{"event": event, "count": count} for event, count in summary.event_counts.items()]
        )
        step_df = pd.DataFrame([step.model_dump(mode="json") for step in summary.step_summaries])
        edges_df = pd.DataFrame([edge.model_dump(mode="json") for edge in summary.graph_edges])
    return agents_df, edges_df, event_counts_df, step_df


@app.cell
def _(pd, summary):
    if summary is None:
        agent_series_df = pd.DataFrame()
    else:
        agent_series_df = pd.DataFrame(
            [point.model_dump(mode="json") for point in summary.agent_series]
        )
    return (agent_series_df,)


@app.cell
def _(agents_df, event_counts_df, plt, step_df):
    if agents_df.empty or event_counts_df.empty or step_df.empty:
        fig_agents = None
        fig_events = None
        fig_steps = None
    else:
        fig_agents, _agent_axes = plt.subplots(1, 2, figsize=(10, 4))
        agents_df.plot.bar(
            x="name",
            y="balance",
            ax=_agent_axes[0],
            color="#4C78A8",
            legend=False,
        )
        _agent_axes[0].set_title("Final balances")
        _agent_axes[0].set_ylabel("balance")
        agents_df.plot.bar(
            x="name",
            y="payoff",
            ax=_agent_axes[1],
            color="#F58518",
            legend=False,
        )
        _agent_axes[1].set_title("Final payoffs")
        _agent_axes[1].set_ylabel("payoff")
        fig_agents.tight_layout()

        fig_events, ax_events = plt.subplots(figsize=(7, 4))
        event_counts_df.plot.bar(x="event", y="count", ax=ax_events, color="#54A24B", legend=False)
        ax_events.set_title("Event mix")
        ax_events.set_ylabel("count")
        fig_events.tight_layout()

        fig_steps, ax_steps = plt.subplots(figsize=(8, 4))
        step_df.plot.line(x="step", y=["total_balance", "total_payoff"], marker="o", ax=ax_steps)
        ax_steps.set_title("System trajectory by step")
        ax_steps.set_ylabel("value")
        fig_steps.tight_layout()
    return fig_agents, fig_events, fig_steps


@app.cell
def _(agent_series_df, plt):
    if agent_series_df.empty:
        fig_series = None
    else:
        fig_series, _series_axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, subset in agent_series_df.groupby("name"):
            subset.plot.line(x="step", y="balance", ax=_series_axes[0], marker="o", label=name)
            subset.plot.line(x="step", y="payoff", ax=_series_axes[1], marker="o", label=name)
        _series_axes[0].set_title("Balance trajectory by agent")
        _series_axes[0].set_ylabel("balance")
        _series_axes[1].set_title("Payoff trajectory by agent")
        _series_axes[1].set_ylabel("payoff")
        fig_series.tight_layout()
    return (fig_series,)


@app.cell
def _(edges_df, nx, plt):
    if edges_df.empty:
        fig_graph = None
    else:
        graph = nx.DiGraph()
        for row in edges_df.to_dict(orient="records"):
            graph.add_edge(row["source"], row["target"], weight=row["weight"])

        fig_graph, ax_graph = plt.subplots(figsize=(6, 5))
        positions = nx.spring_layout(graph, seed=7)
        edge_widths = [max(graph[u][v]["weight"], 1.0) for u, v in graph.edges()]
        nx.draw_networkx(
            graph,
            pos=positions,
            ax=ax_graph,
            node_color="#4C78A8",
            font_color="white",
            width=edge_widths,
            arrows=True,
        )
        ax_graph.set_title("Interaction graph")
        ax_graph.axis("off")
        fig_graph.tight_layout()
    return (fig_graph,)


@app.cell(hide_code=True)
def _(agents_df, edges_df, event_counts_df, mo, step_df, summary):
    if summary is None:
        tables = mo.md("Generate a run first to inspect tables.")
    else:
        tables = mo.vstack(
            [
                mo.md("## Tabular outputs"),
                mo.md("### Per-agent outcomes"),
                agents_df,
                mo.md("### Event counts"),
                event_counts_df,
                mo.md("### Step summaries"),
                step_df,
                mo.md("### Graph edges"),
                edges_df,
            ]
        )
    return (tables,)


if __name__ == "__main__":
    app.run()
