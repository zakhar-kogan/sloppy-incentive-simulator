# ICFRAME Workbench Design

## Character

The workbench is restrained, light, and utilitarian: a normally lit desktop research tool,
not a marketing page. Dense information uses alignment, rules, typography, and whitespace
instead of nested cards or decorative gradients.

## Structure

- A persistent header switches between `Setup` and `Results`.
- Setup uses a narrow domain rail and a full-width authoring surface with a stable launch bar.
- Results uses a bounded history rail and an unframed artifact surface.
- Run views are `Overview`, `Charts`, `Mechanics`, `Agents`, and conditional `LLM`.
- Study views are `Overview`, `Charts`, `Trials`, and `Retained runs`.

## Visual System

- Warm neutral paper, white data surfaces, dark green primary actions, blue links, and red
  only for failures or infeasibility.
- Avenir Next is preferred where available, with platform sans-serif fallback. Numeric data
  uses tabular figures; artifact identifiers use monospace.
- Controls use 2px corners, tables use compact rows, and individual plots may be framed.
- Motion is limited to progress activity and state transitions. Reduced-motion users receive
  no required animation.

## Data Rules

- Metric labels, units, formats, direction, and grouping come from the persisted domain-pack
  manifest. Generated labels are a compatibility fallback only.
- Metrics with different units never share an axis. Run charts use independent small
  multiples, and action frequencies use checkpoint deltas rather than cumulative counts.
- Findings are deterministic. Good/bad language requires a declared direction, trusted
  constraint, or explicit comparison.
- Mechanics is a projection of the persisted v0.4 spec, not a second runtime graph.
- Prompts, responses, and secrets respect artifact redaction settings in every surface.

## Responsive And Accessible Behavior

Desktop is optimized for repeated authoring. Below 900px, rails become full-width bands;
below 600px, form and chart grids collapse. All tabs, mechanics nodes, tables, filters, and
commands remain keyboard reachable and visibly focused.
