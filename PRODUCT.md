# ICFRAME Product

ICFRAME is a local research workbench for authors who need to design, run, inspect, and
reproduce small incentive experiments. It favors precise evidence over dashboard spectacle.

## Primary User

The primary user is an experiment author working repeatedly with domain packs, seeds,
retention profiles, and bounded studies. Hackathon reviewers are a secondary audience who
should be able to understand a completed artifact without learning internal identifiers.

## v0.4.1 Promise

- Setup and Results are independent workspaces. Inspecting history never changes an
  experiment configuration.
- Results explain what happened using declared metric semantics, trusted constraints,
  checkpoint trends, agent statistics, and exercised mechanics.
- LLM runs expose bounded, redacted per-run usage and failures. Unknown cost is never shown
  as zero.
- Interactive and exported reports derive from the same artifact view models.
- Existing v0.4 domain packs and artifacts remain readable.

## Acceptance

An author can configure and run an experiment or study, follow live progress, open history,
compare compatible artifacts, inspect every relevant result tab, export the same semantic
report, and return to an unchanged Setup workspace. The workflow is keyboard accessible on
desktop and remains functional on a narrow mobile viewport.

## Non-Goals

This release does not add social topology, generic graph composition, Python builders,
distributed orchestration, or cross-run spend management.
