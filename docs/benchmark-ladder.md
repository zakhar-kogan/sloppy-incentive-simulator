# Performance Acceptance

The performance target is bounded memory rather than an unqualified events-per-second number. Run the benchmark with:

```bash
python benchmarks/benchmark_training.py --steps 100000
```

It verifies:

- training summaries contain no trace or checkpoint arrays;
- policy history remains within declared visibility bounds;
- elapsed time and peak traced memory are printed for regression tracking.

Artifact-retention bounds, PettingZoo API compatibility, symbolic compilation, and
study-process determinism are covered by the normal test suite.
