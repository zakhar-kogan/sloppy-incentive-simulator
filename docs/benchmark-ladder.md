# Performance Acceptance

The performance target is bounded memory rather than an unqualified events-per-second number. Run the benchmark with:

```bash
python benchmarks/benchmark_training.py --steps 100000
```

It verifies:

- training summaries contain no trace or checkpoint arrays;
- policy history remains within declared visibility bounds;
- retained files do not grow with turns beyond normalized inputs and the final summary;
- symbolic compilation occurs before execution and no solver object is created per turn;
- elapsed time and peak traced memory are printed for regression tracking.

PettingZoo API tests and study process determinism are part of the normal test suite.
