# Benchmark ladder

ICFRAME should graduate scenarios in this order:

1. **Spec conformance**
   - Golden allow/forbid traces for the law engine.
   - Regression worlds with fixed seeds.
2. **Goodhart and reward microbenchmarks**
   - Proxy reward increases while the trusted score drops.
   - Reward mapping loopholes are explicitly represented.
3. **Collusion benchmark**
   - Reciprocal concentration must exceed isolated or scrambled-topology baselines.
4. **System-hacking benchmark**
   - Unauthorized evaluator or provenance tampering is detected.
5. **Domain regressions**
   - Public goods.
   - Insider information or asymmetric-information exchange.
6. **Combined stress tests**
   - Multiple exploit classes active at once.

The repository currently implements the first rung and a compact public-goods microbenchmark that is sufficient to exercise the full pipeline.
