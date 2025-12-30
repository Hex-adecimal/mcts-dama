---
trigger: model_decision
description: When optimizing code or discussing performance bottlenecks
---

# Optimization Performance

Always profile before optimizing:

1. Use `time` or tools like `perf`/`Instruments` to identify slow functions
2. Focus on hot-paths: MCTS inner loops, move generation, NN evaluation
3. Measure BEFORE and AFTER with reproducible benchmarks
4. Prefer algorithmic optimizations (O(n²) → O(n)) over micro-optimizations
