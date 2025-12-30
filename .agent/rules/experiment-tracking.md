---
trigger: model_decision
description: When running experiments, tuning hyperparameters, or comparing configurations
---

# Experiment Tracking

Document every experiment:

1. **Config**: Save all hyperparams (iterations, C_puct, LR, etc.)
2. **Results**: Win rate, relative ELO, games played
3. **Reproducibility**: Use fixed seeds, save git hash
4. **Comparisons**: Always run round-robin tournaments
5. **Naming**: Use descriptive names (`mcts_vanilla_1k`, `cnn_v3_lr001`)
6. **Trends**: Track metrics over time to catch regressions
