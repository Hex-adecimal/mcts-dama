---
trigger: model_decision
description: When running experiments, tuning hyperparameters, or comparing configurations
---

Documenta ogni esperimento:

1. **Config**: salva tutti gli iperparametri (iterations, C_puct, learning rate, etc.)
2. **Risultati**: win rate, ELO relativo, games giocati
3. **Riproducibilità**: usa seed fissi, salva versione del codice (git hash)
4. **Confronti**: esegui sempre round-robin tournament tra configurazioni
5. **Naming**: usa nomi descrittivi per config (es: `mcts_vanilla_1k`, `cnn_v3_lr001`)
6. **Trend**: traccia le metriche nel tempo per identificare regressioni
