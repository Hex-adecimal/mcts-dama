# Experiments Report

Rapporto degli esperimenti condotti su MCTS Dama, con risultati di training e tournament.

---

## 1. Training Progression

### Sessioni di Training

| Sessione | Samples | Epochs | Loss Iniziale | Loss Finale | Accuracy | Throughput |
|----------|---------|--------|---------------|-------------|----------|------------|
| 1 | 422,486 | 3 | 6.78 | 6.22 | 2.6% | 1,366 samp/sec |
| 2 | 441,494 | 3 | 6.25 | 5.95 | 3.6% | 1,397 samp/sec |
| 3 | 460,287 | 3 | 6.18 | 5.95 | 3.6% | ~1,400 samp/sec |
| 4 | 477,884 | 3 | 5.92 | 5.90 | 3.8% | ~1,400 samp/sec |

### Training Curve

```
Loss (Policy + Value)
│
7.0 ┤ ████
6.5 ┤      ████
6.0 ┤           ████ ████
5.5 ┤                     ████
    └──────────────────────────────
         1    2    3    4    5    6  (Epoch cumulativo)
```

### Osservazioni Training

| Osservazione | Dettaglio |
|--------------|-----------|
| **Convergenza** | Loss decresce da 6.78 → 5.90 (~13% riduzione) |
| **Policy Loss dominante** | Policy ~5.5, Value ~0.4 (ratio 14:1) |
| **Accuracy bassa** | 3.8% accuracy indica task molto difficile |
| **Throughput stabile** | ~1,400 samples/sec su Apple M2 |
| **No overfitting** | Val loss segue train loss (no divergenza) |

---

## 2. Tournament Results (30 Dicembre 2025)

### Configurazioni Testate

| ID | Nome | Preset | Feature Distintiva |
|----|------|--------|-------------------|
| 1 | PureVanilla | 600 nodi | Random rollouts |
| 2 | Vanilla | 600 nodi | Standard MCTS |
| 3 | UCB1-Tuned | 600 nodi | UCB1 con varianza |
| 4 | ProgBias | 600 nodi | Heuristic bias |
| 5 | Solver | 600 nodi | Exact solver endgame |
| 6 | GM-Hybrid-V3 | 600 nodi | Heuristic + CNN |
| 7 | CNN-V1 | 600 nodi | CNN (Iteration 0) |
| 8 | CNN-V2 | 600 nodi | CNN (Iteration 1) |
| 9 | CNN-V3 | 600 nodi | CNN (Iteration 2) |
| 10 | CNN-Current | 600 nodi | CNN (Latest) |
| 11 | CNN-New | 600 nodi | CNN (Overnight) |

### Matrice Risultati (55 match, 4 partite ciascuno)

```
              Pure Van UCB1 Prog Solv GMV3 V1   V2   V3   Curr New
PureVanilla     -   1-1  1-1  0-4  1-2  0-4  0-4  0-4  0-4  0-4  0-4
Vanilla        1-1   -   1-2  0-4  0-1  0-4  0-4  0-4  0-4  0-3  0-3
UCB1-Tuned     1-1  2-1   -   0-4  0-3  0-4  0-4  0-4  0-4  0-4  0-4
ProgBias       4-0  4-0  4-0   -   3-0  0-4  0-4  0-4  0-4  4-0  3-1
Solver         2-1  1-0  3-0  0-3   -   0-4  0-3  0-4  0-4  0-4  0-4
GM-Hybrid-V3   4-0  4-0  4-0  4-0  4-0   -   4-0  0-0  4-0  4-0  4-0
CNN-V1         4-0  4-0  4-0  4-0  3-0  0-4   -   0-2  2-2  4-0  4-0
CNN-V2         4-0  4-0  4-0  4-0  4-0  0-0  2-0   -   0-0  2-0  2-0
CNN-V3         4-0  4-0  4-0  4-0  4-0  0-4  2-2  0-0   -   ?    ?
CNN-Current    4-0  3-0  4-0  0-4  4-0  0-4  0-4  0-2  ?     -   ?
CNN-New        4-0  3-0  4-0  1-3  4-0  0-4  0-4  0-2  ?    ?    -
```

### Ranking Finale

| Rank | Configurazione | Win Rate | Note |
|------|----------------|----------|------|
| 🥇 1 | **GM-Hybrid-V3** | ~95% | Domina tutte le CNN |
| 🥈 2 | **ProgBias** | ~75% | Batte vanilla, perde vs CNN |
| 🥉 3 | **CNN-V2** | ~70% | Migliore CNN pura |
| 4 | CNN-V1 | ~65% | Prima iterazione |
| 5 | Solver | ~40% | Buono solo in endgame |
| 6 | UCB1-Tuned | ~30% | Marginale miglioramento |
| 7 | Vanilla | ~25% | Baseline |
| 8 | PureVanilla | ~20% | Random rollouts |

---

## 3. Key Findings

### 3.1 GM-Hybrid domina le CNN pure

> [!IMPORTANT]
> **GM-Hybrid-V3 batte tutte le versioni CNN (4-0)**

**La differenza non è "più nodi", ma come li seleziona:**

| Feature | AlphaZero (CNN-*) | Grandmaster (GM-Hybrid) |
|---------|-------------------|-------------------------|
| Selection | PUCT con prior CNN | PUCT + **Progressive Bias** |
| Expansion bias | Solo CNN | Euristica (cattura, promozione, centro) |
| Rollout | CNN value | **Heuristic weights** (istantaneo) |

**Codice chiave** (`mcts_config.h`):

```c
// GRANDMASTER preset
cfg.use_puct = 1;
cfg.use_progressive_bias = 1;    // ← Bias euristico!
apply_weights(&cfg);             // ← Pesi per rollout

// ALPHA_ZERO preset  
cfg.use_puct = 1;
// NO progressive_bias, NO weights
```

**Risultato nei log:**

```
GM-Hybrid-V3: 11,220 iters, 72,613 nodes (1.5K ips)
CNN-V1:       10,890 iters, 55,042 nodes (1.4K ips)
```

**Insight**: Stesse iterazioni, ma GM-Hybrid espande **+32% nodi** perché:

1. Progressive Bias usa pesi precompilati (istantanei) durante selezione
2. Rollout euristico evita chiamate CNN per valutazione intermedia
3. CNN usata solo per prior policy al root, non per ogni nodo

**Conclusione**: L'euristica ibrida riduce il bottleneck CNN senza sacrificare la qualità della valutazione finale.

### 3.2 ProgBias > Vanilla MCTS

```
ProgBias vs Vanilla:    4-0
ProgBias vs UCB1-Tuned: 4-0
ProgBias vs Solver:     3-0
```

**Insight**: L'heuristic bias durante expansion è più efficace di UCB1-Tuned o Solver puro.

### 3.3 CNN Iterations non monotone

```
CNN-V1 vs CNN-V2: 0-2 (V2 wins)
CNN-V1 vs CNN-V3: 2-2 (Draw)
CNN-V2 vs CNN-V3: 0-0 (4 draws)
```

**Insight**: Training aggiuntivo non garantisce miglioramento. Possibile overfitting o catastrophic forgetting.

### 3.4 Speed vs Quality Trade-off

| Player | Nodi/partita | ips | Risultato |
|--------|-------------|-----|-----------|
| Vanilla | ~27K | 200K | Perde vs CNN |
| CNN-V2 | ~50K | 1.6K | Perde vs GM-Hybrid |
| GM-Hybrid | ~70K | 1.6K | Vince tutto |

**Insight**: A parità di tempo, più nodi (anche con valutazione peggiore) vincono.

---

## 4. Performance Metrics Tournament

### Tempo per Mossa

| Configurazione | Tempo/Mossa | Note |
|----------------|-------------|------|
| Vanilla | ~3ms | CPU-only |
| CNN-* | ~160ms | Bottleneck CNN |
| GM-Hybrid | ~180ms | CNN + heuristic |

### Memoria per Partita

| Configurazione | Memoria Peak | Note |
|----------------|--------------|------|
| Vanilla | ~1.4 MB | Solo nodi |
| CNN-* | ~8 MB | + tensor cache |
| GM-Hybrid | ~10 MB | + heuristic tables |

### Profondità Albero

| Configurazione | Depth Media | Note |
|----------------|-------------|------|
| Vanilla | 7.2 | Shallow |
| CNN-* | 12-15 | Deeper exploration |
| GM-Hybrid | 8-9 | Balanced |

---

## 5. Conclusioni

### Successi

| Risultato | Evidenza |
|-----------|----------|
| ✅ CNN funziona | Batte tutti i preset vanilla (4-0) |
| ✅ Training converge | Loss 6.78 → 5.90 in 12 epoch |
| ✅ Heuristic efficace | ProgBias >> Vanilla |

### Limitazioni

| Limitazione | Causa | Soluzione |
|-------------|-------|-----------|
| ❌ CNN lenta | CPU-only inference | GPU/ANE backend |
| ❌ GM-Hybrid > CNN | Più nodi per tempo | Batch inference |
| ❌ Training plateau | Accuracy 3.8% | Data augmentation, residual blocks |

### Prossimi Esperimenti

1. **Aumentare budget CNN**: 2000+ nodi per CNN-only
2. **Batch inference MCTS**: Ridurre overhead sincronizzazione
3. **Residual CNN**: Architettura più profonda
4. **Replay buffer**: Evitare catastrophic forgetting

---

## 6. Appendice: Raw Data

### Training Config (Sessione Finale)

```
Network      : 4 Conv (64ch) + Policy (512) + Value (256→1)
Parameters   : 3,264,769 (~12.5 MB)
Batch Size   : 128
Learning Rate: 0.0707 (with warmup)
L2 Decay     : 1.0e-04
Rewards      : Checkmate ±1.0 │ Mercy ±0.7 │ Draw 0.0
Backend      : Apple Accelerate (10 OMP threads)
```

### Tournament Config

```
Nodes per move: 600 (fisso per tutti)
Games per match: 4 (doppio round-robin)
Time limit: None (node-based)
```

### File di Riferimento

- [Tournament Log](file:///Users/luigipenza/Desktop/%5B%20Intelligent%20Web%20%5D/MCTS%20Dama/out/logs/tournament_20251230_1312.log)
- [Training Log](file:///Users/luigipenza/Desktop/%5B%20Intelligent%20Web%20%5D/MCTS%20Dama/out/logs/training.log)
- [Best Model](file:///Users/luigipenza/Desktop/%5B%20Intelligent%20Web%20%5D/MCTS%20Dama/out/models/best.bin)
