---
trigger: model_decision
description: When creating new files, reorganizing code, or deciding where to put new features
---

# Project Organization

## File Size Balance

| Type | Target Lines | Action if Exceeds |
|------|--------------|-------------------|
| Header (.h) | <150 | Split by responsibility |
| Implementation (.c) | 200-400 | Ideal |
| Implementation (.c) | 400-600 | Consider split if separable |
| Implementation (.c) | >600 | **Must split** |

### When to Split
- ✅ If 2+ distinct responsibilities
- ✅ If feature is self-contained
- ❌ Don't split just to meet line limit if logic is cohesive

---

## Folder Structure

```
src/
├── core/           # Game logic (game, movegen)
├── mcts/           # MCTS: tree, selection, rollout
├── nn/             # Neural network: model, training, inference
├── debug/          # Print/debug functions only
└── params.h        # Global parameters
```

---

## Data Organization

```
out/
├── models/         # NN checkpoints (.bin)
├── data/           # Training datasets (.dat)
├── logs/           # Training/tournament logs
└── results/        # Final results (ELO, stats)
```

### Naming
- Models: `{type}_v{N}.bin` (e.g., `cnn_v3.bin`)
- Datasets: `{source}_{YYYYMMDD}.dat`
- Logs: `{type}_{YYYYMMDD_HHMM}.log`

---

## Anti-Patterns to Avoid
- ❌ Files >800 lines
- ❌ Print functions scattered in logic files
- ❌ Data saved in root or random folders
- ❌ Generic names (`utils.c`, `helpers.c`)
