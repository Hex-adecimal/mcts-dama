---
trigger: model_decision
description: When creating new files, reorganizing code, or deciding where to put new features
---

# Project Organization

## File Size Guidelines

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

## Current Folder Structure

```
src/
├── engine/         # Game logic (game.c, movegen.c, endgame.c)
├── search/         # MCTS (mcts_search.c, mcts_tree.c, mcts_selection.c, etc.)
├── neural/         # CNN (cnn_core.c, cnn_inference.c, conv_ops.c, etc.)
├── training/       # Training (cnn_training.c, selfplay.c, dataset.c)
├── tournament/     # Tournament runner
├── tuning/         # CLOP hyperparameter tuning
└── common/         # Shared utilities (cli_view.c)

include/dama/
├── engine/         # Engine headers
├── search/         # MCTS headers (mcts.h, mcts_types.h)
├── neural/         # CNN headers (cnn.h, conv_ops.h)
├── training/       # Training headers
├── tournament/     # Tournament headers
├── tuning/         # Tuning headers
└── common/         # Shared headers (params.h, error_codes.h, debug.h, rng.h)

apps/cli/           # CLI application
tests/              # Unit tests and benchmarks
scripts/            # Shell scripts (train_loop.sh)
docs/               # Documentation
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

### Naming Conventions
- Models: `cnn_v{N}.bin` (e.g., `cnn_v3.bin`)
- Datasets: `selfplay_{YYYYMMDD}.dat`
- Logs: `{type}_{YYYYMMDD_HHMM}.log`

---

## Anti-Patterns
- ❌ Files >800 lines
- ❌ Print functions scattered in logic files
- ❌ Data saved in root or random folders
- ❌ Generic names (`utils.c`, `helpers.c`)
