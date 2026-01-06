# MCTS Dama (Italian Checkers)

Monte Carlo Tree Search implementation for Italian Checkers (Dama) with optimized bitboard representation and AlphaZero-style reinforcement learning.

## Features

- **Bitboard Engine**: Efficient 64-bit bitboard representation with **Lookup Tables** for lightning-fast move generation.
- **MCTS AI**: Monte Carlo Tree Search with **Transposition Table (DAG)**, **Solver** (win/loss pruning), and **UCB1-Tuned**.
- **Neural Network**: Deep CNN integration for policy and value prediction, with a fully automated training pipeline.
- **Memory Optimization**: Custom **Arena Allocator** for zero-overhead node allocation and instant tree destruction.
- **Italian Rules**: Full implementation of Italian Checkers rules including mandatory captures and priority rules.
- **Modular Architecture**: Logically separated modules for inference, training, search, and core mechanics.
- **Data & Log Management**: Standardized directory structure for managing multi-iteration datasets and monitoring logs.

## Build

```bash
make          # Build main game CLI (bin/dama)
make gui      # Build graphical interface (bin/game_gui)
make test     # Build and run unit tests
make bench    # Run performance benchmarks
make clean    # Clean build artifacts
```

## Neural Network & Reinforcement Learning

The project implements a complete AlphaZero-style pipeline:

1. **Self-Play**: The network plays against itself to generate policy/value data.

   ```bash
   ./bin/dama selfplay --games 100 --output data/train.bin
   ```

2. **Training**: Train the CNN on the generated data.

   ```bash
   ./bin/dama train --data data/train.bin --epochs 50
   ```

3. **Full Loop**: Automate the entire process using the provided script.

   ```bash
   ./scripts/train_loop.sh
   ```

   This runs a Scaled AlphaZero loop:
   - Self-Play (500 games)
   - Dataset Trim (Window: 1.5M samples)
   - Train (1 Epoch @ 128 batch)
   - Tournament Evaluation & Promotion

For detailed information on the data pipeline, see [docs/architecture.md](docs/architecture.md).

## Project Structure

```
MCTS Dama/
├── src/                      # Source code
│   ├── engine/               # Game logic (4 files)
│   │   ├── game.c            # State management, game_apply_move
│   │   ├── movegen.c         # Move generation with lookup tables
│   │   ├── zobrist.c         # Zobrist hashing for transposition tables
│   │   └── game_view.c       # Board display formatting
│   │
│   ├── common/               # Shared utilities
│   │   └── cli_view.c        # Formatted CLI output
│   │
│   ├── search/               # MCTS engine (6 files, ~1,400 lines)
│   │   ├── mcts_search.c     # Main search loop, async batching
│   │   ├── mcts_selection.c  # UCB1, UCB1-Tuned, PUCT algorithms
│   │   ├── mcts_tree.c       # Node creation, expansion, backprop
│   │   ├── mcts_worker.c     # Thread pool, inference queue
│   │   ├── mcts_rollout.c    # Vanilla rollout policy
│   │   └── mcts_utils.c      # Policy extraction, diagnostics
│   │
│   ├── neural/               # CNN modules (6 files, ~1,000 lines)
│   │   ├── cnn_inference.c   # Forward pass (single & batch)
│   │   ├── cnn_core.c        # Weight init (He/Xavier)
│   │   ├── cnn_batch_norm.c  # Fused BN+ReLU
│   │   ├── cnn_encode.c      # State → tensor encoding
│   │   ├── conv_ops.c        # im2col + sgemm convolutions
│   │   └── cnn_io.c          # Weight save/load
│   │
│   ├── training/             # Training pipeline (6 files, ~1,600 lines)
│   │   ├── cnn_training.c    # Backward pass, SGD optimizer
│   │   ├── selfplay.c        # Self-play data generation
│   │   ├── training_pipeline.c # Epoch loop, LR scheduling
│   │   ├── dataset.c         # Binary dataset I/O
│   │   ├── dataset_analysis.c # Dataset statistics
│   │   └── endgame.c         # Endgame position generator
│   │
│   ├── tournament/           # Tournament system
│   │   └── tournament.c      # Round-robin, ELO calculation
│   │
│   └── tuning/               # Hyperparameter tuning
│       └── clop.c            # CLOP algorithm for tuning
│
├── include/dama/             # Header files
│   ├── common/               # cli_view, debug, error_codes, logging, params, rng
│   ├── engine/               # game.h, movegen.h, zobrist.h, game_view.h
│   ├── neural/               # cnn.h, cnn_types.h, conv_ops.h
│   ├── search/               # mcts.h, mcts_config.h, mcts_types.h
│   ├── tournament/           # tournament.h
│   ├── training/             # dataset.h, selfplay.h
│   └── tuning/               # clop.h, clop_params.h
│
├── apps/                     # Applications
│   ├── cli/                  # Command-line interface
│   │   ├── main.c            # Entry point & command dispatch
│   │   ├── cmd_train.c       # train command
│   │   ├── cmd_data.c        # data inspect/merge/dedupe
│   │   ├── cmd_diagnose.c    # NN diagnostics
│   │   └── cmd_tournament.c  # Tournament launcher
│   └── gui/                  # SDL2 graphical interface
│       └── dama_gui.c        # Board rendering, click handling
│
├── tests/                    # Unit tests (7 test files)
├── docs/                     # Technical documentation (9 docs)
├── scripts/                  # Automation scripts
└── out/                      # Output (data, models, logs)
```

## Usage

### Play a Game

- **CLI**: `./bin/dama`
- **GUI**: `./bin/game_gui`

### Evaluate Performance

Run a round-robin tournament between different MCTS configurations:

```bash
./bin/dama tournament
```

## MCTS Enhancements (Browne et al. 2012)

| Category | Feature | Status |
|----------|---------|--------|
| **Bandit** | UCB1-Tuned | ✅ |
| | PUCT (AlphaZero) | ✅ |
| **Selection** | Transposition Table (DAG) | ✅ |
| | Progressive Bias | ✅ |
| | First Play Urgency (FPU) | ✅ |
| | MCTS-Solver | ✅ |
| **Simulation** | Smart Rollouts | ✅ |
| **Backprop** | Decaying Reward | ✅ |
| **Parallel** | Tree Parallelisation (Virtual Loss) | ✅ |
| | Async CNN Batching | ✅ |

## License

MIT
