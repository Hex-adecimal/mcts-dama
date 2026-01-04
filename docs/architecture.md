# MCTS Dama Architecture

This document describes the modular architecture of the MCTS Dama project.

---

## Module Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         apps/                                   │
│   ┌─────────┐   ┌─────────┐   ┌───────────┐   ┌──────────────┐  │
│   │   CLI   │   │   GUI   │   │ Tournament│   │ CLOP Worker  │  │
│   └────┬────┘   └────┬────┘   └─────┬─────┘   └──────┬───────┘  │
└────────┼─────────────┼──────────────┼────────────────┼──────────┘
         │             │              │                │
         └─────────────┴──────┬───────┴────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                         src/                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │   engine/    │◄──│   search/    │──►│   neural/    │         │
│  │ Game Logic   │   │    MCTS      │   │     CNN      │         │
│  └──────────────┘   └──────────────┘   └──────┬───────┘         │
│                            ▲                   │                 │
│                            │                   ▼                 │
│                     ┌──────┴───────────────────────┐            │
│                     │         training/            │            │
│                     │   Self-play, Dataset, Train  │            │
│                     └──────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### `src/engine/` (4 files, ~900 lines)

Game logic and move generation (Italian Checkers rules).

| File | Lines | Description |
|------|-------|-------------|
| `game.c` | 140 | GameState, Zobrist hashing, apply_move |
| `movegen.c` | 365 | Legal move generation with bitboard LUTs |
| `endgame.c` | 196 | Endgame position generation for training |
| `cli_view.c` | 199 | Formatted CLI output (box-style headers) |

### `src/search/` (6 files, ~1,400 lines)

Monte Carlo Tree Search implementation.

| File | Lines | Description |
|------|-------|-------------|
| `mcts_search.c` | 404 | Main search loop, async CNN batching |
| `mcts_selection.c` | 174 | **UCB1, UCB1-Tuned, PUCT, select_promising_node** |
| `mcts_tree.c` | 273 | Node creation, expansion, backpropagation |
| `mcts_worker.c` | 242 | Thread pool, inference queue |
| `mcts_rollout.c` | 147 | Vanilla rollout/simulation policy |
| `mcts_utils.c` | 191 | Root creation, policy extraction, debug |

### `src/neural/` (6 files, ~1,000 lines)

Convolutional Neural Network for inference.

| File | Lines | Description |
|------|-------|-------------|
| `cnn_inference.c` | 371 | Forward pass (single + batch, BLAS sgemm) |
| `cnn_core.c` | 163 | Weight init (He/Xavier), memory management |
| `cnn_batch_norm.c` | 121 | **Fused BN+ReLU forward pass** |
| `cnn_encode.c` | 79 | **GameState → tensor canonical encoding** |
| `conv_ops.c` | 223 | im2col + sgemm convolutions, backward |
| `cnn_io.c` | 55 | Weight save/load |

### `src/training/` (5 files, ~1,400 lines)

AlphaZero-style training pipeline.

| File | Lines | Description |
|------|-------|-------------|
| `cnn_training.c` | 414 | Backprop, SGD with momentum, gradient clipping |
| `selfplay.c` | 366 | Self-play generation (Dirichlet, temperature) |
| `training_pipeline.c` | 219 | Training loop with LR scheduling |
| `dataset.c` | 220 | Binary dataset I/O |
| `dataset_analysis.c` | 153 | Dataset statistics and duplicate detection |

---

## Header Organization

### `include/dama/search/` (8 files)

| Header | Description |
|--------|-------------|
| `mcts.h` | Public MCTS API |
| `mcts_config.h` | **MCTSConfig, MCTSPreset, mcts_get_preset()** |
| `mcts_types.h` | Node struct, SolverStatus |
| `mcts_tt.h` | **TranspositionTable operations** |
| `mcts_tree.h` | Tree operation declarations |
| `mcts_worker.h` | **Worker thread types, InferenceQueue** |
| `mcts_arena.h` | Arena allocator |
| `tournament.h` | Tournament types |

### `include/dama/neural/` (3 files)

| Header | Description |
|--------|-------------|
| `cnn.h` | Public CNN API |
| `cnn_types.h` | **CNNWeights, CNNOutput, architecture constants** |
| `conv_ops.h` | Convolution operations |

---

## Data Flow

```
1. Self-Play Generation
   selfplay.c → MCTS search → CNN inference → game samples → .bin files

2. Training
   .bin files → dataset.c → cnn_training.c → updated weights → .weights file

3. Evaluation
   tournament.c → MCTS configs → game results → ELO statistics
```

---

## Key Design Decisions

1. **Arena Allocator**: Zero-overhead node allocation, instant tree reset
2. **Bitboard Engine**: 64-bit representation with lookup tables
3. **Canonical Input**: Board flipped for Black's turn (consistent perspective)
4. **Separate LRs**: Policy and value heads use different learning rates
5. **Thread-safe RNG**: Custom xorshift-based RNG for parallel code
6. **Modular Headers**: Configuration, types, and TT split into separate files
7. **Fused Operations**: BN+ReLU fused for inference efficiency
