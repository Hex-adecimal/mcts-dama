# Project Architecture & Directory Structure

This document provides a detailed overview of the project's organization and the function of various directories and files.

## Directory Overview

### Core Source Code (`src/`)

The engine is written in C99, optimized for performance and modularity.

- **`src/core/`**: The heart of the game engine.
  - `game.c/h`: Board representation, game state management, and rule enforcement.
  - `movegen.c/h`: Lightning-fast move generation using bitboards and lookup tables.
- **`src/mcts/`**: The Monte Carlo Tree Search implementation.
  - `mcts.c/h`: Main MCTS search loop and thread management.
  - `selection.c`: Selection logic (UCB1, PUCT).
  - `expansion.c`: Node expansion logic.
  - `simulation.c`: Rollout/Playout logic.
  - `backprop.c`: Result propagation and node update logic.
  - `mcts_internal.c/h`: Internal utilities including the Arena Allocator and Transposition Table.
- **`src/nn/`**: Neural Network integration.
  - `cnn_core.c`: Memory management, weight initialization, and state encoding.
  - `cnn_inference.c`: Forward pass logic for real-time move evaluation during MCTS.
  - `cnn_training.c`: Backpropagation and SGD optimizer for network learning.
  - `conv_ops.c/h`: Low-level convolution and tensor operations (using Accelerate/BLAS).
  - `dataset.c/h`: Binary dataset I/O and sample management.

### Tools & Utilities (`tools/`)

Functional wrappers around the core engine for various tasks.

- **`tools/training/`**: Tools for the AlphaZero-style learning loop.
  - `bootstrap.c`: Generates initial training data using heuristic MCTS.
  - `selfplay.c`: Generates high-quality data by playing games (CNN vs CNN).
  - `trainer.c`: Trains the network on collected datasets.
- **`tools/evaluation/`**: Assessing agent performance.
  - `tournament.c`: Runs round-robin tournaments between different models or heuristics.
- **`tools/data/`**: Dataset management.
  - `merger.c`: Combines multiple binary dataset files into a single training set.
  - `inspector.c`: Debug tool to visualize samples within a binary dataset.

### Data Management (`data/`)

Standardized storage for training data at different stages.

- **`data/bootstrap/`**: Raw samples generated from heuristic-only play.
- **`data/selfplay/`**: Raw samples generated from neural network self-play.
- **`data/iterations/`**: Consolidated datasets representing specific training iterations.
- **`data/training/`**: The active training (`train.bin`) and validation sets.

### Log Management (`logs/`)

Categorized logs for monitoring the system.

- **`logs/bootstrap/`**: Logs from the initial bootstrap phase.
- **`logs/selfplay/`**: Logs from game generation threads.
- **`logs/training/`**: Detailed training metrics (Loss, Accuracy, LR).
- **`logs/system/`**: Automation logs from `train_loop.sh`.

### Automation & Apps

- **`apps/`**: Main entry points (CLI and Graphical Interface).
- **`scripts/`**: Shell scripts like `train_loop.sh` for automating the full RL cycle.
- **`test/`**: Unit tests for verifying game logic and search correctness.
- **`bin/`**: Destination for all compiled binaries.
