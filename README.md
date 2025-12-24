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
make tests    # Build and run unit tests
make tournament # Build and run full tournament
make selfplay # Build training data generator (selfplay)
make bootstrap # Build initial heuristic data generator
make trainer  # Build neural network trainer
make merger   # Build dataset management tool
make clean    # Clean build artifacts
```

## Neural Network & Reinforcement Learning

The project implements a complete AlphaZero-style pipeline:

1. **Bootstrap (Optional)**: Generate initial data from heuristic-only MCTS to kickstart the network.

   ```bash
   make bootstrap && ./bin/bootstrap
   ```

2. **Self-Play**: The network plays against itself to generate improved policy/value data.

   ```bash
   make selfplay && ./bin/selfplay [num_games]
   ```

3. **Training**: Train the CNN on the newly generated data.

   ```bash
   make trainer && ./bin/trainer [data_file.bin]
   ```

4. **Full Loop**: Automate the entire process using the provided script.

   ```bash
   ./scripts/train_loop.sh [iterations] [games_per_iter]
   ```

For detailed information on the data pipeline, see [docs/architecture.md](docs/architecture.md).

## Project Structure

The codebase is organized into modular components for easier maintenance:

- **`src/core/`**: Game logic and rule enforcement.
- **`src/mcts/`**: Modular MCTS engine (Selection, Expansion, Simulation, Backprop).
- **`src/nn/`**: Standardized CNN modules (`cnn_core.c`, `cnn_inference.c`, `cnn_training.c`).
- **`tools/`**: Domain-specific tools for training, evaluation, and data processing.
- **`data/`**: Standardized directory for datasets (`bootstrap/`, `selfplay/`, `iterations/`).
- **`logs/`**: Categorized logs for monitoring the entire pipeline.

For a full breakdown of the files and architecture, refer to the [Architecture Documentation](docs/architecture.md).

## Usage

### Play a Game

- **CLI**: `./bin/dama`
- **GUI**: `./bin/game_gui`

### Evaluate Performance

Run a round-robin tournament between different MCTS configurations:

```bash
make tournament && ./bin/tournament
```

## Roadmap & Enhancements

Structured according to *Browne et al. (2012) MCTS Survey*.

### 1\. Tree Policy Enhancements

#### Bandit Based

- [x] **UCB1-Tuned**: Implement variance-based upper confidence bound to handle nodes with high score variance.
- [ ] **UCB2**: A variant of UCB1 that is asymptotically efficient for K-armed bandits.
- [x] **PUCT (Predictor + UCT)**: Incorporates prior probabilities (from CNN) into the UCB formula.

#### Selection

- [x] **Zobrist Hashing**: Incremental hashing to identify unique board states.
- [x] **Transposition Table (DAG)**: Map identical states reached via different paths.
- [x] **Progressive Bias**: Domain-specific heuristic knowledge added to UCB formula.
- [x] **First Play Urgency (FPU)**: Assign fixed high value to unvisited nodes to encourage exploration.
- [ ] **Decisive Moves**: Immediately play moves that lead to a win or prevent an immediate loss.
- [ ] **Opening Books**: Use expert opening databases or MCTS-generated books.
- [ ] **History Heuristic**: Use historical success of moves to bias selection.

#### AMAF (All Moves As First)

- [ ] **RAVE (Rapid Action Value Estimation)**: Share statistics between tree nodes assuming move value independence.

#### Game Theoretic

- [x] **MCTS-Solver**: Propagate "Proven Win" (+∞) and "Proven Loss" (-∞) values up the tree.
- [ ] **Monte Carlo Proof-Number Search (MC-PNS)**: Use MC simulations to guide Proof-Number Search.

#### Move Pruning

- [ ] **Soft Pruning (Progressive Widening)**: Temporarily prune moves, revisiting them as visits increase.
- [ ] **Hard Pruning**: Permanently remove moves that are statistically inferior.

### 2\. Other Enhancements

#### Simulation

- [x] **Rule-Based Simulation Policy**: Use domain knowledge (captures, promotion) to bias simulations (Smart Rollouts).
- [ ] **Learning a Simulation Policy (MAST/PAST)**: Use global move statistics to bias future simulations.
- [ ] **Last Good Reply (LGR)**: Store successful replies and reuse them in simulations.

#### Backpropagation

- [x] **Transposition Table Updates**: Share statistics between nodes of the same state.
- [x] **Decaying Reward**: Multiply reward by $\gamma < 1$ to weight early wins more heavily.
- [ ] **Score Bonus**: Add bonus for short wins or long losses.

#### Parallelisation

- [ ] **Leaf Parallelisation**: Execute multiple simultaneous simulations from a leaf node.
- [ ] **Root Parallelisation**: Build multiple trees simultaneously and merge results.
- [x] **Tree Parallelisation**: All threads work on a shared tree using **Virtual Loss** and mutexes.
- [x] **Async Batching**: Master thread processes NN inference batches from worker queues.

## License

MIT
