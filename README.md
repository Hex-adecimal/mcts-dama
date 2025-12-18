# MCTS Dama (Italian Checkers)

Monte Carlo Tree Search implementation for Italian Checkers (Dama) with optimized bitboard representation.

## Features

- **Bitboard Engine**: Efficient 64-bit bitboard representation for fast move generation.
- **MCTS AI**: Monte Carlo Tree Search with Smart Rollouts (Epsilon-Greedy heuristics).
- **Memory Optimization**: Custom **Arena Allocator** for zero-overhead node allocation and instant tree destruction.
- **Italian Rules**: Full implementation of Italian Checkers rules including mandatory captures and priority rules.
- **Debug Tools**: ASCII tree visualization, Graphviz (.dot) export, and statistical analysis.
- **Performance Optimized**: Aggressive compiler optimizations (`-O3`, `-march=native`, `-flto`).

## Build

```bash
make          # Build main game
make tests    # Build and run unit tests
make debug    # Build the debug/analysis tool
make clean    # Clean build artifacts
````

## Roadmap & Enhancements (ToDo)

Structured according to *Browne et al. (2012) MCTS Survey*.

### 1\. Tree Policy Enhancements

#### Bandit Based

- [x] **UCB1-Tuned**: Implement variance-based upper confidence bound (Eq. 4 in the survey) to handle nodes with high score variance more aggressively than standard UCB1.

#### Selection

- [x] **Zobrist Hashing**: Implement incremental hashing in `game.c` to identify unique board states.
- [x] **Transposition Table (DAG)**: Map identical states reached via different paths to the same node in memory. This is critical for Dama due to frequent move transpositions.
- [ ] **History Heuristic**: Detect 3-fold repetition to correctly evaluate draw states.

#### AMAF (All Moves As First)

- [ ] **RAVE (Rapid Action Value Estimation)**: *[Low Priority]* Share statistics between tree nodes assuming that a move\'s value is independent of when it is played.

#### Game Theoretic

- [x] **MCTS-Solver**: Propagate "Proven Win" (+∞) and "Proven Loss" (-∞) values up the tree. This allows the MCTS to play perfectly in endgames without requiring infinite simulations.

#### Move Pruning (Domain Knowledge)

- [ ] **Progressive Bias (Soft Pruning)**: Add a heuristic term to the UCB equation based on static evaluation (e.g., piece count). This biases the search toward "obvious" good moves early on, effectively pruning bad branches softy.
- [ ] **Absolute Pruning**: Prune moves that are statistically impossible to become the best move within the remaining time budget.

### 2\. Other Enhancements

#### Simulation (Rollout Policy)

- [x] **Rule Based Simulation (Epsilon-Greedy)**: Implemented `pick_smart_move` to bias rollouts towards promotion, safety, and captures (`ROLLOUT_EPSILON`).
- [ ] **Parameter Tuning**: Use Round Robin Tournaments or CLOP to optimize heuristic weights (`WEIGHT_PROMOTION`, etc.).

#### Backpropagation

- [ ] **Average vs Max**: Experiment with storing the Max score (Minimax style) instead of Average score in nodes, which might be more suitable for tactical games like Checkers.

#### Parallelisation

- [ ] **Root Parallelization**: Run $N$ independent MCTS instances on separate threads (each with its own Arena/Seed) and merge move statistics at the root level.
- [ ] **Leaf Parallelization**: *[Alternative]* Execute multiple rollouts in parallel from the same leaf node.

## Usage

### Play a game (Human vs AI)

```bash
./bin/dama
```

### Analyze a Position (Debug Mode)

Generates statistics for the search tree.

```bash
./bin/debug
```

## Project Structure

```text
├── src/
│   ├── game.c/h       # Core game logic, bitboards, move gen
│   ├── mcts.c/h       # MCTS core, Arena Allocator, Selection/Expansion
│   ├── debug.c/h      # Debug utilities (board printing, move list display)
│   └── params.h       # Hyperparameters and Configuration
├── tools/
│   ├── tuner.c        # SPSA parameter tuning
│   └── tournament.c   # Round-robin tournament system
├── test/
│   └── test_game.c    # Unit tests for rule compliance
├── main.c             # Main CLI game loop
└── Makefile
```

## License

MIT

```
```
