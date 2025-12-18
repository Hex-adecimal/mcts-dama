# MCTS Dama (Italian Checkers)

Monte Carlo Tree Search implementation for Italian Checkers (Dama) with optimized bitboard representation.

## Features

- **Bitboard Engine**: Efficient 64-bit bitboard representation with **Lookup Tables** for lightning-fast move generation.
- **MCTS AI**: Monte Carlo Tree Search with **Transposition Table (DAG)**, **Solver** (win/loss pruning), and **UCB1-Tuned**.
- **Memory Optimization**: Custom **Arena Allocator** for zero-overhead node allocation and instant tree destruction.
- **Italian Rules**: Full implementation of Italian Checkers rules including mandatory captures and priority rules.
- **Debug Tools**: ASCII tree visualization, Tournament system, SPSA Tuner.
- **Performance Optimized**: Aggressive compiler optimizations (`-O3`, `-march=native`, `-flto`, `-fopenmp`).

## Build

```bash
make          # Build main game (dama)
make tests    # Build and run unit tests
make fast     # Build fast 1v1 benchmark
perch
make tournament # Build and run full tournament
make clean    # Clean build artifacts
````

## Roadmap & Enhancements (ToDo)

Structured according to *Browne et al. (2012) MCTS Survey*.

### 1\. Tree Policy Enhancements

#### Bandit Based

- [x] **UCB1-Tuned**: Implement variance-based upper confidence bound to handle nodes with high score variance.
- [ ] **UCB2**: A variant of UCB1 that is asymptotically efficient for K-armed bandits (uses recursive runs).
- [ ] **PUCT (Predictor + UCT)**: Incorporates prior probabilities (e.g., from heuristics or neural networks) into the UCB formula to guide exploration.

#### Selection

- [x] **Zobrist Hashing**: Implement incremental hashing to identify unique board states .
- [x] **Transposition Table (DAG)**: Map identical states reached via different paths .
- [x] **Progressive Bias**: Domain-specific heuristic knowledge added to UCB formula . (Implemented, currently disabled).
- [ ] **First Play Urgency (FPU)**: Assign fixed high value to unvisited nodes to encourage early exploitation .
- [ ] **Decisive Moves**: Immediately play moves that lead to a win or prevent an immediate loss .
- [ ] **Opening Books**: Use expert opening databases or MCTS-generated books to guide early play .
- [ ] **Search Seeding**: Initialize node statistics with heuristic knowledge instead of zero .
- [ ] **History Heuristic**: Use historical success of moves (e.g., in other branches) to bias selection .
- [ ] **Progressive History**: Combine Progressive Bias with History Heuristic for dynamic bias .

#### AMAF (All Moves As First)

- [ ] **RAVE (Rapid Action Value Estimation)**: *[Low Priority]* Share statistics between tree nodes assuming that a move\'s value is independent of when it is played.

#### Game Theoretic

- [x] **MCTS-Solver**: Propagate "Proven Win" (+∞) and "Proven Loss" (-∞) values up the tree.
- [ ] **Monte Carlo Proof-Number Search (MC-PNS)**: Use MC simulations to guide Proof-Number Search for faster endgame solving .
- [ ] **Score Bounded MCTS**: Maintain optimistic/pessimistic bounds for nodes to handle multiple outcomes (Win/Loss/Draw) more efficiently.

#### Move Pruning (Domain Knowledge)

- [ ] **Soft Pruning (Progressive Widening)**: Temporarily prune moves based on heuristics, revisiting them as visit counts increase.
- [ ] **Hard Pruning (Absolute/Relative)**: Permanently remove moves that are statistically inferior (Absolute) or bounded by an upper confidence limit (Relative).
- [ ] **Pruning with Domain Knowledge**: Use game-specific knowledge (e.g., formations, territory) to prune weak moves.

### 2\. Other Enhancements

#### Simulation

- [x] **Rule-Based Simulation Policy**: Use domain knowledge (captures, promotion) to bias simulations. (Tested "Smart Rollout", currently Random is default).
- [ ] **Learning a Simulation Policy (MAST/PAST)**: Move-Average Sampling Technique (MAST) keeps global statistics for moves to bias future simulations.
- [ ] **Last Good Reply (LGR)**: Store successful replies to moves and reuse them in subsequent simulations to punish mistakes.
- [ ] **Evaluation Function**: Use a static evaluation function to avoid obvious bad moves or cut off simulations early.
- [ ] **Patterns**: Detect specific board patterns (e.g., traps, formations) to guide the simulation.

#### Backpropagation

- [x] **Transposition Table Updates**: Share statistics between different nodes corresponding to the same state (handling transpositions). Strategies like UCT1/2/3 can be explored.
- [ ] **Decaying Reward**: Multiply reward by a constant $\gamma < 1$ between nodes to weight early wins more heavily than later wins.
- [ ] **Score Bonus**: Backpropagate values in intervals $[0, \phi]$ for loss and $[\phi, 1]$ for win to distinguish between strong and weak wins/losses.
- [ ] **Weighting Simulation Results**: Weight simulations based on duration or timing (e.g., later/shorter simulations are often more accurate).

#### Parallelisation

The independent nature of each simulation in MCTS makes it a good target for parallelisation.

- [ ] **Leaf Parallelisation**: Execute multiple simultaneous simulations from a leaf node to collect better initial statistics.
- [ ] **Root Parallelisation**: Build multiple MCTS trees simultaneously (Root-Parallel) and merge results (e.g., Majority Voting or Sum of Visits).
- [ ] **Tree Parallelisation**: All threads work on the same shared tree using locks (global or local mutexes) or "Virtual Loss" to handle concurrency.

## Usage

### Play a game (Human vs AI)

```bash
./bin/dama
```

### Run a Tournament

Runs a round-robin tournament between different MCTS configurations.

```bash
./bin/tournament
```

## Project Structure

```text
├── src/
│   ├── game.c/h       # Core game logic, bitboards, move lookups
│   ├── mcts.c/h       # MCTS core, Arena, TT, Solver
│   ├── debug.c/h      # Centralized debug utilities and printing
│   └── params.h       # Hyperparameters, Weights, Time Control
├── tools/
│   ├── tuner.c           # SPSA parameter tuning
│   ├── tournament.c      # Round-robin tournament system
│   └── fast_tournament.c # Quick 1v1 Benchmark (GM vs Vanilla)
├── test/
│   └── test_game.c    # Unit tests for rule compliance
├── main.c             # Main CLI game loop
└── Makefile
```

## License

MIT
