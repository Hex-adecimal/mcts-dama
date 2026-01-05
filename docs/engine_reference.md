# Engine Module Reference

This document provides a comprehensive overview of the `engine` module in MCTS Dama. It covers the core architecture, key functions, implementation choices, and performance benchmarks.

## Architecture Overview

The engine module is designed with **Separation of Concerns** and **Performance** as primary goals. It is strictly separated into the following components:

| Component | Files | Description |
|-----------|-------|-------------|
| **Game State** | `game.h`, `game.c` | Defines `GameState` struct, piece representation (Bitboards), and core state transitions (`apply_move`). |
| **Move Generation** | `movegen.h`, `movegen.c` | Generates legal moves using precomputed lookup tables and bitwise operations. Handles Italian checkers rules (mandatory capture priority). |
| **Zobrist Hashing** | `zobrist.h`, `zobrist.c` | Handles state hashing for transposition tables. Uses a custom 64-bit Xorshift RNG for determinism. |
| **View/IO** | `game_view.h`, `game_view.c` | Handles all printing and debug visualization. Decouples core logic from `<stdio.h>`. |

## Key Functions & Complexity

| Function | File | Time Complexity | Description |
|----------|------|-----------------|-------------|
| `init_game` | `game.c` | **O(1)** | Initializes the board to the starting position and computes the initial hash. |
| `apply_move` | `game.c` | **O(1)** | Updates the board state for a given move. Uses incremental Zobrist updates (XOR) for speed. |
| `movegen_generate` | `movegen.c` | **O(B)** * | Generates all legal moves. *Complexity is O(B) where B is the number of pieces (popcount), effectively constant time due to board size limits. |
| `zobrist_compute_hash` | `zobrist.c` | **O(B)** | Computes the full hash from scratch. Used only for initialization/verification. O(B) where B is the number of pieces. |
| `zobrist_init` | `zobrist.c` | **O(1)** | Initializes the random keys. Must be called once at startup. |

## Implementation Choices

### 1. Bitboards

We use **Bitboards** (64-bit integers) to represent the board.

- **Mapping**: We map the 32 playable dark squares to specific bits in a full 64-bit integer (using a sparse mapping where light squares are unused).
- **Why**: This allows using consistent shift offsets (e.g., +7, +9 for NE/NW) for all squares without checking for wrap-around on every individual file, and leverages SIMD-like parallel bitwise operations.
- **Benefit**: Checking for empty squares (`~occupied & target`), valid moves, or captures is performed branchlessly for sets of pieces.

### 2. Lookup Tables

`movegen.c` relies heavily on precomputed static arrays (`PAWN_MOVE_TARGETS`, `LADY_MOBILITY`, etc.).

- **Why**: Avoids repeating expensive calculations (like boundary checks or direction offsets) during search.
- **Benefit**: Move generation is reduced to array lookups and bitwise masking.

### 3. Zobrist Hashing

We use **Zobrist Hashing** for state identification.

- **Why**: To support Transposition Tables (TT) and fast state repetition detection.
- **Choice**: Custom 64-bit Xorshift RNG ensures determinism across platforms (critical for distributed training) compared to `rand()`.
- **Optimization**: We use incremental updates in `apply_move` (XORing only changed pieces) rather than recomputing the full hash.
- **Replacement Policy**: The Transposition Table (in `mcts.h`) uses an **Always Replace** policy with a thread-safe global lock. It stores pointers to MCTS nodes (Graph-based MCTS) rather than simple scores, allowing efficient re-use of search sub-trees (DAG).

### 4. Italian Rules Priority

The move generator implements the strict Italian Checkers capture priority:

1. Maximize number of captures (K).
2. If K is equal, maximize value of captured pieces (Lady > Pawn).
3. If still equal, prioritize moving with a Lady.
4. If still equal, prioritize capturing a Lady first.

**Implementation Logic:**
The move generator uses a **Generate & Filter** approach:

1. **Separation**: Functions differentiate between Simple moves and Captures. If any capture is available, simple move generation is skipped entirely (First-level pruning).
2. **Expansion**: `movegen_generate_captures` uses recursion to find *all* valid capture chains (including multi-jumps).
3. **Filtering**: The `filter_moves` function post-processes the list, calculating a priority score for each move based on the rules above, and pruning sub-optimal moves. While not fully branchless, this ensures strict adherence to the complex Italian rules.

## Benchmarks

Benchmarks were run on Apple M2 (ARM64).

| Operation | Ops/Sec | Avg Time (μs) | Notes |
|-----------|---------|---------------|-------|
| `movegen` (Initial) | ~9.8M | 0.10 μs | Full move generation from start position. |
| `movegen` (Midgame) | ~10.0M | 0.10 μs | Move generation in complex midgame positions. |
| `apply_move` | ~50.4M | 0.02 μs | Applying a move and updating hash. |
| `MCTS` (Vanilla) | ~110,000 NPS | - | Nodes Per Second (Pure CPU Heuristic). |
| `MCTS` (CNN) | ~4,000 NPS* | - | *Est. Single-threaded with NN inference overhead. |

## Integration with AI (MCTS/CNN)

The engine bridges the gap between raw bitboards and the Neural Network via the **Tensor Encoder** (`cnn_encode.c`):

### 1. Canonical State Representation

The board is always transformed to the **Current Player's perspective**:

- If Black to move, the board is mirrored vertically.
- Channels are invariant: [0] My Pawns, [1] My Ladies, [2] Opponent Pawns, [3] Opponent Ladies.
- This allows the CNN to learn a single strategy (e.g., "White's perspective") applicable to both sides.

### 2. Thread Safety

- **GameState**: Passed by copy/pointer on stack. Thread-safe for parallel searches if not shared.
- **Transposition Table**: Protected by fine-grained or global mutexes (current: Global Lock for simplicity).
- **MCTS Tree**: Uses Compare-and-Swap (atomics) for access stats, and locks for node expansion.

### 3. Verification (Perft)

Correctness of move generation and Italian rules is verified via a comprehensive **Unit Test Suite** (`tests/unit/test_engine.c`) rather than a simple perft counter, ensuring that specific edge cases (e.g., Lady capture priority) are handled correctly.

***

*Generated automatically by Antigravity Agent.*
