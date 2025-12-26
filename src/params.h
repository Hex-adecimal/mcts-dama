#ifndef PARAMS_H
#define PARAMS_H

// =============================================================================
// GAME LIMITS
// =============================================================================

#define MAX_MOVES                   64      // Max legal moves per position
#define MAX_CAPTURES                12      // Max captures in a chain
#define MAX_MOVES_WITHOUT_CAPTURES  40      // Forced draw after this

// =============================================================================
// MCTS - EXPLORATION & SCORING
// =============================================================================

#define UCB1_C                  1.30        // UCB1 exploration (tuned, base: sqrt(2))
#define PUCT_C                  1.5         // PUCT exploration (AlphaZero: ~1.5-2.5)
#define WIN_SCORE               1.0 
#define DRAW_SCORE              0.25
#define LOSS_SCORE              0.0

// First Play Urgency: Value for unvisited nodes
#define FPU_VALUE               100.0

// Decaying Reward: Discount per rollout step (.999 = 0.1% decay)
#define DEFAULT_DECAY_FACTOR    0.999

// =============================================================================
// MCTS - ROLLOUT CONFIGURATION
// =============================================================================

#define DEFAULT_ROLLOUT_EPSILON     0.2     // Epsilon-greedy probability
#define ROLLOUT_EPSILON_SMART       0.1     // Greedy (10% random)
#define ROLLOUT_EPSILON_HEURISTIC   0.0     // 100% Heuristic
#define ROLLOUT_EPSILON_RANDOM      1.0     // Fully random
#define ROLLOUT_EPSILON_NN          0.0     // No rollouts (Neural Net only)

#define EXPANSION_THRESHOLD         0
#define DEFAULT_USE_LOOKAHEAD       1
#define DEFAULT_TREE_REUSE          0
#define DEFAULT_USE_TT              1
#define DEFAULT_USE_SOLVER          1
#define DEFAULT_USE_UCB1_TUNED      1
#define DEFAULT_USE_FPU             1
#define DEFAULT_USE_DECAY           1
#define DEFAULT_BIAS_CONSTANT       1.2     // Progressive bias constant

// =============================================================================
// MCTS - LIMITS
// =============================================================================

#define MAX_ROLLOUT_DEPTH       200         // Max simulation depth
#define MAX_GAME_TURNS          200         // Max turns before forced draw
#define MAX_GAME_TURNS_TUNER    400         // Safety limit for tuner

// =============================================================================
// TIME BUDGETS (seconds)
// =============================================================================

#define TIME_LOWER              0.05        // Very fast (training/tuning)
#define TIME_LOW                0.2         // Fast (tournament)
#define TIME_MID                1.0         // Medium
#define TIME_HIGH               3.0         // Strong play

// Per-player defaults
#define TIME_WHITE              TIME_HIGH
#define TIME_BLACK              TIME_HIGH

// =============================================================================
// MEMORY BUDGETS
// =============================================================================

#define ARENA_SIZE              ((size_t)8 * 1024 * 1024 * 1024)    // 8GB (game)
#define ARENA_SIZE_TOURNAMENT   ((size_t)4 * 1024 * 1024 * 1024)    // 4GB (tournament)
#define ARENA_SIZE_TUNER        ((size_t)256 * 1024 * 1024)         // 256MB (tuner)

// =============================================================================
// HEURISTIC WEIGHTS (SPSA-tuned)
// =============================================================================

#define W_CAPTURE               9.85
#define W_PROMOTION             4.89
#define W_ADVANCE               1.36
#define W_CENTER                2.31
#define W_EDGE                  0.90
#define W_BASE                  1.53
#define W_THREAT                9.74
#define W_LADY_ACTIVITY         5.12
#define WEIGHT_DANGER           200         // Rollout danger penalty

// =============================================================================
// MCTS - THREADING
// =============================================================================

#define NUM_MCTS_THREADS        4           // Number of async MCTS worker threads
#define MCTS_BATCH_SIZE         64          // Max batch size for Async MCTS inference

#endif // PARAMS_H
