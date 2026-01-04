#ifndef PARAMS_H
#define PARAMS_H

// =============================================================================
// NEURAL NETWORK ARCHITECTURE
// =============================================================================

#define CNN_BOARD_SIZE      8
#define CNN_HISTORY_T       3       // Number of timesteps (current + 2 previous)
#define CNN_PIECE_CHANNELS  4       // white_pawns, white_ladies, black_pawns, black_ladies
#define CNN_POLICY_SIZE     512     // 64 squares × 8 channels (4 moves + 4 captures)
#define CNN_VALUE_HIDDEN    256     // Value head hidden layer size

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

// Fast Rollout Material Advantage Thresholds
#define FAST_ROLLOUT_MATERIAL_THRESHOLD  3      // Piece diff for early termination
#define FAST_ROLLOUT_WIN_SCORE           0.85   // Score when ahead by threshold
#define FAST_ROLLOUT_LOSS_SCORE          0.15   // Score when behind by threshold
#define FAST_ROLLOUT_MATERIAL_WEIGHT     0.05   // Weight for material-based eval

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
#define W_LADY_ACTIVITY         12.0
#define WEIGHT_DANGER           200         // Rollout danger penalty

// =============================================================================
// MCTS - THREADING
// =============================================================================

// IMPORTANT: When using parallel self-play (--threads 10), each game spawns
// NUM_MCTS_THREADS worker threads. With 10 games × 4 threads = 40 threads on
// 10 cores, causing massive context switching overhead (~50% performance loss).
// 
// RECOMMENDATION: 
//   - Use 1 thread for parallel self-play/tournament (optimal: 10 games × 1 = 10 threads)
//   - Use 4 threads for single-game analysis (maximum search speed)
//
#define NUM_MCTS_THREADS        0           // Set to 0 for maximal throughput in parallel self-play (sequential search)
#define MCTS_BATCH_SIZE         64          // Max batch size for Async MCTS inference

// =============================================================================
// TRAINING & LOGGING
// =============================================================================

#define LOG_RETENTION_COUNT     10          // Keep last N log files

// Mixed Opponent Training
#define MIX_OPPONENT_PROB       0.25        // 25% of games vs Grandmaster Heuristics

// Memory Configurations
#define ARENA_SIZE_SELFPLAY         ((size_t)512 * 1024 * 1024) // 512 MB for persistence
#define ARENA_SIZE_TOURNAMENT       ((size_t)512 * 1024 * 1024)  // 512 MB (reset per move)
#define ARENA_SIZE_BENCHMARK        ((size_t)64 * 1024 * 1024)

// =============================================================================
// SELFPLAY CONFIGURATION
// =============================================================================

#define SELFPLAY_MAX_MOVES          200         // Max moves per self-play game
#define RESIGN_THRESHOLD            -0.90f      // Neural network value below which to resign
#define RESIGN_CHECK_THRESHOLD      40          // Moves before resignation checks begin
#define EARLY_EXIT_CHECK_INTERVAL   10          // Check early exit every N nodes
#define EARLY_EXIT_MIN_VISITS       40          // Minimum visits before early exit checks
// =============================================================================
// TRAINING - NEURAL NETWORK
// =============================================================================

#define CNN_POLICY_LR           0.5f        // Policy head LR (high: 512-class softmax needs strong gradients)
#define CNN_VALUE_LR            0.01f       // Value head LR (low: already learns well)
#define CNN_DEFAULT_BATCH_SIZE  64          // Batch size
#define CNN_DEFAULT_L2_DECAY    1e-4f       // L2 regularization
#define CNN_DEFAULT_MOMENTUM    0.9f        // SGD momentum
#define CNN_DEFAULT_EPOCHS      10          // Epochs per training run
#define CNN_PATIENCE            5           // Early stopping patience

// LR Schedule: Linear warmup for epoch 1, then decay on plateau
#define CNN_LR_WARMUP_EPOCHS    1           // Linear warmup for first N epochs
#define CNN_LR_DECAY_FACTOR     0.5f        // Multiply LR by this on plateau
#define CNN_LR_DECAY_PATIENCE   2           // Epochs without improvement before decay

#define TT_SIZE_DEFAULT             (1024 * 1024)

#endif // PARAMS_H
