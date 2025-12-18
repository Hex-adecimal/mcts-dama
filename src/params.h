#ifndef PARAMS_H
#define PARAMS_H

// =============================================================================
// UCB1 & SCORING
// =============================================================================

// =============================================================================
// UCB1 & SCORING
// =============================================================================

// =============================================================================
// UCB1 & SCORING
// =============================================================================

#define UCB1_C              1.30  // Tuned - Base is sqrt(2)
#define WIN_SCORE           1.0
#define DRAW_SCORE          0.25
#define LOSS_SCORE          0.0

// First Play Urgency (FPU): Value assigned to unvisited nodes.
// If too high (1e9), forces full width exploration (breadth-first locally).
#define FPU_VALUE           100.0

// Decaying Reward: Discount factor for rewards based on simulation depth.
// 0.999 means reward decays by .1% per move in rollout. Encourages fast wins.
#define DEFAULT_DECAY_FACTOR 0.999

// =============================================================================
// MCTS DEFAULTS
// =============================================================================

#define DEFAULT_ROLLOUT_EPSILON  0.2
#define ROLLOUT_EPSILON_SMART    0.1    
#define ROLLOUT_EPSILON_RANDOM   1.0    

#define EXPANSION_THRESHOLD      0
#define DEFAULT_USE_LOOKAHEAD    1
#define DEFAULT_TREE_REUSE       1
#define DEFAULT_BIAS_CONSTANT    1.2  // Tuned (Run 2). Note: Progressive Bias disabled in GM by default.

// =============================================================================
// LIMITS
// =============================================================================

#define MAX_ROLLOUT_DEPTH        200    // Max simulation depth
#define MAX_GAME_TURNS           150    // Max turns before forced draw
#define MAX_GAME_TURNS_TUNER     400    // Safety limit for tuner

// Tournament settings
// Sample size for ±5% precision @ 95% CI: n = (1.96² × 0.5 × 0.5) / 0.05² ≈ 385
// Sample size for ±10% precision @ 90% CI: n = (1.645² × 0.5 × 0.5) / 0.05² ≈ 100
#define GAMES_PER_PAIRING        100    // Full tournament: games per pairing
#define GAMES_FAST               100    // Fast 1v1 tournament vs Vanilla

// =============================================================================
// TIME & MEMORY
// =============================================================================

// Main game (Human vs AI) - Time presets
#define TIME_LOW            0.2
#define TIME_MID            1.0
#define TIME_HIGH           3.0

#define ARENA_SIZE          ((size_t)8 * 1024 * 1024 * 1024)  // 8GB
#define TIME_WHITE          TIME_HIGH
#define TIME_BLACK          TIME_HIGH

// Tournament (faster games, large memory)
#define ARENA_SIZE_TOURNAMENT  ((size_t)4 * 1024 * 1024 * 1024)   // 4GB per player
#define TIME_TOURNAMENT        TIME_LOW

// Tuner (fast games, minimal memory for speed)
#define ARENA_SIZE_TUNER       ((size_t)256 * 1024 * 1024)    // 256MB per player
#define TIME_TUNER             TIME_LOW

// =============================================================================
// HEURISTIC WEIGHTS (SPSA-tuned defaults)
// =============================================================================
// These are the optimized weights from SPSA tuning with corrected MCTS.
// Used by main.c and can be used as starting point for tuner.c.

#define W_CAPTURE           9.85
#define W_PROMOTION         4.89
#define W_ADVANCE           1.36  // Significant increase (was 0.43)
#define W_CENTER            2.31
#define W_EDGE              0.90  // Significant decrease (was 1.59)
#define W_BASE              1.53  // Increased
#define W_THREAT            9.74
#define W_LADY_ACTIVITY     5.12

// Legacy weight (used in rollout danger check)
#define WEIGHT_DANGER       200

#endif // PARAMS_H
