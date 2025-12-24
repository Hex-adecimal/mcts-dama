#ifndef MCTS_PRESETS_H
#define MCTS_PRESETS_H

#include "mcts_types.h"

/**
 * MCTS Preset Types
 */
typedef enum {
    MCTS_PRESET_PURE_VANILLA,   // Truly Pure MCTS: UCB1 + Random Rollouts (No Lookahead/Reuse)
    MCTS_PRESET_VANILLA,        // Standard Baseline: UCB1 + Random Rollouts + Lookahead
    MCTS_PRESET_GRANDMASTER,    // Best Optimized: UCB1-Tuned + TT + Solver + FPU + Decay
    MCTS_PRESET_ALPHA_ZERO,     // Neural Net guided: PUCT + CNN
    
    // Single Enhancement Ablations
    MCTS_PRESET_TT_ONLY,        // Vanilla + Transposition Table
    MCTS_PRESET_SOLVER_ONLY,    // Vanilla + Endgame Solver
    MCTS_PRESET_TUNED_ONLY,     // Vanilla + UCB1-Tuned
    MCTS_PRESET_FPU_ONLY,       // Vanilla + First Play Urgency (FPU)
    MCTS_PRESET_DECAY_ONLY,     // Vanilla + Decaying Reward
    MCTS_PRESET_LOOKAHEAD_ONLY, // Pure Vanilla + 1-ply Lookahead
    MCTS_PRESET_TREE_REUSE_ONLY,// Pure Vanilla + Tree Reuse
    
    // Feature Stress Tests (Broken/Weak Features)
    MCTS_PRESET_WEIGHTS_ONLY,   // Vanilla + Heuristic Weights (Weak)
    MCTS_PRESET_SMART_ROLLOUTS, // Vanilla + 100% Heuristic Rollouts (Slow/Weak)
    MCTS_PRESET_PROG_BIAS_ONLY  // Vanilla + Progressive Bias (Unstable)
} MCTSPreset;

/**
 * Returns a default configuration for a given preset.
 * @param preset The type of agent profile to retrieve.
 * @return MCTSConfig populated with standard values from params.h.
 */
MCTSConfig mcts_get_preset(MCTSPreset preset);

#endif // MCTS_PRESETS_H
