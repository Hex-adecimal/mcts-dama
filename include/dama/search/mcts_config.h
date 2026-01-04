/**
 * mcts_config.h - MCTS Configuration, Presets, and Statistics
 * 
 * Extracted from mcts_types.h for better modularity.
 * Contains: MCTSConfig, MCTSStats, MCTSPreset, mcts_get_preset()
 */

#ifndef MCTS_CONFIG_H
#define MCTS_CONFIG_H

#include "dama/common/params.h"
#include <string.h>

// =============================================================================
// MCTS CONFIGURATION
// =============================================================================

typedef struct MCTSConfig {
    double ucb1_c;
    double rollout_epsilon;
    double draw_score;
    int expansion_threshold;
    
    int verbose;
    int use_lookahead;
    int use_tree_reuse;
    int use_ucb1_tuned;
    int use_tt;
    int use_solver;
    int use_progressive_bias;
    double bias_constant;
    int use_fpu;
    double fpu_value;
    int use_decaying_reward;
    double decay_factor;
    
    // Fast rollout: early termination on material advantage, shorter depth
    int use_fast_rollout;
    int fast_rollout_depth;  // Max depth when fast rollout enabled (default: 50)

    struct {
        double w_capture;
        double w_promotion;
        double w_advance;
        double w_center;
        double w_edge;
        double w_base;
        double w_threat;
        double w_lady_activity;
    } weights;

    int use_puct;
    double puct_c;
    void *nn_weights;
    void *cnn_weights;
    int max_nodes;
} MCTSConfig;

// =============================================================================
// MCTS STATISTICS
// =============================================================================

typedef struct {
    int total_moves;
    long total_iterations;
    long total_nodes;
    long current_move_iterations;
    long total_depth;
    double total_time;
    size_t total_memory;
    // Debug stats for tree analysis
    long total_expansions;      // How many nodes were expanded
    long total_policy_cached;   // How many times CNN policy was computed
} MCTSStats;

// =============================================================================
// PRESETS
// =============================================================================

typedef enum {
    MCTS_PRESET_PURE_VANILLA,
    MCTS_PRESET_VANILLA,
    MCTS_PRESET_GRANDMASTER,
    MCTS_PRESET_ALPHA_ZERO,
    MCTS_PRESET_TT_ONLY,
    MCTS_PRESET_SOLVER_ONLY,
    MCTS_PRESET_TUNED_ONLY,
    MCTS_PRESET_FPU_ONLY,
    MCTS_PRESET_DECAY_ONLY,
    MCTS_PRESET_LOOKAHEAD_ONLY,
    MCTS_PRESET_TREE_REUSE_ONLY,
    MCTS_PRESET_WEIGHTS_ONLY,
    MCTS_PRESET_SMART_ROLLOUTS,
    MCTS_PRESET_PROG_BIAS_ONLY
} MCTSPreset;

static inline void apply_weights(MCTSConfig *c) {
    c->weights.w_capture = W_CAPTURE;
    c->weights.w_promotion = W_PROMOTION;
    c->weights.w_advance = W_ADVANCE;
    c->weights.w_center = W_CENTER;
    c->weights.w_edge = W_EDGE;
    c->weights.w_base = W_BASE;
    c->weights.w_threat = W_THREAT;
    c->weights.w_lady_activity = W_LADY_ACTIVITY;
}

static inline MCTSConfig mcts_get_preset(MCTSPreset preset) {
    MCTSConfig cfg;
    memset(&cfg, 0, sizeof(MCTSConfig));

    cfg.ucb1_c = UCB1_C;
    cfg.draw_score = DRAW_SCORE;
    cfg.expansion_threshold = EXPANSION_THRESHOLD;
    cfg.rollout_epsilon = ROLLOUT_EPSILON_RANDOM;
    cfg.use_lookahead = 0;
    cfg.use_tree_reuse = 0;

    switch (preset) {
        case MCTS_PRESET_PURE_VANILLA:
            break;
        case MCTS_PRESET_VANILLA:
            cfg.use_lookahead = DEFAULT_USE_LOOKAHEAD;
            cfg.use_tree_reuse = DEFAULT_TREE_REUSE;
            break;
        case MCTS_PRESET_GRANDMASTER:
            cfg.use_puct = 1;
            cfg.puct_c = PUCT_C;
            cfg.rollout_epsilon = ROLLOUT_EPSILON_NN;
            cfg.use_solver = DEFAULT_USE_SOLVER;
            cfg.use_progressive_bias = 1;
            cfg.bias_constant = DEFAULT_BIAS_CONSTANT;
            apply_weights(&cfg);
            break;
        case MCTS_PRESET_ALPHA_ZERO:
            cfg.use_puct = 1;
            cfg.puct_c = PUCT_C;
            cfg.rollout_epsilon = ROLLOUT_EPSILON_NN;
            cfg.use_solver = DEFAULT_USE_SOLVER;
            break;
        case MCTS_PRESET_TT_ONLY:
            cfg.use_tt = DEFAULT_USE_TT;
            break;
        case MCTS_PRESET_SOLVER_ONLY:
            cfg.use_solver = DEFAULT_USE_SOLVER;
            break;
        case MCTS_PRESET_TUNED_ONLY:
            cfg.use_ucb1_tuned = DEFAULT_USE_UCB1_TUNED;
            break;
        case MCTS_PRESET_FPU_ONLY:
            cfg.use_fpu = DEFAULT_USE_FPU;
            cfg.fpu_value = FPU_VALUE;
            break;
        case MCTS_PRESET_DECAY_ONLY:
            cfg.use_decaying_reward = DEFAULT_USE_DECAY;
            cfg.decay_factor = DEFAULT_DECAY_FACTOR;
            break;
        case MCTS_PRESET_LOOKAHEAD_ONLY:
            cfg.use_lookahead = 1;
            break;
        case MCTS_PRESET_TREE_REUSE_ONLY:
            cfg.use_tree_reuse = 1;
            break;
        case MCTS_PRESET_WEIGHTS_ONLY:
            apply_weights(&cfg);
            cfg.rollout_epsilon = DEFAULT_ROLLOUT_EPSILON;
            break;
        case MCTS_PRESET_SMART_ROLLOUTS:
            apply_weights(&cfg);
            cfg.rollout_epsilon = ROLLOUT_EPSILON_HEURISTIC;
            break;
        case MCTS_PRESET_PROG_BIAS_ONLY:
            apply_weights(&cfg);
            cfg.use_progressive_bias = 1;
            cfg.bias_constant = DEFAULT_BIAS_CONSTANT;
            break;
    }

    return cfg;
}

#endif // MCTS_CONFIG_H
