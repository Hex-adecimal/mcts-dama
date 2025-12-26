#include "mcts_presets.h"
#include "../params.h"
#include <string.h>

static void apply_weights(MCTSConfig *c) {
    c->weights.w_capture = W_CAPTURE;
    c->weights.w_promotion = W_PROMOTION;
    c->weights.w_advance = W_ADVANCE;
    c->weights.w_center = W_CENTER;
    c->weights.w_edge = W_EDGE;
    c->weights.w_base = W_BASE;
    c->weights.w_threat = W_THREAT;
    c->weights.w_lady_activity = W_LADY_ACTIVITY;
}

MCTSConfig mcts_get_preset(MCTSPreset preset) {
    MCTSConfig cfg;
    memset(&cfg, 0, sizeof(MCTSConfig));

    // --- GLOBAL BASELINE (Truly Pure MCTS) ---
    cfg.ucb1_c = UCB1_C;
    cfg.draw_score = DRAW_SCORE;
    cfg.expansion_threshold = EXPANSION_THRESHOLD;
    cfg.rollout_epsilon = ROLLOUT_EPSILON_RANDOM;
    cfg.use_lookahead = 0;
    cfg.use_tree_reuse = 0;

    switch (preset) {
        case MCTS_PRESET_PURE_VANILLA:
            break; // Truly pure

        case MCTS_PRESET_VANILLA:
            cfg.use_lookahead = DEFAULT_USE_LOOKAHEAD;
            cfg.use_tree_reuse = DEFAULT_TREE_REUSE;
            break;

        case MCTS_PRESET_GRANDMASTER:
            // Minimal but powerful: PUCT + Progressive Bias + Solver
            cfg.use_puct = 1;
            cfg.puct_c = PUCT_C;
            cfg.rollout_epsilon = ROLLOUT_EPSILON_NN; // No rollouts when CNN available
            cfg.use_solver = DEFAULT_USE_SOLVER;
            cfg.use_progressive_bias = 1;
            cfg.bias_constant = DEFAULT_BIAS_CONSTANT;
            apply_weights(&cfg);
            break;

        case MCTS_PRESET_ALPHA_ZERO:
            cfg.use_puct = 1;
            cfg.puct_c = PUCT_C;
            cfg.rollout_epsilon = ROLLOUT_EPSILON_NN;
            cfg.use_lookahead = DEFAULT_USE_LOOKAHEAD;
            cfg.use_tree_reuse = DEFAULT_TREE_REUSE;
            cfg.use_tt = DEFAULT_USE_TT;
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
