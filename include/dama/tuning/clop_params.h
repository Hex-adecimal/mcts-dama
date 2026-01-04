/**
 * clop_params.h - Default CLOP Parameter Definitions
 * 
 * Defines which MCTS parameters to tune and their bounds.
 */

#ifndef CLOP_PARAMS_H
#define CLOP_PARAMS_H

#include "dama/tuning/clop.h"

// =============================================================================
// TUNABLE MCTS PARAMETERS
// =============================================================================

/**
 * Default parameter set for CLOP tuning.
 * Format: {name, lower_bound, upper_bound, initial_value}
 */
static const CLOPParamDef CLOP_DEFAULT_PARAMS[] = {
    // Exploration constants
    {"ucb1_c",      0.5,  3.0,   1.414},
    {"puct_c",      0.5,  4.0,   2.0},
    
    // Progressive bias
    {"bias_const",  0.0,  2.0,   0.5},
    
    // First Play Urgency
    {"fpu_value",  -1.0,  1.0,   0.0},
    
    // Decay
    {"decay_factor", 0.90, 1.0,  0.99},
    
    // Heuristic weights for rollouts
    {"w_capture",   0.0, 10.0,   5.0},
    {"w_promotion", 0.0, 10.0,   4.0},
    {"w_advance",   0.0,  5.0,   1.0},
    {"w_center",    0.0,  5.0,   0.5},
    {"w_edge",     -2.0,  2.0,  -0.3},
    {"w_threat",    0.0,  5.0,   2.0},
    {"w_lady",      0.0,  5.0,   1.5},
};

#define CLOP_DEFAULT_NUM_PARAMS \
    (sizeof(CLOP_DEFAULT_PARAMS) / sizeof(CLOP_DEFAULT_PARAMS[0]))

#endif // CLOP_PARAMS_H
