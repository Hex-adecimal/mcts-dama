/**
 * clop.h - CLOP (Confident Local Optimization for Noisy Black-Box Parameter Tuning)
 * 
 * Implements the CLOP algorithm by Rémi Coulom for optimizing MCTS hyperparameters.
 * Uses local quadratic regression with confident sample filtering.
 * 
 * Reference: https://www.remi-coulom.fr/CLOP/
 */

#ifndef CLOP_H
#define CLOP_H

#include <stddef.h>

// =============================================================================
// CLOP PARAMETER DEFINITION
// =============================================================================

/**
 * Definition of a single tunable parameter.
 */
typedef struct {
    const char *name;
    double lower;
    double upper;
    double initial;
} CLOPParamDef;

// =============================================================================
// CLOP STATE
// =============================================================================

/**
 * CLOP optimizer state.
 * Maintains samples, outcomes, and the fitted quadratic model.
 */
typedef struct {
    int num_params;
    const CLOPParamDef *param_defs;
    
    // Current and best parameters
    double *current;
    double *best_params;
    double best_outcome;
    
    // Sample storage
    double *samples;       // [max_samples * num_params] - flattened
    double *outcomes;      // [max_samples]
    int num_samples;
    int max_samples;
    
    // Quadratic model: f(x) = c + Σ l_i*x_i + Σ q_ij*x_i*x_j
    double constant;
    double *linear;        // [num_params]
    double *quadratic;     // [num_params * num_params] upper triangular
    
    // Configuration
    double confidence_threshold;  // For discarding inferior samples
    int min_samples_for_model;    // Minimum samples before fitting model
    unsigned int rng_state;       // RNG for exploration
} CLOPState;

// =============================================================================
// API
// =============================================================================

/**
 * Initialize CLOP optimizer.
 * 
 * @param state       Pointer to state to initialize
 * @param param_defs  Array of parameter definitions
 * @param num_params  Number of parameters
 * @param max_samples Maximum samples to store
 * @return 0 on success, -1 on error
 */
int clop_init(CLOPState *state, const CLOPParamDef *param_defs, 
              int num_params, int max_samples);

/**
 * Suggest next parameter values to evaluate.
 * Uses the quadratic model to find promising regions.
 * 
 * @param state      CLOP state
 * @param params_out Output array for suggested parameters [num_params]
 * @return 0 on success, -1 on error
 */
int clop_suggest(CLOPState *state, double *params_out);

/**
 * Update CLOP with a new sample.
 * 
 * @param state   CLOP state
 * @param params  Parameters that were evaluated [num_params]
 * @param outcome Outcome value (higher is better, e.g., win rate 0.0-1.0)
 * @return 0 on success, -1 on error
 */
int clop_update(CLOPState *state, const double *params, double outcome);

/**
 * Get the best parameters found so far.
 * 
 * @param state      CLOP state
 * @param params_out Output array for best parameters [num_params]
 */
void clop_get_best(const CLOPState *state, double *params_out);

/**
 * Get current number of samples.
 */
int clop_get_num_samples(const CLOPState *state);

/**
 * Free CLOP resources.
 */
void clop_free(CLOPState *state);

#endif // CLOP_H
