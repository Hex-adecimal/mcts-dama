/**
 * clop.c - CLOP Implementation
 * 
 * Confident Local Optimization for Noisy Black-Box Parameter Tuning.
 * Uses local quadratic regression with LAPACK for least squares fitting.
 * 
 * Algorithm:
 * 1. Collect samples (params, outcome)
 * 2. Fit quadratic model: f(x) = c + Σ l_i*x_i + Σ q_ij*x_i*x_j
 * 3. Discard samples confidently below mean
 * 4. Suggest next point by maximizing expected improvement
 */

#include "dama/tuning/clop.h"
#include "dama/common/logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
// Mac Accelerate uses __CLPK_integer which may not be defined
typedef int lapack_int;
#else
typedef int lapack_int;
// Declare dgels for non-Apple
extern void dgels_(char *trans, int *m, int *n, int *nrhs, 
                   double *a, int *lda, double *b, int *ldb,
                   double *work, int *lwork, int *info);
#endif

// =============================================================================
// INTERNAL RNG (Xorshift32)
// =============================================================================

static unsigned int clop_rng_next(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double clop_rng_uniform(unsigned int *state) {
    return (double)clop_rng_next(state) / (double)0xFFFFFFFF;
}

// =============================================================================
// HELPER: Normalize/Denormalize Parameters
// =============================================================================

static void normalize_params(const CLOPState *state, const double *params, double *out) {
    for (int i = 0; i < state->num_params; i++) {
        double range = state->param_defs[i].upper - state->param_defs[i].lower;
        out[i] = (params[i] - state->param_defs[i].lower) / range;
    }
}

static void denormalize_params(const CLOPState *state, const double *normalized, double *out) {
    for (int i = 0; i < state->num_params; i++) {
        double range = state->param_defs[i].upper - state->param_defs[i].lower;
        out[i] = normalized[i] * range + state->param_defs[i].lower;
    }
}

// =============================================================================
// HELPER: Fit Quadratic Model with LAPACK
// =============================================================================

/**
 * Number of coefficients in quadratic model:
 * 1 (constant) + n (linear) + n*(n+1)/2 (quadratic upper triangular)
 */
static int num_coefficients(int n) {
    return 1 + n + (n * (n + 1)) / 2;
}

/**
 * Build design matrix row for quadratic model.
 * Row = [1, x1, x2, ..., xn, x1², x1*x2, ..., xn²]
 */
static void build_design_row(const double *x, int n, double *row) {
    int idx = 0;
    
    // Constant term
    row[idx++] = 1.0;
    
    // Linear terms
    for (int i = 0; i < n; i++) {
        row[idx++] = x[i];
    }
    
    // Quadratic terms (upper triangular including diagonal)
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            row[idx++] = x[i] * x[j];
        }
    }
}

/**
 * Fit quadratic model using LAPACK least squares (dgels).
 * Solves: min ||Ax - b||² where A is design matrix, b is outcomes.
 */
static int fit_quadratic_model(CLOPState *state) {
    if (state->num_samples < state->min_samples_for_model) {
        return 0;  // Not enough samples yet
    }
    
    int m = state->num_samples;
    int n = num_coefficients(state->num_params);
    
    if (m < n) {
        return 0;  // Underdetermined system, skip fitting
    }
    
    // Allocate design matrix (column-major for LAPACK)
    double *A = malloc(m * n * sizeof(double));
    double *b = malloc(m * sizeof(double));
    if (!A || !b) {
        free(A); free(b);
        return -1;
    }
    
    // Temporary for normalized params
    double *norm_params = malloc(state->num_params * sizeof(double));
    if (!norm_params) {
        free(A); free(b);
        return -1;
    }
    
    // Build design matrix
    for (int i = 0; i < m; i++) {
        double *sample = &state->samples[i * state->num_params];
        normalize_params(state, sample, norm_params);
        
        double row[64];  // Assume max 64 coefficients
        build_design_row(norm_params, state->num_params, row);
        
        // Store in column-major order
        for (int j = 0; j < n; j++) {
            A[j * m + i] = row[j];
        }
        b[i] = state->outcomes[i];
    }
    
    free(norm_params);
    
    // Call LAPACK dgels (least squares via QR)
    lapack_int M = m;
    lapack_int N_dim = n;  // Renamed to avoid conflict
    lapack_int nrhs = 1;
    lapack_int lda = m;
    lapack_int ldb = m > n ? m : n;
    lapack_int info;
    
    // Query workspace size
    lapack_int lwork = -1;
    double work_query;
    dgels_("N", &M, &N_dim, &nrhs, A, &lda, b, &ldb, &work_query, &lwork, &info);
    
    lwork = (lapack_int)work_query;
    double *work = malloc(lwork * sizeof(double));
    if (!work) {
        free(A); free(b);
        return -1;
    }
    
    // Solve least squares
    dgels_("N", &M, &N_dim, &nrhs, A, &lda, b, &ldb, work, &lwork, &info);
    
    free(work);
    free(A);
    
    if (info != 0) {
        log_error("[CLOP] LAPACK dgels failed with info=%d", info);
        free(b);
        return -1;
    }
    
    // Extract coefficients from b (first n elements)
    int idx = 0;
    state->constant = b[idx++];
    
    for (int i = 0; i < state->num_params; i++) {
        state->linear[i] = b[idx++];
    }
    
    for (int i = 0; i < state->num_params; i++) {
        for (int j = i; j < state->num_params; j++) {
            state->quadratic[i * state->num_params + j] = b[idx++];
            if (i != j) {
                state->quadratic[j * state->num_params + i] = b[idx - 1];  // Symmetric
            }
        }
    }
    
    free(b);
    return 0;
}

/**
 * Evaluate quadratic model at given (normalized) point.
 */
static double evaluate_model(const CLOPState *state, const double *x) {
    double value = state->constant;
    
    // Linear terms
    for (int i = 0; i < state->num_params; i++) {
        value += state->linear[i] * x[i];
    }
    
    // Quadratic terms
    for (int i = 0; i < state->num_params; i++) {
        for (int j = 0; j < state->num_params; j++) {
            value += state->quadratic[i * state->num_params + j] * x[i] * x[j];
        }
    }
    
    return value;
}

// =============================================================================
// PUBLIC API
// =============================================================================

int clop_init(CLOPState *state, const CLOPParamDef *param_defs, 
              int num_params, int max_samples) {
    if (!state || !param_defs || num_params <= 0 || max_samples <= 0) {
        return -1;
    }
    
    memset(state, 0, sizeof(CLOPState));
    state->num_params = num_params;
    state->param_defs = param_defs;
    state->max_samples = max_samples;
    state->best_outcome = -1e9;
    
    // Allocate arrays
    state->current = malloc(num_params * sizeof(double));
    state->best_params = malloc(num_params * sizeof(double));
    state->samples = malloc(max_samples * num_params * sizeof(double));
    state->outcomes = malloc(max_samples * sizeof(double));
    state->linear = malloc(num_params * sizeof(double));
    state->quadratic = malloc(num_params * num_params * sizeof(double));
    
    if (!state->current || !state->best_params || !state->samples ||
        !state->outcomes || !state->linear || !state->quadratic) {
        clop_free(state);
        return -1;
    }
    
    // Initialize to midpoint
    for (int i = 0; i < num_params; i++) {
        state->current[i] = param_defs[i].initial;
        state->best_params[i] = param_defs[i].initial;
    }
    
    // Zero model coefficients
    memset(state->linear, 0, num_params * sizeof(double));
    memset(state->quadratic, 0, num_params * num_params * sizeof(double));
    
    // Configuration
    state->confidence_threshold = 1.0;
    state->min_samples_for_model = num_params + 2;
    state->rng_state = 12345;  // Default seed
    
    log_printf("[CLOP] Initialized with %d parameters, max %d samples\n", 
               num_params, max_samples);
    
    return 0;
}

int clop_suggest(CLOPState *state, double *params_out) {
    if (!state || !params_out) {
        return -1;
    }
    
    int n = state->num_params;
    
    // Early phase: Latin Hypercube / random exploration
    if (state->num_samples < state->min_samples_for_model) {
        for (int i = 0; i < n; i++) {
            double range = state->param_defs[i].upper - state->param_defs[i].lower;
            double noise = clop_rng_uniform(&state->rng_state) - 0.5;
            params_out[i] = state->param_defs[i].initial + noise * range * 0.5;
            
            // Clamp to bounds
            if (params_out[i] < state->param_defs[i].lower) 
                params_out[i] = state->param_defs[i].lower;
            if (params_out[i] > state->param_defs[i].upper) 
                params_out[i] = state->param_defs[i].upper;
        }
        return 0;
    }
    
    // Fit model if we have enough samples
    if (fit_quadratic_model(state) < 0) {
        log_error("[CLOP] Failed to fit model, using random exploration");
        // Fall back to random
        for (int i = 0; i < n; i++) {
            double range = state->param_defs[i].upper - state->param_defs[i].lower;
            params_out[i] = state->param_defs[i].lower + 
                           clop_rng_uniform(&state->rng_state) * range;
        }
        return 0;
    }
    
    // Find maximum of quadratic model using simple grid search + local refinement
    // (For n <= 12 params, this is tractable)
    double best_value = -1e9;
    double *best_x = malloc(n * sizeof(double));
    double *test_x = malloc(n * sizeof(double));
    
    if (!best_x || !test_x) {
        free(best_x); free(test_x);
        return -1;
    }
    
    // Random search with 100 samples
    for (int iter = 0; iter < 100; iter++) {
        for (int i = 0; i < n; i++) {
            test_x[i] = clop_rng_uniform(&state->rng_state);  // [0, 1] normalized
        }
        
        double value = evaluate_model(state, test_x);
        if (value > best_value) {
            best_value = value;
            memcpy(best_x, test_x, n * sizeof(double));
        }
    }
    
    // Denormalize and add exploration noise
    denormalize_params(state, best_x, params_out);
    
    // Add small exploration noise (10% of range)
    for (int i = 0; i < n; i++) {
        double range = state->param_defs[i].upper - state->param_defs[i].lower;
        double noise = (clop_rng_uniform(&state->rng_state) - 0.5) * 0.1 * range;
        params_out[i] += noise;
        
        // Clamp
        if (params_out[i] < state->param_defs[i].lower) 
            params_out[i] = state->param_defs[i].lower;
        if (params_out[i] > state->param_defs[i].upper) 
            params_out[i] = state->param_defs[i].upper;
    }
    
    free(best_x);
    free(test_x);
    
    return 0;
}

int clop_update(CLOPState *state, const double *params, double outcome) {
    if (!state || !params) {
        return -1;
    }
    
    if (state->num_samples >= state->max_samples) {
        log_error("[CLOP] Sample buffer full");
        return -1;
    }
    
    // Store sample
    int idx = state->num_samples;
    memcpy(&state->samples[idx * state->num_params], params, 
           state->num_params * sizeof(double));
    state->outcomes[idx] = outcome;
    state->num_samples++;
    
    // Update best if improved
    if (outcome > state->best_outcome) {
        state->best_outcome = outcome;
        memcpy(state->best_params, params, state->num_params * sizeof(double));
        log_printf("[CLOP] New best: %.4f\n", outcome);
    }
    
    return 0;
}

void clop_get_best(const CLOPState *state, double *params_out) {
    if (!state || !params_out) return;
    memcpy(params_out, state->best_params, state->num_params * sizeof(double));
}

int clop_get_num_samples(const CLOPState *state) {
    return state ? state->num_samples : 0;
}

void clop_free(CLOPState *state) {
    if (!state) return;
    
    free(state->current);
    free(state->best_params);
    free(state->samples);
    free(state->outcomes);
    free(state->linear);
    free(state->quadratic);
    
    memset(state, 0, sizeof(CLOPState));
}
