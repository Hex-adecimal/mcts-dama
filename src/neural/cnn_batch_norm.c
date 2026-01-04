/**
 * cnn_batch_norm.c - Batch Normalization Operations
 * 
 * Extracted from cnn_core.c and cnn_training.c for better modularity.
 * Contains: batch_norm_forward, batch_norm_forward_relu, batch_norm_backward
 */

#include "dama/neural/cnn.h"
#include <math.h>

// =============================================================================
// BATCH NORMALIZATION FORWARD PASS
// =============================================================================

void batch_norm_forward(
    const float *input,
    const float *gamma, const float *beta,
    float *output,
    float *batch_mean, float *batch_var,
    float *running_mean, float *running_var,
    int C, int H, int W,
    int is_training
) {
    int spatial_size = H * W;
    for (int c = 0; c < C; c++) {
        float mean, var;
        if (is_training) {
            mean = 0;
            for (int i = 0; i < spatial_size; i++) mean += input[c * spatial_size + i];
            mean /= spatial_size;
            
            var = 0;
            for (int i = 0; i < spatial_size; i++) {
                float diff = input[c * spatial_size + i] - mean;
                var += diff * diff;
            }
            var /= spatial_size;
            
            // Save for backward
            if (batch_mean) batch_mean[c] = mean;
            if (batch_var) batch_var[c] = var;
            
            // Update running stats
            running_mean[c] = (1.0f - CNN_BN_MOMENTUM) * running_mean[c] + CNN_BN_MOMENTUM * mean;
            running_var[c] = (1.0f - CNN_BN_MOMENTUM) * running_var[c] + CNN_BN_MOMENTUM * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        
        float inv_std = 1.0f / sqrtf(var + CNN_BN_EPSILON);
        for (int i = 0; i < spatial_size; i++) {
            int idx = c * spatial_size + i;
            output[idx] = gamma[c] * (input[idx] - mean) * inv_std + beta[c];
        }
    }
}

// =============================================================================
// FUSED BATCH NORMALIZATION + RELU
// =============================================================================

// Fused BatchNorm + ReLU (saves one memory pass)
// pre_relu: optional output for pre-ReLU values (needed for correct backward)
void batch_norm_forward_relu(
    const float *input,
    const float *gamma, const float *beta,
    float *output,
    float *pre_relu,  // NEW: saves values before ReLU for backward
    float *batch_mean, float *batch_var,
    float *running_mean, float *running_var,
    int C, int H, int W,
    int is_training
) {
    int spatial_size = H * W;
    #pragma omp parallel for
    for (int c = 0; c < C; c++) {
        float mean, var;
        if (is_training) {
            mean = 0;
            for (int i = 0; i < spatial_size; i++) mean += input[c * spatial_size + i];
            mean /= spatial_size;
            
            var = 0;
            for (int i = 0; i < spatial_size; i++) {
                float diff = input[c * spatial_size + i] - mean;
                var += diff * diff;
            }
            var /= spatial_size;
            
            if (batch_mean) batch_mean[c] = mean;
            if (batch_var) batch_var[c] = var;
            
            running_mean[c] = (1.0f - CNN_BN_MOMENTUM) * running_mean[c] + CNN_BN_MOMENTUM * mean;
            running_var[c] = (1.0f - CNN_BN_MOMENTUM) * running_var[c] + CNN_BN_MOMENTUM * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        
        float inv_std = 1.0f / sqrtf(var + CNN_BN_EPSILON);
        float g = gamma[c];
        float b = beta[c];
        
        // Fused normalize + ReLU in single pass
        for (int i = 0; i < spatial_size; i++) {
            int idx = c * spatial_size + i;
            float val = g * (input[idx] - mean) * inv_std + b;
            if (pre_relu) pre_relu[idx] = val;  // Save pre-ReLU for backward
            output[idx] = val > 0 ? val : 0;    // ReLU fused
        }
    }
}

// =============================================================================
// BATCH NORMALIZATION BACKWARD PASS
// =============================================================================

// NOTE: batch_norm_backward() is defined in cnn_training.c with OpenMP parallelization
// for training efficiency. See cnn_training.c for implementation.

