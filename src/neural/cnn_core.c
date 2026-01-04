/**
 * cnn_core.c - CNN Core Initialization and Memory Management
 * 
 * Contains: random_normal, relu, tanh_act, alloc_weights, cnn_init, cnn_free
 * 
 * Batch norm functions moved to cnn_batch_norm.c
 * Encoding functions moved to cnn_encode.c
 */

#include "dama/neural/cnn.h"
#include "dama/common/rng.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Generate random normal using Box-Muller transform
float random_normal(void) {
    RNG *rng = rng_global();
    float u = rng_f32(rng) * 2.0f - 1.0f;
    float v = rng_f32(rng) * 2.0f - 1.0f;
    float r = u * u + v * v;
    if (r == 0 || r >= 1) return random_normal();
    return u * sqrtf(-2.0f * logf(r) / r);
}

float relu(float x) { return x > 0 ? x : 0; }
float tanh_act(float x) { return tanhf(x); }

// =============================================================================
// INITIALIZATION & MEMORY
// =============================================================================

static float* alloc_weights(size_t size) {
    return (float*)calloc(size, sizeof(float));
}

void cnn_init(CNNWeights *w) {
    // Convolutional layers
    w->conv1_w = alloc_weights(64 * CNN_INPUT_CHANNELS * 3 * 3); w->conv1_b = alloc_weights(64);
    w->conv2_w = alloc_weights(64 * 64 * 3 * 3); w->conv2_b = alloc_weights(64);
    w->conv3_w = alloc_weights(64 * 64 * 3 * 3); w->conv3_b = alloc_weights(64);
    w->conv4_w = alloc_weights(64 * 64 * 3 * 3); w->conv4_b = alloc_weights(64);
    
    // Batch normalization parameters
    w->bn1_gamma = alloc_weights(64); w->bn1_beta = alloc_weights(64);
    w->bn2_gamma = alloc_weights(64); w->bn2_beta = alloc_weights(64);
    w->bn3_gamma = alloc_weights(64); w->bn3_beta = alloc_weights(64);
    w->bn4_gamma = alloc_weights(64); w->bn4_beta = alloc_weights(64);
    
    // Batch normalization running statistics
    w->bn1_mean = alloc_weights(64); w->bn1_var = alloc_weights(64);
    w->bn2_mean = alloc_weights(64); w->bn2_var = alloc_weights(64);
    w->bn3_mean = alloc_weights(64); w->bn3_var = alloc_weights(64);
    w->bn4_mean = alloc_weights(64); w->bn4_var = alloc_weights(64);
    
    // Initialize BN gamma=1, beta=0, var=1
    for(int i=0; i<64; i++) {
        w->bn1_gamma[i] = w->bn2_gamma[i] = w->bn3_gamma[i] = w->bn4_gamma[i] = 1.0f;
        w->bn1_var[i] = w->bn2_var[i] = w->bn3_var[i] = w->bn4_var[i] = 1.0f;
    }

    // Fully connected layers
    w->policy_w = alloc_weights(512 * 4097); w->policy_b = alloc_weights(512);
    w->value_w1 = alloc_weights(256 * 4097); w->value_b1 = alloc_weights(256);
    w->value_w2 = alloc_weights(1 * 256); w->value_b2 = alloc_weights(1);

    // Gradients
    w->d_conv1_w = alloc_weights(64 * CNN_INPUT_CHANNELS * 3 * 3); w->d_conv1_b = alloc_weights(64);
    w->d_conv2_w = alloc_weights(64 * 64 * 3 * 3); w->d_conv2_b = alloc_weights(64);
    w->d_conv3_w = alloc_weights(64 * 64 * 3 * 3); w->d_conv3_b = alloc_weights(64);
    w->d_conv4_w = alloc_weights(64 * 64 * 3 * 3); w->d_conv4_b = alloc_weights(64);
    w->d_bn1_gamma = alloc_weights(64); w->d_bn1_beta = alloc_weights(64);
    w->d_bn2_gamma = alloc_weights(64); w->d_bn2_beta = alloc_weights(64);
    w->d_bn3_gamma = alloc_weights(64); w->d_bn3_beta = alloc_weights(64);
    w->d_bn4_gamma = alloc_weights(64); w->d_bn4_beta = alloc_weights(64);
    w->d_policy_w = alloc_weights(512 * 4097); w->d_policy_b = alloc_weights(512);
    w->d_value_w1 = alloc_weights(256 * 4097); w->d_value_b1 = alloc_weights(256);
    w->d_value_w2 = alloc_weights(1 * 256); w->d_value_b2 = alloc_weights(1);

    // Momentum
    w->v_conv1_w = alloc_weights(64 * CNN_INPUT_CHANNELS * 3 * 3); w->v_conv1_b = alloc_weights(64);
    w->v_conv2_w = alloc_weights(64 * 64 * 3 * 3); w->v_conv2_b = alloc_weights(64);
    w->v_conv3_w = alloc_weights(64 * 64 * 3 * 3); w->v_conv3_b = alloc_weights(64);
    w->v_conv4_w = alloc_weights(64 * 64 * 3 * 3); w->v_conv4_b = alloc_weights(64);
    w->v_bn1_gamma = alloc_weights(64); w->v_bn1_beta = alloc_weights(64);
    w->v_bn2_gamma = alloc_weights(64); w->v_bn2_beta = alloc_weights(64);
    w->v_bn3_gamma = alloc_weights(64); w->v_bn3_beta = alloc_weights(64);
    w->v_bn4_gamma = alloc_weights(64); w->v_bn4_beta = alloc_weights(64);
    w->v_policy_w = alloc_weights(512 * 4097); w->v_policy_b = alloc_weights(512);
    w->v_value_w1 = alloc_weights(256 * 4097); w->v_value_b1 = alloc_weights(256);
    w->v_value_w2 = alloc_weights(1 * 256); w->v_value_b2 = alloc_weights(1);

    // He initialization for Conv layers
    float scale1 = sqrtf(2.0f / (CNN_INPUT_CHANNELS * 3 * 3));
    for (int i=0; i<64*CNN_INPUT_CHANNELS*9; i++) w->conv1_w[i] = random_normal() * scale1;
    float scale2 = sqrtf(2.0f / (64 * 3 * 3));
    for (int i=0; i<64*64*9; i++) {
        w->conv2_w[i] = random_normal() * scale2;
        w->conv3_w[i] = random_normal() * scale2;
        w->conv4_w[i] = random_normal() * scale2;
    }
    
    // Xavier initialization for FC layers
    float scale_fc = sqrtf(1.0f / 4097);
    for (int i=0; i<512*4097; i++) w->policy_w[i] = random_normal() * scale_fc;
    for (int i=0; i<256*4097; i++) w->value_w1[i] = random_normal() * scale_fc;
    float scale_v2 = sqrtf(1.0f / 256);
    for (int i=0; i<256; i++) w->value_w2[i] = random_normal() * scale_v2;
}

void cnn_free(CNNWeights *w) {
    // Convolutional layers
    free(w->conv1_w); free(w->conv1_b);
    free(w->conv2_w); free(w->conv2_b);
    free(w->conv3_w); free(w->conv3_b);
    free(w->conv4_w); free(w->conv4_b);
    
    // Batch normalization
    free(w->bn1_gamma); free(w->bn1_beta);
    free(w->bn2_gamma); free(w->bn2_beta);
    free(w->bn3_gamma); free(w->bn3_beta);
    free(w->bn4_gamma); free(w->bn4_beta);
    free(w->bn1_mean); free(w->bn1_var);
    free(w->bn2_mean); free(w->bn2_var);
    free(w->bn3_mean); free(w->bn3_var);
    free(w->bn4_mean); free(w->bn4_var);
    
    // Fully connected layers
    free(w->policy_w); free(w->policy_b);
    free(w->value_w1); free(w->value_b1);
    free(w->value_w2); free(w->value_b2);
    
    // Gradients
    free(w->d_conv1_w); free(w->d_conv1_b);
    free(w->d_conv2_w); free(w->d_conv2_b);
    free(w->d_conv3_w); free(w->d_conv3_b);
    free(w->d_conv4_w); free(w->d_conv4_b);
    free(w->d_bn1_gamma); free(w->d_bn1_beta);
    free(w->d_bn2_gamma); free(w->d_bn2_beta);
    free(w->d_bn3_gamma); free(w->d_bn3_beta);
    free(w->d_bn4_gamma); free(w->d_bn4_beta);
    free(w->d_policy_w); free(w->d_policy_b);
    free(w->d_value_w1); free(w->d_value_b1);
    free(w->d_value_w2); free(w->d_value_b2);
    
    // Momentum
    free(w->v_conv1_w); free(w->v_conv1_b);
    free(w->v_conv2_w); free(w->v_conv2_b);
    free(w->v_conv3_w); free(w->v_conv3_b);
    free(w->v_conv4_w); free(w->v_conv4_b);
    free(w->v_bn1_gamma); free(w->v_bn1_beta);
    free(w->v_bn2_gamma); free(w->v_bn2_beta);
    free(w->v_bn3_gamma); free(w->v_bn3_beta);
    free(w->v_bn4_gamma); free(w->v_bn4_beta);
    free(w->v_policy_w); free(w->v_policy_b);
    free(w->v_value_w1); free(w->v_value_b1);
    free(w->v_value_w2); free(w->v_value_b2);
}
