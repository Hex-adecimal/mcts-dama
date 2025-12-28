/**
 * cnn_core.c - Infrastructure, Initialization, and Encoding for CNN
 */

#include "cnn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

float random_normal(void) {
    float u = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    float v = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    float r = u * u + v * v;
    if (r == 0 || r >= 1) return random_normal();
    return u * sqrtf(-2.0f * logf(r) / r);
}

float relu(float x) { return x > 0 ? x : 0; }
float tanh_act(float x) { return tanhf(x); }

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
// INITIALIZATION & MEMORY
// =============================================================================

static float* alloc_weights(size_t size) {
    return (float*)calloc(size, sizeof(float));
}

void cnn_init(CNNWeights *w) {
    // Allocation logic (shortened for brevity in this plan, but full in code)
    w->conv1_w = alloc_weights(64 * CNN_INPUT_CHANNELS * 3 * 3); w->conv1_b = alloc_weights(64);
    w->conv2_w = alloc_weights(64 * 64 * 3 * 3); w->conv2_b = alloc_weights(64);
    w->conv3_w = alloc_weights(64 * 64 * 3 * 3); w->conv3_b = alloc_weights(64);
    w->conv4_w = alloc_weights(64 * 64 * 3 * 3); w->conv4_b = alloc_weights(64);
    
    w->bn1_gamma = alloc_weights(64); w->bn1_beta = alloc_weights(64);
    w->bn2_gamma = alloc_weights(64); w->bn2_beta = alloc_weights(64);
    w->bn3_gamma = alloc_weights(64); w->bn3_beta = alloc_weights(64);
    w->bn4_gamma = alloc_weights(64); w->bn4_beta = alloc_weights(64);
    
    w->bn1_mean = alloc_weights(64); w->bn1_var = alloc_weights(64);
    w->bn2_mean = alloc_weights(64); w->bn2_var = alloc_weights(64);
    w->bn3_mean = alloc_weights(64); w->bn3_var = alloc_weights(64);
    w->bn4_mean = alloc_weights(64); w->bn4_var = alloc_weights(64);
    
    for(int i=0; i<64; i++) {
        w->bn1_gamma[i] = w->bn2_gamma[i] = w->bn3_gamma[i] = w->bn4_gamma[i] = 1.0f;
        w->bn1_var[i] = w->bn2_var[i] = w->bn3_var[i] = w->bn4_var[i] = 1.0f;
    }

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

    // He initialization for Conv, Xavier for FC
    float scale1 = sqrtf(2.0f / (CNN_INPUT_CHANNELS * 3 * 3));
    for (int i=0; i<64*CNN_INPUT_CHANNELS*9; i++) w->conv1_w[i] = random_normal() * scale1;
    float scale2 = sqrtf(2.0f / (64 * 3 * 3));
    for (int i=0; i<64*64*9; i++) {
        w->conv2_w[i] = random_normal() * scale2;
        w->conv3_w[i] = random_normal() * scale2;
        w->conv4_w[i] = random_normal() * scale2;
    }
    float scale_fc = sqrtf(1.0f / 4097);
    for (int i=0; i<512*4097; i++) w->policy_w[i] = random_normal() * scale_fc;
    for (int i=0; i<256*4097; i++) w->value_w1[i] = random_normal() * scale_fc;
    float scale_v2 = sqrtf(1.0f / 256);
    for (int i=0; i<256; i++) w->value_w2[i] = random_normal() * scale_v2;
}

void cnn_free(CNNWeights *w) {
    free(w->conv1_w); free(w->conv1_b);
    free(w->conv2_w); free(w->conv2_b);
    free(w->conv3_w); free(w->conv3_b);
    free(w->conv4_w); free(w->conv4_b);
    free(w->bn1_gamma); free(w->bn1_beta);
    free(w->bn2_gamma); free(w->bn2_beta);
    free(w->bn3_gamma); free(w->bn3_beta);
    free(w->bn4_gamma); free(w->bn4_beta);
    free(w->bn1_mean); free(w->bn1_var);
    free(w->bn2_mean); free(w->bn2_var);
    free(w->bn3_mean); free(w->bn3_var);
    free(w->bn4_mean); free(w->bn4_var);
    free(w->policy_w); free(w->policy_b);
    free(w->value_w1); free(w->value_b1);
    free(w->value_w2); free(w->value_b2);
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

// =============================================================================
// STATE ENCODING (CANONICAL FORM)
// =============================================================================

// Flip a square index vertically (row 0 <-> row 7, etc.)
static inline int flip_square(int sq) {
    int row = sq / 8;
    int col = sq % 8;
    return (7 - row) * 8 + col;
}

// Encode state in CANONICAL FORM: board is always from current player's perspective
// - Channel 0: "my" pawns
// - Channel 1: "my" ladies  
// - Channel 2: "opponent" pawns
// - Channel 3: "opponent" ladies
// When Black's turn: flip board vertically so Black "starts from bottom"
void encode_state_channels_canonical(const GameState *state, float *tensor, int channel_offset) {
    int is_white = (state->current_player == WHITE);
    
    for (int sq = 0; sq < 64; sq++) {
        int target_sq = is_white ? sq : flip_square(sq);
        
        if (is_white) {
            // White's perspective: White = "my", Black = "opponent"
            if (check_bit(state->white_pieces, sq)) tensor[channel_offset * 64 + target_sq] = 1.0f;
            if (check_bit(state->white_ladies, sq)) tensor[(channel_offset + 1) * 64 + target_sq] = 1.0f;
            if (check_bit(state->black_pieces, sq)) tensor[(channel_offset + 2) * 64 + target_sq] = 1.0f;
            if (check_bit(state->black_ladies, sq)) tensor[(channel_offset + 3) * 64 + target_sq] = 1.0f;
        } else {
            // Black's perspective: Black = "my", White = "opponent" (flipped board)
            if (check_bit(state->black_pieces, sq)) tensor[channel_offset * 64 + target_sq] = 1.0f;
            if (check_bit(state->black_ladies, sq)) tensor[(channel_offset + 1) * 64 + target_sq] = 1.0f;
            if (check_bit(state->white_pieces, sq)) tensor[(channel_offset + 2) * 64 + target_sq] = 1.0f;
            if (check_bit(state->white_ladies, sq)) tensor[(channel_offset + 3) * 64 + target_sq] = 1.0f;
        }
    }
}

// Legacy function for compatibility (no canonical form)
void encode_state_channels(const GameState *state, float *tensor, int channel_offset) {
    encode_state_channels_canonical(state, tensor, channel_offset);
}

void cnn_encode_state(const GameState *state, float *tensor, float *player) {
    memset(tensor, 0, CNN_INPUT_CHANNELS * 64 * sizeof(float));
    encode_state_channels_canonical(state, tensor, 0);
    *player = 1.0f;  // Always "my turn" in canonical form
}

void cnn_encode_sample(const TrainingSample *sample, float *tensor, float *player) {
    memset(tensor, 0, CNN_INPUT_CHANNELS * 64 * sizeof(float));
    encode_state_channels_canonical(&sample->state, tensor, 0);
    // Note: History states need same canonical transform as main state
    
    // Temporarily set history player to match main state for consistent encoding
    GameState hist0 = sample->history[0];
    GameState hist1 = sample->history[1];
    hist0.current_player = sample->state.current_player;
    hist1.current_player = sample->state.current_player;
    
    encode_state_channels_canonical(&hist0, tensor, 4);
    encode_state_channels_canonical(&hist1, tensor, 8);
    *player = 1.0f;  // Always 1.0 in canonical form (it's always "my turn")
}

// =============================================================================
// WEIGHT I/O
// =============================================================================

void cnn_save_weights(const CNNWeights *w, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(w->conv1_w, sizeof(float), 64*CNN_INPUT_CHANNELS*9, f); fwrite(w->conv1_b, sizeof(float), 64, f);
    fwrite(w->conv2_w, sizeof(float), 64*64*9, f); fwrite(w->conv2_b, sizeof(float), 64, f);
    fwrite(w->conv3_w, sizeof(float), 64*64*9, f); fwrite(w->conv3_b, sizeof(float), 64, f);
    fwrite(w->conv4_w, sizeof(float), 64*64*9, f); fwrite(w->conv4_b, sizeof(float), 64, f);
    fwrite(w->bn1_gamma, sizeof(float), 64, f); fwrite(w->bn1_beta, sizeof(float), 64, f);
    fwrite(w->bn2_gamma, sizeof(float), 64, f); fwrite(w->bn2_beta, sizeof(float), 64, f);
    fwrite(w->bn3_gamma, sizeof(float), 64, f); fwrite(w->bn3_beta, sizeof(float), 64, f);
    fwrite(w->bn4_gamma, sizeof(float), 64, f); fwrite(w->bn4_beta, sizeof(float), 64, f);
    fwrite(w->bn1_mean, sizeof(float), 64, f); fwrite(w->bn1_var, sizeof(float), 64, f);
    fwrite(w->bn2_mean, sizeof(float), 64, f); fwrite(w->bn2_var, sizeof(float), 64, f);
    fwrite(w->bn3_mean, sizeof(float), 64, f); fwrite(w->bn3_var, sizeof(float), 64, f);
    fwrite(w->bn4_mean, sizeof(float), 64, f); fwrite(w->bn4_var, sizeof(float), 64, f);
    fwrite(w->policy_w, sizeof(float), 512*4097, f); fwrite(w->policy_b, sizeof(float), 512, f);
    fwrite(w->value_w1, sizeof(float), 256*4097, f); fwrite(w->value_b1, sizeof(float), 256, f);
    fwrite(w->value_w2, sizeof(float), 1*256, f); fwrite(w->value_b2, sizeof(float), 1, f);
    fclose(f);
}

int cnn_load_weights(CNNWeights *w, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fread(w->conv1_w, sizeof(float), 64*CNN_INPUT_CHANNELS*9, f); fread(w->conv1_b, sizeof(float), 64, f);
    fread(w->conv2_w, sizeof(float), 64*64*9, f); fread(w->conv2_b, sizeof(float), 64, f);
    fread(w->conv3_w, sizeof(float), 64*64*9, f); fread(w->conv3_b, sizeof(float), 64, f);
    fread(w->conv4_w, sizeof(float), 64*64*9, f); fread(w->conv4_b, sizeof(float), 64, f);
    fread(w->bn1_gamma, sizeof(float), 64, f); fread(w->bn1_beta, sizeof(float), 64, f);
    fread(w->bn2_gamma, sizeof(float), 64, f); fread(w->bn2_beta, sizeof(float), 64, f);
    fread(w->bn3_gamma, sizeof(float), 64, f); fread(w->bn3_beta, sizeof(float), 64, f);
    fread(w->bn4_gamma, sizeof(float), 64, f); fread(w->bn4_beta, sizeof(float), 64, f);
    fread(w->bn1_mean, sizeof(float), 64, f); fread(w->bn1_var, sizeof(float), 64, f);
    fread(w->bn2_mean, sizeof(float), 64, f); fread(w->bn2_var, sizeof(float), 64, f);
    fread(w->bn3_mean, sizeof(float), 64, f); fread(w->bn3_var, sizeof(float), 64, f);
    fread(w->bn4_mean, sizeof(float), 64, f); fread(w->bn4_var, sizeof(float), 64, f);
    fread(w->policy_w, sizeof(float), 512*4097, f); fread(w->policy_b, sizeof(float), 512, f);
    fread(w->value_w1, sizeof(float), 256*4097, f); fread(w->value_b1, sizeof(float), 256, f);
    fread(w->value_w2, sizeof(float), 1*256, f); fread(w->value_b2, sizeof(float), 1, f);
    fclose(f);
    return 0;
}
