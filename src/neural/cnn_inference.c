/**
 * cnn_inference.c - Forward Pass and Move Mapping for CNN
 */

#include "dama/neural/cnn.h"
#include "dama/neural/conv_ops.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <Accelerate/Accelerate.h>

// =============================================================================
// FORWARD PASS
// =============================================================================

// NOTE: cnn_forward() without history has been removed.
// Use cnn_forward_with_history() for all inference with proper history support.

void cnn_forward_with_history(const CNNWeights *w, const GameState *state, 
                            const GameState *hist1, const GameState *hist2, 
                            CNNOutput *out) {
    float player = 1.0f;  // Canonical form: always "my turn"
    float input[CNN_INPUT_CHANNELS * 64];
    memset(input, 0, sizeof(input));
    
    extern void encode_state_channels_canonical(const GameState *state, float *tensor, int channel_offset);
    encode_state_channels_canonical(state, input, 0);
    
    // Encode history with same player perspective for consistent canonical form
    if (hist1) {
        GameState h1 = *hist1;
        h1.current_player = state->current_player;
        encode_state_channels_canonical(&h1, input, 4);
    }
    if (hist2) {
        GameState h2 = *hist2;
        h2.current_player = state->current_player;
        encode_state_channels_canonical(&h2, input, 8);
    }

    // Buffers for activations
    float conv1_out[64 * 64], bn1_out[64 * 64];
    float conv2_out[64 * 64], bn2_out[64 * 64];
    float conv3_out[64 * 64], bn3_out[64 * 64];
    float conv4_out[64 * 64], bn4_out[64 * 64];
    float fc_input[4097], policy_out[512], value_h[256];

    // Layer 1 (Fused BN+ReLU)
    conv2d_forward(input, w->conv1_w, w->conv1_b, conv1_out, 8, 8, CNN_INPUT_CHANNELS, 64, 3);
    batch_norm_forward_relu(conv1_out, w->bn1_gamma, w->bn1_beta, bn1_out, NULL, NULL, NULL, w->bn1_mean, w->bn1_var, 64, 8, 8, 0);

    // Layer 2
    conv2d_forward(bn1_out, w->conv2_w, w->conv2_b, conv2_out, 8, 8, 64, 64, 3);
    batch_norm_forward_relu(conv2_out, w->bn2_gamma, w->bn2_beta, bn2_out, NULL, NULL, NULL, w->bn2_mean, w->bn2_var, 64, 8, 8, 0);

    // Layer 3
    conv2d_forward(bn2_out, w->conv3_w, w->conv3_b, conv3_out, 8, 8, 64, 64, 3);
    batch_norm_forward_relu(conv3_out, w->bn3_gamma, w->bn3_beta, bn3_out, NULL, NULL, NULL, w->bn3_mean, w->bn3_var, 64, 8, 8, 0);

    // Layer 4
    conv2d_forward(bn3_out, w->conv4_w, w->conv4_b, conv4_out, 8, 8, 64, 64, 3);
    batch_norm_forward_relu(conv4_out, w->bn4_gamma, w->bn4_beta, bn4_out, NULL, NULL, NULL, w->bn4_mean, w->bn4_var, 64, 8, 8, 0);

    // Flatten + Player
    memcpy(fc_input, bn4_out, 4096 * sizeof(float));
    fc_input[4096] = player;

    // Policy Head (BLAS optimized)
    memcpy(policy_out, w->policy_b, 512 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 512, 4097, 1.0f, 
                w->policy_w, 4097, fc_input, 1, 1.0f, policy_out, 1);
    
    // Softmax (vDSP vectorized)
    float max_p;
    vDSP_maxv(policy_out, 1, &max_p, 512);
    float neg_max = -max_p;
    vDSP_vsadd(policy_out, 1, &neg_max, policy_out, 1, 512);
    int n512 = 512;
    vvexpf(policy_out, policy_out, &n512);
    float sum_exp;
    vDSP_sve(policy_out, 1, &sum_exp, 512);
    vDSP_vsdiv(policy_out, 1, &sum_exp, out->policy, 1, 512);

    // Value Head (BLAS optimized)
    memcpy(value_h, w->value_b1, 256 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 256, 4097, 1.0f,
                w->value_w1, 4097, fc_input, 1, 1.0f, value_h, 1);
    for (int i = 0; i < 256; i++) value_h[i] = relu(value_h[i]);
    float v_sum = w->value_b2[0];
    for (int i = 0; i < 256; i++) v_sum += value_h[i] * w->value_w2[i];
    out->value = tanh_act(v_sum);
}

// =============================================================================
// BATCH FORWARD PASS
// =============================================================================

/**
 * Batch forward pass for multiple states. Uses BLAS sgemm for FC layers.
 * Convolutions are still per-sample (no efficient batch conv in Accelerate).
 * 
 * @param w        Network weights
 * @param states   Array of game states (batch_size)
 * @param hist1s   Array of history-1 states (can have NULLs)
 * @param hist2s   Array of history-2 states (can have NULLs)
 * @param outs     Output array (batch_size)
 * @param batch_size  Number of samples in batch
 */
void cnn_forward_batch(const CNNWeights *w, 
                       const GameState **states,
                       const GameState **hist1s,
                       const GameState **hist2s,
                       CNNOutput *outs, 
                       int batch_size) {
    if (batch_size <= 0) return;
    
    // Fallback to single forward if batch is small
    if (batch_size == 1) {
        cnn_forward_with_history(w, states[0], 
                                hist1s ? hist1s[0] : NULL, 
                                hist2s ? hist2s[0] : NULL, 
                                &outs[0]);
        return;
    }
    
    // Allocate batch buffers (stack for small batches, heap for large)
    #define MAX_STACK_BATCH 16
    float _stack_fc[MAX_STACK_BATCH * 4097];
    float _stack_policy[MAX_STACK_BATCH * 512];
    float _stack_value_h[MAX_STACK_BATCH * 256];
    
    float *fc_inputs = (batch_size <= MAX_STACK_BATCH) ? _stack_fc : malloc(batch_size * 4097 * sizeof(float));
    float *policy_outs = (batch_size <= MAX_STACK_BATCH) ? _stack_policy : malloc(batch_size * 512 * sizeof(float));
    float *value_hs = (batch_size <= MAX_STACK_BATCH) ? _stack_value_h : malloc(batch_size * 256 * sizeof(float));
    
    // Check for memory allocation failure - fallback to sequential processing on OOM
    if (batch_size > MAX_STACK_BATCH && (!fc_inputs || !policy_outs || !value_hs)) {
        free(fc_inputs);
        free(policy_outs);
        free(value_hs);
        for (int b = 0; b < batch_size; b++) {
            cnn_forward_with_history(w, states[b], 
                hist1s ? hist1s[b] : NULL, 
                hist2s ? hist2s[b] : NULL, &outs[b]);
        }
        return;
    }
    
    extern void encode_state_channels_canonical(const GameState *state, float *tensor, int channel_offset);
    
    // Per-sample convolutions (Accelerate doesn't batch these efficiently yet)
    // Future optimization: layout transformation for im2col batching
    for (int b = 0; b < batch_size; b++) {
        float input[CNN_INPUT_CHANNELS * 64];
        memset(input, 0, sizeof(input));
        encode_state_channels_canonical(states[b], input, 0);
        
        if (hist1s && hist1s[b]) {
            GameState h1 = *hist1s[b];
            h1.current_player = states[b]->current_player;
            encode_state_channels_canonical(&h1, input, 4);
        }
        if (hist2s && hist2s[b]) {
            GameState h2 = *hist2s[b];
            h2.current_player = states[b]->current_player;
            encode_state_channels_canonical(&h2, input, 8);
        }
        
        float conv1_out[64 * 64], bn1_out[64 * 64];
        float conv2_out[64 * 64], bn2_out[64 * 64];
        float conv3_out[64 * 64], bn3_out[64 * 64];
        float conv4_out[64 * 64], bn4_out[64 * 64];
        
        conv2d_forward(input, w->conv1_w, w->conv1_b, conv1_out, 8, 8, CNN_INPUT_CHANNELS, 64, 3);
        batch_norm_forward_relu(conv1_out, w->bn1_gamma, w->bn1_beta, bn1_out, NULL, NULL, NULL, w->bn1_mean, w->bn1_var, 64, 8, 8, 0);
        
        conv2d_forward(bn1_out, w->conv2_w, w->conv2_b, conv2_out, 8, 8, 64, 64, 3);
        batch_norm_forward_relu(conv2_out, w->bn2_gamma, w->bn2_beta, bn2_out, NULL, NULL, NULL, w->bn2_mean, w->bn2_var, 64, 8, 8, 0);
        
        conv2d_forward(bn2_out, w->conv3_w, w->conv3_b, conv3_out, 8, 8, 64, 64, 3);
        batch_norm_forward_relu(conv3_out, w->bn3_gamma, w->bn3_beta, bn3_out, NULL, NULL, NULL, w->bn3_mean, w->bn3_var, 64, 8, 8, 0);
        
        conv2d_forward(bn3_out, w->conv4_w, w->conv4_b, conv4_out, 8, 8, 64, 64, 3);
        batch_norm_forward_relu(conv4_out, w->bn4_gamma, w->bn4_beta, bn4_out, NULL, NULL, NULL, w->bn4_mean, w->bn4_var, 64, 8, 8, 0);
        
        // Store flattened + player for batch FC
        memcpy(&fc_inputs[b * 4097], bn4_out, 4096 * sizeof(float));
        fc_inputs[b * 4097 + 4096] = 1.0f;  // Player always 1.0 in canonical form
    }
    
    // =========================================================================
    // BATCH FC: Policy Head (sgemm instead of sgemv)
    // policy_outs[batch_size x 512] = fc_inputs[batch_size x 4097] * policy_w^T[4097 x 512]
    // =========================================================================
    
    // Initialize with bias (broadcast)
    for (int b = 0; b < batch_size; b++) {
        memcpy(&policy_outs[b * 512], w->policy_b, 512 * sizeof(float));
    }
    
    // Batched matrix multiply: C = A * B^T + C
    // A = fc_inputs [batch_size x 4097]
    // B = policy_w  [512 x 4097]
    // C = policy_outs [batch_size x 512]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, 512, 4097,       // M, N, K
                1.0f,                         // alpha
                fc_inputs, 4097,              // A, lda
                w->policy_w, 4097,            // B, ldb
                1.0f,                         // beta
                policy_outs, 512);            // C, ldc
    
    // Per-sample softmax and copy to output
    for (int b = 0; b < batch_size; b++) {
        float *p = &policy_outs[b * 512];
        float max_p;
        vDSP_maxv(p, 1, &max_p, 512);
        float neg_max = -max_p;
        vDSP_vsadd(p, 1, &neg_max, p, 1, 512);
        int n512 = 512;
        vvexpf(p, p, &n512);
        float sum_exp;
        vDSP_sve(p, 1, &sum_exp, 512);
        vDSP_vsdiv(p, 1, &sum_exp, outs[b].policy, 1, 512);
    }
    
    // =========================================================================
    // BATCH FC: Value Head (sgemm for hidden layer)
    // value_hs[batch_size x 256] = fc_inputs[batch_size x 4097] * value_w1^T[4097 x 256]
    // =========================================================================
    
    for (int b = 0; b < batch_size; b++) {
        memcpy(&value_hs[b * 256], w->value_b1, 256 * sizeof(float));
    }
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, 256, 4097,
                1.0f,
                fc_inputs, 4097,
                w->value_w1, 4097,
                1.0f,
                value_hs, 256);
    
    // ReLU + final linear per-sample (value_w2 is just 256 -> 1)
    for (int b = 0; b < batch_size; b++) {
        float *vh = &value_hs[b * 256];
        for (int i = 0; i < 256; i++) vh[i] = vh[i] > 0 ? vh[i] : 0;  // ReLU
        
        float v_sum = w->value_b2[0];
        for (int i = 0; i < 256; i++) v_sum += vh[i] * w->value_w2[i];
        outs[b].value = tanhf(v_sum);
    }
    
    // Cleanup if heap allocated
    if (batch_size > MAX_STACK_BATCH) {
        free(fc_inputs);
        free(policy_outs);
        free(value_hs);
    }
    #undef MAX_STACK_BATCH
}

void cnn_forward_sample(const CNNWeights *w, const TrainingSample *sample, CNNOutput *out) {
    float player;
    float input[CNN_INPUT_CHANNELS * 64];
    cnn_encode_sample(sample, input, &player);

    // Buffers for activations
    float conv1_out[64 * 64], bn1_out[64 * 64];
    float conv2_out[64 * 64], bn2_out[64 * 64];
    float conv3_out[64 * 64], bn3_out[64 * 64];
    float conv4_out[64 * 64], bn4_out[64 * 64];
    float fc_input[4097], policy_out[512], value_h[256];

    // Layer 1 (Fused BN+ReLU)
    conv2d_forward(input, w->conv1_w, w->conv1_b, conv1_out, 8, 8, CNN_INPUT_CHANNELS, 64, 3);
    batch_norm_forward_relu(conv1_out, w->bn1_gamma, w->bn1_beta, bn1_out, NULL, NULL, NULL, w->bn1_mean, w->bn1_var, 64, 8, 8, 0);

    // Layer 2
    conv2d_forward(bn1_out, w->conv2_w, w->conv2_b, conv2_out, 8, 8, 64, 64, 3);
    batch_norm_forward_relu(conv2_out, w->bn2_gamma, w->bn2_beta, bn2_out, NULL, NULL, NULL, w->bn2_mean, w->bn2_var, 64, 8, 8, 0);

    // Layer 3
    conv2d_forward(bn2_out, w->conv3_w, w->conv3_b, conv3_out, 8, 8, 64, 64, 3);
    batch_norm_forward_relu(conv3_out, w->bn3_gamma, w->bn3_beta, bn3_out, NULL, NULL, NULL, w->bn3_mean, w->bn3_var, 64, 8, 8, 0);

    // Layer 4
    conv2d_forward(bn3_out, w->conv4_w, w->conv4_b, conv4_out, 8, 8, 64, 64, 3);
    batch_norm_forward_relu(conv4_out, w->bn4_gamma, w->bn4_beta, bn4_out, NULL, NULL, NULL, w->bn4_mean, w->bn4_var, 64, 8, 8, 0);

    // Flatten + Player
    memcpy(fc_input, bn4_out, 4096 * sizeof(float));
    fc_input[4096] = player;

    // Policy Head (BLAS optimized)
    memcpy(policy_out, w->policy_b, 512 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 512, 4097, 1.0f, 
                w->policy_w, 4097, fc_input, 1, 1.0f, policy_out, 1);
    
    // Softmax (vDSP vectorized)
    float max_p;
    vDSP_maxv(policy_out, 1, &max_p, 512);
    float neg_max = -max_p;
    vDSP_vsadd(policy_out, 1, &neg_max, policy_out, 1, 512);
    int n512 = 512;
    vvexpf(policy_out, policy_out, &n512);
    float sum_exp;
    vDSP_sve(policy_out, 1, &sum_exp, 512);
    vDSP_vsdiv(policy_out, 1, &sum_exp, out->policy, 1, 512);

    // Value Head (BLAS optimized)
    memcpy(value_h, w->value_b1, 256 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 256, 4097, 1.0f,
                w->value_w1, 4097, fc_input, 1, 1.0f, value_h, 1);
    for (int i = 0; i < 256; i++) value_h[i] = relu(value_h[i]);
    float v_sum = w->value_b2[0];
    for (int i = 0; i < 256; i++) v_sum += value_h[i] * w->value_w2[i];
    out->value = tanh_act(v_sum);
}

// =============================================================================
// MOVE MAPPING
// =============================================================================

int get_move_direction(int from, int to) {
    int df = (to % 8) - (from % 8);
    int dr = (to / 8) - (from / 8);
    if (dr == 1 && df == 1) return 0; // Up-Right
    if (dr == 1 && df == -1) return 1; // Up-Left
    if (dr == -1 && df == 1) return 2; // Down-Right
    if (dr == -1 && df == -1) return 3; // Down-Left
    if (dr == 2 && df == 2) return 4; // Jump Up-Right
    if (dr == 2 && df == -2) return 5; // Jump Up-Left
    if (dr == -2 && df == 2) return 6; // Jump Down-Right
    if (dr == -2 && df == -2) return 7; // Jump Down-Left
    return -1;
}

int cnn_move_to_index(const Move *move, int color) {
    int from = move->path[0];
    int to = (move->length == 0) ? move->path[1] : move->path[1]; // First jump defines direction
    int dir = get_move_direction(from, to);
    if (dir == -1) return -1;
    
    /**
     * CANONICAL MAPPING:
     * When it's Black's turn, the input board is flipped vertically.
     * To maintain consistency, we must also flip:
     * 1. Directions (Up becomes Down, and vice versa)
     * 2. The source square (Must match the flipped board perspective)
     */
    if (color == BLACK) {
        // 1. Flip Directions
        if (dir < 2) dir += 2; else if (dir < 4) dir -= 2;          // Walk moves
        else if (dir < 6) dir += 2; else if (dir < 8) dir -= 2;    // Jump moves
        
        // 2. Flip Source Square
        from = flip_square(from);
    }
    
    return from * 8 + dir;
}

float cnn_get_move_prior(const CNNWeights *w, const GameState *state, 
                         const GameState *hist1, const GameState *hist2,
                         const Move *move) {
    CNNOutput out;
    cnn_forward_with_history(w, state, hist1, hist2, &out);
    int idx = cnn_move_to_index(move, state->current_player);
    return (idx >= 0) ? out.policy[idx] : 0.0f;
}
