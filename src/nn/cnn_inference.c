/**
 * cnn_inference.c - Forward Pass and Move Mapping for CNN
 */

#include "cnn.h"
#include "conv_ops.h"
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
    
    // Canonical perspective: if black, we swap directions? 
    // No, movegen already handles squares. 
    // But we want directions relative to player.
    if (color == BLACK) {
        // Swap up/down directions for canonical policy head
        if (dir < 2) dir += 2; else if (dir < 4) dir -= 2;
        else if (dir < 6) dir += 2; else if (dir < 8) dir -= 2;
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
