/**
 * cnn_types.h - CNN Core Types and Constants
 * 
 * Extracted from cnn.h for better modularity.
 * Contains: Architecture constants, CNNWeights, CNNOutput
 */

#ifndef CNN_TYPES_H
#define CNN_TYPES_H

#include "dama/engine/game.h"

// =============================================================================
// ARCHITECTURE CONSTANTS (derived from params.h)
// =============================================================================

#define CNN_INPUT_CHANNELS  (CNN_HISTORY_T * CNN_PIECE_CHANNELS)  // 12 channels total
#define CNN_CONV1_CHANNELS  64  // Wider first layer for better feature capture
#define CNN_CONV2_CHANNELS  64
#define CNN_CONV3_CHANNELS  64
#define CNN_CONV4_CHANNELS  64  // 4th layer for 9x9 receptive field
#define CNN_KERNEL_SIZE     3
#define CNN_FLATTEN_SIZE    (CNN_BOARD_SIZE * CNN_BOARD_SIZE * CNN_CONV4_CHANNELS)  // 4096
#define CNN_FC_INPUT_SIZE   (CNN_FLATTEN_SIZE + 1)  // 4097

// Batch Normalization constants
#define CNN_BN_EPSILON     1e-5f   // Numerical stability
#define CNN_BN_MOMENTUM    0.1f    // Running stats update rate

// =============================================================================
// STRUCTURES
// =============================================================================

/**
 * CNN Weights structure containing all learnable parameters.
 */
typedef struct {
    // === Convolutional Layers ===
    // Conv1: 4 → 64 channels
    float *conv1_w;     // [64][4][3][3] = 64*4*9 = 2304
    float *conv1_b;     // [64]
    
    // Conv2: 64 → 64 channels
    float *conv2_w;     // [64][64][3][3] = 64*64*9 = 36864
    float *conv2_b;     // [64]
    
    // Conv3: 64 → 64 channels
    float *conv3_w;     // [64][64][3][3] = 64*64*9 = 36864
    float *conv3_b;     // [64]
    
    // Conv4: 64 → 64 channels (9x9 receptive field)
    float *conv4_w;     // [64][64][3][3] = 64*64*9 = 36864
    float *conv4_b;     // [64]
    
    // === Batch Normalization Layers ===
    // BN params: gamma (scale), beta (shift)
    float *bn1_gamma, *bn1_beta;  // [64] after Conv1
    float *bn2_gamma, *bn2_beta;  // [64] after Conv2
    float *bn3_gamma, *bn3_beta;  // [64] after Conv3
    float *bn4_gamma, *bn4_beta;  // [64] after Conv4
    
    // BN running statistics (for inference)
    float *bn1_mean, *bn1_var;    // [64]
    float *bn2_mean, *bn2_var;    // [64]
    float *bn3_mean, *bn3_var;    // [64]
    float *bn4_mean, *bn4_var;    // [64]
    
    // === Policy Head ===
    float *policy_w;    // [256][4097]
    float *policy_b;    // [256]
    
    // === Value Head ===
    float *value_w1;    // [64][4097]
    float *value_b1;    // [64]
    float *value_w2;    // [1][64]
    float *value_b2;    // [1]
    
    // === Gradients (d_ prefix) ===
    float *d_conv1_w, *d_conv1_b;
    float *d_conv2_w, *d_conv2_b;
    float *d_conv3_w, *d_conv3_b;
    float *d_conv4_w, *d_conv4_b;
    float *d_bn1_gamma, *d_bn1_beta;  // BN gradients
    float *d_bn2_gamma, *d_bn2_beta;
    float *d_bn3_gamma, *d_bn3_beta;
    float *d_bn4_gamma, *d_bn4_beta;
    float *d_policy_w, *d_policy_b;
    float *d_value_w1, *d_value_b1;
    float *d_value_w2, *d_value_b2;
    
    // === Momentum (v_ prefix) for SGD ===
    float *v_conv1_w, *v_conv1_b;
    float *v_conv2_w, *v_conv2_b;
    float *v_conv3_w, *v_conv3_b;
    float *v_conv4_w, *v_conv4_b;
    float *v_bn1_gamma, *v_bn1_beta;  // BN momentum
    float *v_bn2_gamma, *v_bn2_beta;
    float *v_bn3_gamma, *v_bn3_beta;
    float *v_bn4_gamma, *v_bn4_beta;
    float *v_policy_w, *v_policy_b;
    float *v_value_w1, *v_value_b1;
    float *v_value_w2, *v_value_b2;
    
} CNNWeights;

/**
 * CNN Output containing policy distribution and value.
 */
typedef struct {
    float policy[CNN_POLICY_SIZE];  // Move probabilities
    float value;                     // Position value [-1, 1]
} CNNOutput;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Flip a square index vertically (row 0 <-> row 7, etc.)
static inline int flip_square(int sq) {
    int row = sq / 8;
    int col = sq % 8;
    return (7 - row) * 8 + col;
}

#endif // CNN_TYPES_H
