/**
 * cnn.h - Convolutional Neural Network for Dama
 * 
 * Architecture:
 * - Input: 8×8×4 (4 channels: white_pawns, white_ladies, black_pawns, black_ladies)
 * - Conv1: 3×3, 4→64 channels, ReLU, same padding
 * - Conv2: 3×3, 64→64 channels, ReLU, same padding
 * - Conv3: 3×3, 64→64 channels, ReLU, same padding
 * - Conv4: 3×3, 64→64 channels, ReLU, same padding (9×9 receptive field!)
 * - Flatten: 8×8×64 = 4096
 * - Policy Head: FC(4096 + 1) → 512, Softmax (+1 for current player)
 * - Value Head: FC(4096 + 1) → 256 → 1, Tanh
 */

#ifndef CNN_H
#define CNN_H

#include "../core/game.h"
#include "dataset.h"

// =============================================================================
// ARCHITECTURE CONSTANTS
// =============================================================================

#define CNN_BOARD_SIZE      8
#define CNN_HISTORY_T       3   // Number of timesteps (current + 2 previous)
#define CNN_PIECE_CHANNELS  4   // white_pawns, white_ladies, black_pawns, black_ladies
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
#define CNN_POLICY_SIZE     512 // 64 squares × 8 channels (4 moves + 4 captures)
#define CNN_VALUE_HIDDEN    256  // Larger value head for better position evaluation

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
// API FUNCTIONS
// =============================================================================

/**
 * Initialize CNN weights with Xavier/He initialization.
 */
void cnn_init(CNNWeights *w);

/**
 * Free all CNN memory.
 */
void cnn_free(CNNWeights *w);

/**
 * Encode game state into 8×8×4 tensor.
 * @param state   Game state to encode
 * @param tensor  Output tensor [4][8][8] as flat array (256 floats)
 * @param player  Output: encoded current player (-1 or +1)
 */
void cnn_encode_state(const GameState *state, float *tensor, float *player);

/**
 * Forward pass: use cnn_forward_with_history() for proper history support.
 * cnn_forward() without history has been deprecated.
 */

/**
 * Encode game state with history from TrainingSample.
 */
void cnn_encode_sample(const TrainingSample *sample, float *tensor, float *player);

/**
 * Forward pass using TrainingSample (with full history encoding).
 */
void cnn_forward_sample(const CNNWeights *w, const TrainingSample *sample, CNNOutput *out);

/**
 * Zero all gradients before accumulating batch.
 */
void cnn_zero_gradients(CNNWeights *w);

/**
 * Backward pass: compute gradients for a single sample (no history).
 * Gradients are accumulated (not overwritten).
 */
void cnn_backward(
    CNNWeights *w,
    const GameState *state,
    const float *target_policy,
    float target_value
);

// NOTE: cnn_backward_sample was removed - logic is inlined in cnn_train_step

/**
 * Update weights using SGD with momentum and Elastic Net regularization (L1 + L2).
 */
void cnn_clip_gradients(CNNWeights *w, float threshold);
void cnn_update_weights(CNNWeights *w, float learning_rate, float momentum, float l1_decay, float l2_decay, int batch_size);

/**
 * Train on a batch of samples. Returns average loss.
 * Writes separate policy and value loss to out pointers if not NULL.
 */
float cnn_train_step(
    CNNWeights *w,
    const TrainingSample *batch,
    int batch_size,
    float learning_rate,
    float l1_decay,
    float l2_decay,
    float *out_policy_loss,
    float *out_value_loss
);

/**
 * Cleanup thread-local buffers after training.
 * Call once at the end of training to free memory.
 */
void cnn_training_cleanup(void);

/**
 * Save weights to binary file.
 */
void cnn_save_weights(const CNNWeights *w, const char *path);

/**
 * Load weights from binary file. Returns 0 on success, -1 on failure.
 */
int cnn_load_weights(CNNWeights *w, const char *path);

/**
 * Convert move to policy index (canonical: always relative to player).
 * @param move The move to convert.
 * @param color The color of the player moving (WHITE/BLACK).
 */
int cnn_move_to_index(const Move *move, int color);

// NEW: Inference with explicit history
void cnn_forward_with_history(const CNNWeights *w, const GameState *state, 
                            const GameState *hist1, const GameState *hist2, 
                            CNNOutput *out);

/**
 * Get prior probability for a move (for PUCT).
 */
float cnn_get_move_prior(const CNNWeights *w, const GameState *state, 
                         const GameState *hist1, const GameState *hist2,
                         const Move *move);

// =============================================================================
// INTERNAL HELPERS (Shared across modules)
// =============================================================================

float relu(float x);
float tanh_act(float x);
void batch_norm_forward(
    const float *input,
    const float *gamma, const float *beta,
    float *output,
    float *batch_mean, float *batch_var,
    float *running_mean, float *running_var,
    int C, int H, int W,
    int is_training
);

// Fused version with inline ReLU (faster, saves one memory pass)
void batch_norm_forward_relu(
    const float *input,
    const float *gamma, const float *beta,
    float *output,
    float *pre_relu,  // Output: pre-ReLU values for backward (can be NULL for inference)
    float *batch_mean, float *batch_var,
    float *running_mean, float *running_var,
    int C, int H, int W,
    int is_training
);

#endif // CNN_H
