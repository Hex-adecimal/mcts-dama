/**
 * cnn.h - Convolutional Neural Network for Dama
 * 
 * Main public API header. Includes cnn_types.h for structures.
 * 
 * Architecture:
 * - Input: 8×8×12 (12 channels: 3 timesteps × 4 piece types)
 * - Conv1-4: 3×3, 64 channels each, BatchNorm + ReLU
 * - Policy Head: FC(4097) → 512, Softmax
 * - Value Head: FC(4097) → 256 → 1, Tanh
 */

#ifndef CNN_H
#define CNN_H

#include "dama/neural/cnn_types.h"
#include "dama/training/dataset.h"

// =============================================================================
// API FUNCTIONS - INITIALIZATION
// =============================================================================

/**
 * Initialize CNN weights with Xavier/He initialization.
 */
void cnn_init(CNNWeights *w);

/**
 * Free all CNN memory.
 */
void cnn_free(CNNWeights *w);

// =============================================================================
// API FUNCTIONS - ENCODING
// =============================================================================

/**
 * Encode game state into 8×8×12 tensor.
 * @param state   Game state to encode
 * @param tensor  Output tensor [12][8][8] as flat array
 * @param player  Output: encoded current player (-1 or +1)
 */
void cnn_encode_state(const GameState *state, float *tensor, float *player);

/**
 * Encode game state with history from TrainingSample.
 */
void cnn_encode_sample(const TrainingSample *sample, float *tensor, float *player);

// =============================================================================
// API FUNCTIONS - INFERENCE
// =============================================================================

/**
 * Forward pass using TrainingSample (with full history encoding).
 */
void cnn_forward_sample(const CNNWeights *w, const TrainingSample *sample, CNNOutput *out);

/**
 * Forward pass with explicit history states.
 */
void cnn_forward_with_history(const CNNWeights *w, const GameState *state, 
                            const GameState *hist1, const GameState *hist2, 
                            CNNOutput *out);

/**
 * Batch forward pass for multiple states (optimized with BLAS sgemm).
 * @param w          Network weights
 * @param states     Array of game state pointers (batch_size)
 * @param hist1s     Array of history-1 state pointers (can be NULL)
 * @param hist2s     Array of history-2 state pointers (can be NULL)
 * @param outs       Output array (batch_size)
 * @param batch_size Number of samples in batch
 */
void cnn_forward_batch(const CNNWeights *w, 
                       const GameState **states,
                       const GameState **hist1s,
                       const GameState **hist2s,
                       CNNOutput *outs, 
                       int batch_size);

/**
 * Get prior probability for a move (for PUCT).
 */
float cnn_get_move_prior(const CNNWeights *w, const GameState *state, 
                         const GameState *hist1, const GameState *hist2,
                         const Move *move);

// =============================================================================
// API FUNCTIONS - TRAINING
// =============================================================================

/**
 * Zero all gradients before accumulating batch.
 */
void cnn_zero_gradients(CNNWeights *w);

/**
 * Update weights using SGD with momentum.
 */
void cnn_clip_gradients(CNNWeights *w, float threshold);
void cnn_update_weights(CNNWeights *w, float policy_lr, float value_lr, float momentum, float l1_decay, float l2_decay, int batch_size);

/**
 * Train on a batch of samples. Returns average loss.
 */
float cnn_train_step(
    CNNWeights *w,
    const TrainingSample *batch,
    int batch_size,
    float policy_lr,
    float value_lr,
    float l1_decay,
    float l2_decay,
    float *out_policy_loss,
    float *out_value_loss
);

/**
 * Cleanup thread-local buffers after training.
 */
void cnn_training_cleanup(void);

// =============================================================================
// API FUNCTIONS - I/O
// =============================================================================

/**
 * Save weights to binary file.
 */
void cnn_save_weights(const CNNWeights *w, const char *path);

/**
 * Load weights from binary file. Returns 0 on success, -1 on failure.
 */
int cnn_load_weights(CNNWeights *w, const char *path);

// =============================================================================
// API FUNCTIONS - MOVE MAPPING
// =============================================================================

/**
 * Convert move to policy index (canonical: always relative to player).
 * @param move The move to convert.
 * @param color The color of the player moving (WHITE/BLACK).
 */
int cnn_move_to_index(const Move *move, int color);

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

void batch_norm_forward_relu(
    const float *input,
    const float *gamma, const float *beta,
    float *output,
    float *pre_relu,
    float *batch_mean, float *batch_var,
    float *running_mean, float *running_var,
    int C, int H, int W,
    int is_training
);

void batch_norm_backward(
    const float *d_output, const float *input,
    const float *gamma, const float *batch_mean, const float *batch_var,
    float *d_input, float *d_gamma, float *d_beta,
    int C, int H, int W
);

// Used by encoding functions
void encode_state_channels_canonical(const GameState *state, float *tensor, int channel_offset);
void encode_state_channels(const GameState *state, float *tensor, int channel_offset);

#endif // CNN_H
