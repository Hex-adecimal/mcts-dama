/**
 * nn.h - Neural Network Module for PUCT
 * 
 * MLP (Multi-Layer Perceptron) for policy and value prediction.
 * Used with PUCT selection in MCTS.
 */

#ifndef NN_H
#define NN_H

#include "game.h"
#include "params.h"
#include <stdlib.h>

// =============================================================================
// NETWORK CONFIGURATION
// =============================================================================

#define NN_INPUT_SIZE   129   // Encoded board features (32*4 + 1 for player)
#define NN_HIDDEN_SIZE  256   // Hidden layer width
#define NN_OUTPUT_SIZE  MAX_MOVES    // Policy output size (64)

// =============================================================================
// STRUCTURES
// =============================================================================

/**
 * MLP weight storage.
 * Architecture: Input → Hidden1 → Hidden2 → (Policy, Value)
 */
typedef struct {
    // Layer 1: Input → Hidden1
    float *w1;      // [input_size × hidden_size]
    float *b1;      // [hidden_size]
    
    // Layer 2: Hidden1 → Hidden2
    float *w2;      // [hidden_size × hidden_size]
    float *b2;      // [hidden_size]
    
    // Policy Head: Hidden2 → Policy logits
    float *wp;      // [hidden_size × output_size]
    float *bp;      // [output_size]
    
    // Value Head: Hidden2 → Scalar
    float *wv;      // [hidden_size × 1]
    float *bv;      // [1]
    
    // Dimensions
    int input_size;
    int hidden_size;
    int output_size;
    
    // =========================================================================
    // GRADIENT STORAGE (for training)
    // =========================================================================
    float *dw1, *db1;
    float *dw2, *db2;
    float *dwp, *dbp;
    float *dwv, *dbv;
    
    // =========================================================================
    // VELOCITY STORAGE (for momentum SGD)
    // =========================================================================
    float *vw1, *vb1;
    float *vw2, *vb2;
    float *vwp, *vbp;
    float *vwv, *vbv;
    
} NNWeights;

/**
 * Network output: policy distribution + value estimate.
 */
typedef struct {
    float policy[MAX_MOVES];   // Prior probabilities P(s,a) per move
    float value;               // State value V(s) in range [-1, 1]
} NNOutput;

/**
 * Training sample for supervised learning.
 * Generated from self-play with MCTS.
 */
typedef struct {
    GameState state;            // Board position
    float target_policy[64];    // π = MCTS visit counts (normalized)
    float target_value;         // z = game result (+1 win, -1 loss, 0 draw)
} TrainingSample;

// =============================================================================
// INITIALIZATION & MEMORY
// =============================================================================

/**
 * Allocate and initialize network weights.
 * 
 * @param w         Pointer to NNWeights struct to initialize
 * @param input_size    Size of input feature vector
 * @param hidden_size   Width of hidden layers
 * @param output_size   Size of policy output
 * 
 * TODO: Implement weight allocation and random initialization
 *       Use Xavier/He initialization for better convergence
 */
void nn_init(NNWeights *w, int input_size, int hidden_size, int output_size);

/**
 * Free all allocated memory.
 * 
 * @param w     Pointer to NNWeights to free
 * 
 * TODO: Free all weight and gradient arrays
 */
void nn_free(NNWeights *w);

// =============================================================================
// PERSISTENCE (Save/Load)
// =============================================================================

/**
 * Save weights to binary file.
 * 
 * @param w         Weights to save
 * @param filename  Output file path
 * @return          0 on success, -1 on error
 * 
 * TODO: Write all weight arrays in order
 *       Format: [w1][b1][w2][b2][wp][bp][wv][bv]
 */
int nn_save_weights(const NNWeights *w, const char *filename);

/**
 * Load weights from binary file.
 * 
 * @param w         Weights struct (must be initialized first)
 * @param filename  Input file path
 * @return          0 on success, -1 on error
 * 
 * TODO: Read weight arrays in same order as save
 */
int nn_load_weights(NNWeights *w, const char *filename);

// =============================================================================
// INFERENCE (Forward Pass)
// =============================================================================

/**
 * Encode game state into feature vector.
 * 
 * @param state     Game state to encode
 * @param features  Output array [NN_INPUT_SIZE]
 * 
 * TODO: Design feature encoding:
 *       - Piece positions (bitboards → floats)
 *       - Ladies positions
 *       - Current player
 *       - Move count / draw proximity
 *       Example: 4 bitboards × 32 squares = 128 features
 */
void nn_encode_state(const GameState *state, float *features);

/**
 * Forward pass: compute policy and value from state.
 * 
 * @param w         Network weights
 * @param state     Game state
 * @param out       Output struct for policy + value
 * 
 * TODO: Implement forward propagation:
 *       1. Encode state → features
 *       2. h1 = ReLU(features × w1 + b1)
 *       3. h2 = ReLU(h1 × w2 + b2)
 *       4. policy = softmax(h2 × wp + bp)
 *       5. value = tanh(h2 × wv + bv)
 */
void nn_forward(const NNWeights *w, const GameState *state, NNOutput *out);

// =============================================================================
// MOVE ↔ POLICY INDEX MAPPING (Lookup Table Connection)
// =============================================================================

/**
 * Map a Move struct to policy output index.
 * 
 * @param move      Move to map
 * @return          Index in [0, NN_OUTPUT_SIZE)
 * 
 * TODO: Design consistent mapping:
 *       Option 1: Hash-based (from × 64 + to) mod 64
 *       Option 2: Canonical move ordering
 */
int nn_move_to_index(const Move *move);

/**
 * Get prior probability P(s,a) for a specific move.
 * 
 * @param w         Network weights
 * @param state     Current game state
 * @param move      Move to get prior for
 * @return          Prior probability in [0, 1]
 * 
 * TODO: Implement:
 *       1. Forward pass to get policy
 *       2. Map move to index
 *       3. Return policy[index]
 */
float nn_get_move_prior(const NNWeights *w, const GameState *state, const Move *move);

// =============================================================================
// TRAINING (Backward Pass + SGD)
// =============================================================================

/**
 * Zero all gradient accumulators.
 * 
 * @param w     Network weights with gradient storage
 * 
 * TODO: Set all dw*, db* arrays to zero
 */
void nn_zero_gradients(NNWeights *w);

/**
 * Backpropagation: compute gradients for one sample.
 * 
 * @param w         Network weights
 * @param sample    Training sample (state, target_policy, target_value)
 * @param pred      Predicted output from forward pass
 * 
 * TODO: Implement backprop:
 *       1. Compute loss gradients:
 *          - Policy: d(CrossEntropy)/d(logits)
 *          - Value: d(MSE)/d(output)
 *       2. Backprop through layers (chain rule)
 *       3. Accumulate gradients into dw*, db*
 */
void nn_backward(NNWeights *w, const TrainingSample *sample, const NNOutput *pred);

/**
 * SGD weight update.
 * 
 * @param w             Network weights
 * @param learning_rate Learning rate (e.g., 0.001)
 * @param batch_size    Size of batch (for gradient averaging)
 * 
 * TODO: For each weight array:
 *       w -= learning_rate * (dw / batch_size)
 */
void nn_update_weights(NNWeights *w, float learning_rate, int batch_size);

/**
 * Complete training step on a batch.
 * 
 * @param w             Network weights
 * @param batch         Array of training samples
 * @param batch_size    Number of samples
 * @param learning_rate Learning rate
 * @return              Average loss over the batch
 * 
 * TODO: Implement:
 *       1. nn_zero_gradients()
 *       2. For each sample: forward + backward
 *       3. nn_update_weights()
 */
float nn_train_step(NNWeights *w, TrainingSample *batch, int batch_size, float learning_rate);

#endif // NN_H
