/**
 * nn.c - Neural Network Module Implementation
 * 
 */

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// =============================================================================
// HELPER FUNCTIONS (Implemented for you!)
// =============================================================================

/**
 * ReLU activation: max(0, x)
 */
static inline float relu(float x) {
    return x > 0 ? x : 0;
}

/**
 * ReLU derivative: 1 if x > 0, else 0
 */
static inline float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

/**
 * Tanh activation (for value head): maps to [-1, 1]
 */
static inline float tanh_activation(float x) {
    return tanhf(x);
}

/**
 * Random float in range [-limit, limit] for weight init
 */
static float random_uniform(float limit) {
    return ((float)rand() / RAND_MAX) * 2.0f * limit - limit;
}

// =============================================================================
// INITIALIZATION & MEMORY
// =============================================================================

void nn_init(NNWeights *w, int input_size, int hidden_size, int output_size) {
    w->input_size = input_size;
    w->hidden_size = hidden_size;
    w->output_size = output_size;
    
    // =========================================================================
    // TODO: Allocate weight arrays
    // =========================================================================
    // Hint: Use malloc(rows * cols * sizeof(float))
    
    w->w1 = malloc(input_size * hidden_size * sizeof(float));
    w->b1 = malloc(hidden_size * sizeof(float));

    w->w2 = malloc(hidden_size * hidden_size * sizeof(float));
    w->b2 = malloc(hidden_size * sizeof(float));

    w->wp = malloc(hidden_size * output_size * sizeof(float));
    w->bp = malloc(output_size * sizeof(float));

    w->wv = malloc(hidden_size * sizeof(float));
    w->bv = malloc(sizeof(float));

    w->dw1 = malloc(input_size * hidden_size * sizeof(float));
    w->db1 = malloc(hidden_size * sizeof(float));

    w->dw2 = malloc(hidden_size * hidden_size * sizeof(float));
    w->db2 = malloc(hidden_size * sizeof(float));

    w->dwp = malloc(hidden_size * output_size * sizeof(float));
    w->dbp = malloc(output_size * sizeof(float));

    w->dwv = malloc(hidden_size * sizeof(float));
    w->dbv = malloc(sizeof(float));
    
    // Allocate velocity arrays (for momentum)
    w->vw1 = calloc(input_size * hidden_size, sizeof(float));
    w->vb1 = calloc(hidden_size, sizeof(float));
    w->vw2 = calloc(hidden_size * hidden_size, sizeof(float));
    w->vb2 = calloc(hidden_size, sizeof(float));
    w->vwp = calloc(hidden_size * output_size, sizeof(float));
    w->vbp = calloc(output_size, sizeof(float));
    w->vwv = calloc(hidden_size, sizeof(float));
    w->vbv = calloc(1, sizeof(float));
    
    // =========================================================================
    // Initialize weights with Xavier initialization
    // =========================================================================

    float limit = sqrtf(6.0f / (input_size + hidden_size));
    
    for (int i = 0; i < input_size * hidden_size; i++) {
        w->w1[i] = random_uniform(limit);
    }
    for (int i = 0; i < hidden_size; i++) {
        w->b1[i] = 0.0f;
    }
    
    limit = sqrtf(6.0f / (hidden_size + hidden_size));
    for (int i = 0; i < hidden_size * hidden_size; i++) {
        w->w2[i] = random_uniform(limit);
    }
    for (int i = 0; i < hidden_size; i++) {
        w->b2[i] = 0.0f;
    }
    
    limit = sqrtf(6.0f / (hidden_size + output_size));
    for (int i = 0; i < hidden_size * output_size; i++) {
        w->wp[i] = random_uniform(limit);
    }
    for (int i = 0; i < output_size; i++) {
        w->bp[i] = 0.0f;
    }
    
    limit = sqrtf(6.0f / (hidden_size + 1));
    for (int i = 0; i < hidden_size; i++) {
        w->wv[i] = random_uniform(limit);
    }
    w->bv[0] = 0.0f;
}

void nn_free(NNWeights *w) {
    free(w->w1); free(w->b1);
    free(w->w2); free(w->b2);
    free(w->wp); free(w->bp);
    free(w->wv); free(w->bv);
    free(w->dw1); free(w->db1);
    free(w->dw2); free(w->db2);
    free(w->dwp); free(w->dbp);
    free(w->dwv); free(w->dbv);
    free(w->vw1); free(w->vb1);
    free(w->vw2); free(w->vb2);
    free(w->vwp); free(w->vbp);
    free(w->vwv); free(w->vbv);
}

// =============================================================================
// PERSISTENCE
// =============================================================================

int nn_save_weights(const NNWeights *w, const char *filename) {
    // =========================================================================
    // TODO: Save weights to binary file
    // =========================================================================
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    fwrite(w->w1, sizeof(float), w->input_size * w->hidden_size, f);
    fwrite(w->b1, sizeof(float), w->hidden_size, f);

    fwrite(w->w2, sizeof(float), w->hidden_size * w->hidden_size, f);
    fwrite(w->b2, sizeof(float), w->hidden_size, f);

    fwrite(w->wp, sizeof(float), w->hidden_size * w->output_size, f);
    fwrite(w->bp, sizeof(float), w->output_size, f);

    fwrite(w->wv, sizeof(float), w->hidden_size, f);
    fwrite(w->bv, sizeof(float), 1, f);
    
    fclose(f);
    return 0;
}

int nn_load_weights(NNWeights *w, const char *filename) {
    // =========================================================================
    // TODO: Load weights from binary file
    // =========================================================================
    // FILE *f = fopen(filename, "rb");
    // if (!f) return -1;
    // 
    // fread(w->w1, sizeof(float), w->input_size * w->hidden_size, f);
    // ... same order as save
    // 
    // fclose(f);
    // return 0;

    FILE *f = fopen(filename, "rb");
    if (!f) return -1;

    fread(w->w1, sizeof(float), w->input_size * w->hidden_size, f);
    fread(w->b1, sizeof(float), w->hidden_size, f);

    fread(w->w2, sizeof(float), w->hidden_size * w->hidden_size, f);
    fread(w->b2, sizeof(float), w->hidden_size, f);

    fread(w->wp, sizeof(float), w->hidden_size * w->output_size, f);
    fread(w->bp, sizeof(float), w->output_size, f);

    fread(w->wv, sizeof(float), w->hidden_size, f);
    fread(w->bv, sizeof(float), 1, f);
    
    fclose(f);
    return 0;
}

// =============================================================================
// INFERENCE
// =============================================================================

void nn_encode_state(const GameState *state, float *features) {
    // =========================================================================
    // TODO: Convert GameState to feature vector
    // =========================================================================
    // Option: Use bitboards directly

    
    memset(features, 0, NN_INPUT_SIZE * sizeof(float));
    
    int idx = 0;
    for (int sq = 0; sq < 64; sq++) {
        int row = sq / 8;
        int col = sq % 8;
        if ((row + col) % 2 == 1) {  // Solo caselle scure
            if (check_bit(state->white_pieces, sq)) features[idx] = 1.0f;
            if (check_bit(state->white_ladies, sq)) features[32 + idx] = 1.0f;
            if (check_bit(state->black_pieces, sq)) features[64 + idx] = 1.0f;
            if (check_bit(state->black_ladies, sq)) features[96 + idx] = 1.0f;
            idx++;
        }
    }
    
    features[128] = (state->current_player == WHITE) ? 1.0f : -1.0f;
}

void nn_forward(const NNWeights *w, const GameState *state, NNOutput *out) {
    // =========================================================================
    // TODO: Implement forward pass
    // =========================================================================
    
    // Encode state
    float features[NN_INPUT_SIZE];
    nn_encode_state(state, features);

    // Layer 1 (Input → Hidden1)
    float h1[NN_HIDDEN_SIZE];
    for (int j=0; j<w->hidden_size; j++) {
        float sum = w->b1[j];
        for (int i=0; i<w->input_size; i++) {
            sum += features[i] * w->w1[i * w->hidden_size + j];
        }
        h1[j] = relu(sum);
    }

    // Layer 2 (Hidden1 → Hidden2)
    float h2[NN_HIDDEN_SIZE];
    for (int j=0; j<w->hidden_size; j++) {
        float sum = w->b2[j];
        for (int i=0; i<w->hidden_size; i++) {
            sum += h1[i] * w->w2[i * w->hidden_size + j];
        }
        h2[j] = relu(sum);
    }

    // Policy head (softmax)
    float logits[MAX_MOVES];
    for (int a = 0; a < w->output_size; a++) {
        float sum = w->bp[a];
        for (int j = 0; j < w->hidden_size; j++) {
            sum += h2[j] * w->wp[j * w->output_size + a];
        }
        logits[a] = sum;
    }
    
    // Apply softmax:
    float max_logit = logits[0];
    for (int a = 1; a < w->output_size; a++) {
        if (logits[a] > max_logit) max_logit = logits[a];
    }
    float sum_exp = 0;
    for (int a = 0; a < w->output_size; a++) {
        out->policy[a] = expf(logits[a] - max_logit);
        sum_exp += out->policy[a];
    }
    for (int a = 0; a < w->output_size; a++) {
        out->policy[a] /= sum_exp;
    }
    
    // Step 5: Value head (tanh)
    float v_sum = w->bv[0];
    for (int j = 0; j < w->hidden_size; j++) {
        v_sum += h2[j] * w->wv[j];
    }
    out->value = tanh_activation(v_sum);

}

// =============================================================================
// MOVE ↔ POLICY INDEX MAPPING
// =============================================================================

int nn_move_to_index(const Move *move) {
    int from = move->path[0];
    int to = (move->length == 0) ? move->path[1] : move->path[move->length];
    return (from * 8 + (to % 8)) % 64;
}

float nn_get_move_prior(const NNWeights *w, const GameState *state, const Move *move) {
    NNOutput out;
    nn_forward(w, state, &out);
     
    int idx = nn_move_to_index(move);
    return out.policy[idx];
}

// =============================================================================
// TRAINING
// =============================================================================

void nn_zero_gradients(NNWeights *w) {
    memset(w->dw1, 0, w->input_size * w->hidden_size * sizeof(float));
    memset(w->db1, 0, w->hidden_size * sizeof(float));

    memset(w->dw2, 0, w->hidden_size * w->hidden_size * sizeof(float));
    memset(w->db2, 0, w->hidden_size * sizeof(float));

    memset(w->dwp, 0, w->hidden_size * w->output_size * sizeof(float));
    memset(w->dbp, 0, w->output_size * sizeof(float));

    memset(w->dwv, 0, w->hidden_size * sizeof(float));
    memset(w->dbv, 0, sizeof(float));
}

void nn_backward(NNWeights *w, const TrainingSample *sample, const NNOutput *pred) {
    // =========================================================================
    // BACKPROPAGATION - Guida step-by-step
    // =========================================================================
    //
    // L'obiettivo è calcolare come ogni peso influenza l'errore (loss).
    // Usiamo la CHAIN RULE per propagare i gradienti all'indietro.
    //
    // Loss = L_policy + L_value
    // L_policy = CrossEntropy(pred.policy, target.policy)
    // L_value  = MSE(pred.value, target.value)
    //
    // =========================================================================

    // -------------------------------------------------------------------------
    // STEP 0: Ricalcola le attivazioni intermedie (serve per i gradienti)
    // -------------------------------------------------------------------------
    // Dobbiamo rifare il forward pass per avere h1, h2, h1_pre, h2_pre
    
    float features[NN_INPUT_SIZE];
    nn_encode_state(&sample->state, features);
    
    // Layer 1: z1 = W1*x + b1,  h1 = ReLU(z1)
    float h1_pre[NN_HIDDEN_SIZE];  // Pre-attivazione (prima di ReLU)
    float h1[NN_HIDDEN_SIZE];      // Post-attivazione (dopo ReLU)
    for (int j = 0; j < w->hidden_size; j++) {
        float sum = w->b1[j];
        for (int i = 0; i < w->input_size; i++) {
            sum += features[i] * w->w1[i * w->hidden_size + j];
        }
        h1_pre[j] = sum;
        h1[j] = relu(sum);
    }
    
    // Layer 2: z2 = W2*h1 + b2,  h2 = ReLU(z2)
    float h2_pre[NN_HIDDEN_SIZE];
    float h2[NN_HIDDEN_SIZE];
    for (int j = 0; j < w->hidden_size; j++) {
        float sum = w->b2[j];
        for (int i = 0; i < w->hidden_size; i++) {
            sum += h1[i] * w->w2[i * w->hidden_size + j];
        }
        h2_pre[j] = sum;
        h2[j] = relu(sum);
    }

    // -------------------------------------------------------------------------
    // STEP 1: Gradiente della POLICY HEAD (Cross-Entropy + Softmax)
    // -------------------------------------------------------------------------
    // Per softmax + cross-entropy, il gradiente è semplicemente:
    // d_logit[a] = pred.policy[a] - target.policy[a]
    //
    // Derivazione matematica:
    // L = -Σ target[a] * log(pred[a])
    // ∂L/∂logit[a] = pred[a] - target[a]  (formula elegante!)
    
    float d_policy[NN_OUTPUT_SIZE];
    for (int a = 0; a < w->output_size; a++) {
        d_policy[a] = pred->policy[a] - sample->target_policy[a];
    }

    // -------------------------------------------------------------------------
    // STEP 2: Gradiente della VALUE HEAD (MSE + Tanh)
    // -------------------------------------------------------------------------
    // L = (pred - target)²
    // ∂L/∂pred = 2 * (pred - target)
    //
    // Ma pred = tanh(z), quindi:
    // ∂L/∂z = ∂L/∂pred * ∂pred/∂z = 2*(pred-target) * (1 - tanh²)
    
    float d_value = 2.0f * (pred->value - sample->target_value) 
                        * (1.0f - pred->value * pred->value);

    // -------------------------------------------------------------------------
    // STEP 3: Accumula gradienti per POLICY HEAD (Wp, bp)
    // -------------------------------------------------------------------------
    // policy_logit[a] = Σ_j h2[j] * Wp[j,a] + bp[a]
    //
    // ∂L/∂Wp[j,a] = h2[j] * d_policy[a]
    // ∂L/∂bp[a]   = d_policy[a]
    
    float d_h2[NN_HIDDEN_SIZE];  // Gradiente che arriva a h2
    memset(d_h2, 0, sizeof(d_h2));
    
    for (int j = 0; j < w->hidden_size; j++) {
        for (int a = 0; a < w->output_size; a++) {
            w->dwp[j * w->output_size + a] += h2[j] * d_policy[a];
            d_h2[j] += w->wp[j * w->output_size + a] * d_policy[a];
        }
    }
    for (int a = 0; a < w->output_size; a++) {
        w->dbp[a] += d_policy[a];
    }

    // -------------------------------------------------------------------------
    // STEP 4: Accumula gradienti per VALUE HEAD (Wv, bv)
    // -------------------------------------------------------------------------
    // value_logit = Σ_j h2[j] * Wv[j] + bv
    //
    // ∂L/∂Wv[j] = h2[j] * d_value
    // ∂L/∂bv    = d_value
    
    for (int j = 0; j < w->hidden_size; j++) {
        w->dwv[j] += h2[j] * d_value;
        d_h2[j] += w->wv[j] * d_value;  // Accumula gradiente da value head
    }
    w->dbv[0] += d_value;

    // -------------------------------------------------------------------------
    // STEP 5: Backprop attraverso ReLU di Layer 2
    // -------------------------------------------------------------------------
    // ReLU'(x) = 1 se x > 0, altrimenti 0
    // Il gradiente passa solo se l'attivazione era > 0
    
    for (int j = 0; j < w->hidden_size; j++) {
        d_h2[j] *= relu_derivative(h2_pre[j]);
    }

    // -------------------------------------------------------------------------
    // STEP 6: Accumula gradienti per Layer 2 (W2, b2)
    // -------------------------------------------------------------------------
    // z2[j] = Σ_i h1[i] * W2[i,j] + b2[j]
    //
    // ∂L/∂W2[i,j] = h1[i] * d_h2[j]
    // ∂L/∂b2[j]   = d_h2[j]
    
    float d_h1[NN_HIDDEN_SIZE];
    memset(d_h1, 0, sizeof(d_h1));
    
    for (int i = 0; i < w->hidden_size; i++) {
        for (int j = 0; j < w->hidden_size; j++) {
            w->dw2[i * w->hidden_size + j] += h1[i] * d_h2[j];
            d_h1[i] += w->w2[i * w->hidden_size + j] * d_h2[j];
        }
    }
    for (int j = 0; j < w->hidden_size; j++) {
        w->db2[j] += d_h2[j];
    }

    // -------------------------------------------------------------------------
    // STEP 7: Backprop attraverso ReLU di Layer 1
    // -------------------------------------------------------------------------
    for (int i = 0; i < w->hidden_size; i++) {
        d_h1[i] *= relu_derivative(h1_pre[i]);
    }

    // -------------------------------------------------------------------------
    // STEP 8: Accumula gradienti per Layer 1 (W1, b1)
    // -------------------------------------------------------------------------
    // z1[j] = Σ_i x[i] * W1[i,j] + b1[j]
    //
    // ∂L/∂W1[i,j] = x[i] * d_h1[j]
    // ∂L/∂b1[j]   = d_h1[j]
    
    for (int i = 0; i < w->input_size; i++) {
        for (int j = 0; j < w->hidden_size; j++) {
            w->dw1[i * w->hidden_size + j] += features[i] * d_h1[j];
        }
    }
    for (int j = 0; j < w->hidden_size; j++) {
        w->db1[j] += d_h1[j];
    }
}

void nn_update_weights(NNWeights *w, float learning_rate, int batch_size) {
    // =========================================================================
    // SGD with Momentum: v = β*v + grad,  w = w - lr*v
    // =========================================================================
    const float momentum = 0.9f;  // Standard momentum coefficient
    float scale = 1.0f / batch_size;
    
    // Layer 1
    for (int i = 0; i < w->input_size * w->hidden_size; i++) {
        w->vw1[i] = momentum * w->vw1[i] + scale * w->dw1[i];
        w->w1[i] -= learning_rate * w->vw1[i];
    }
    for (int i = 0; i < w->hidden_size; i++) {
        w->vb1[i] = momentum * w->vb1[i] + scale * w->db1[i];
        w->b1[i] -= learning_rate * w->vb1[i];
    }
    
    // Layer 2
    for (int i = 0; i < w->hidden_size * w->hidden_size; i++) {
        w->vw2[i] = momentum * w->vw2[i] + scale * w->dw2[i];
        w->w2[i] -= learning_rate * w->vw2[i];
    }
    for (int i = 0; i < w->hidden_size; i++) {
        w->vb2[i] = momentum * w->vb2[i] + scale * w->db2[i];
        w->b2[i] -= learning_rate * w->vb2[i];
    }
    
    // Policy head
    for (int i = 0; i < w->hidden_size * w->output_size; i++) {
        w->vwp[i] = momentum * w->vwp[i] + scale * w->dwp[i];
        w->wp[i] -= learning_rate * w->vwp[i];
    }
    for (int i = 0; i < w->output_size; i++) {
        w->vbp[i] = momentum * w->vbp[i] + scale * w->dbp[i];
        w->bp[i] -= learning_rate * w->vbp[i];
    }
    
    // Value head
    for (int i = 0; i < w->hidden_size; i++) {
        w->vwv[i] = momentum * w->vwv[i] + scale * w->dwv[i];
        w->wv[i] -= learning_rate * w->vwv[i];
    }
    w->vbv[0] = momentum * w->vbv[0] + scale * w->dbv[0];
    w->bv[0] -= learning_rate * w->vbv[0];
}

float nn_train_step(NNWeights *w, TrainingSample *batch, int batch_size, float learning_rate) {
    nn_zero_gradients(w);
    
    float total_policy_loss = 0.0f;
    float total_value_loss = 0.0f;
    
    for (int i = 0; i < batch_size; i++) {
        NNOutput pred;
        nn_forward(w, &batch[i].state, &pred);
        
        // Policy loss: Cross-entropy = -Σ target * log(pred)
        float policy_loss = 0.0f;
        for (int a = 0; a < w->output_size; a++) {
            if (batch[i].target_policy[a] > 0.0f) {
                float log_pred = logf(pred.policy[a] + 1e-8f);  // Avoid log(0)
                policy_loss -= batch[i].target_policy[a] * log_pred;
            }
        }
        
        // Value loss: MSE = (pred - target)²
        float value_loss = (pred.value - batch[i].target_value) * 
                          (pred.value - batch[i].target_value);
        
        total_policy_loss += policy_loss;
        total_value_loss += value_loss;
        
        nn_backward(w, &batch[i], &pred);
    }
    
    nn_update_weights(w, learning_rate, batch_size);
    
    // Store separate losses in static variables for trainer to read
    extern float g_last_policy_loss, g_last_value_loss;
    g_last_policy_loss = total_policy_loss / batch_size;
    g_last_value_loss = total_value_loss / batch_size;
    
    return (total_policy_loss + total_value_loss) / batch_size;
}

// Global variables for loss tracking (read by trainer)
float g_last_policy_loss = 0.0f;
float g_last_value_loss = 0.0f;
