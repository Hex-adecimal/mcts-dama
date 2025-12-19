/**
 * trainer.c - Neural Network Training Tool
 * 
 * Self-play data generation + training loop for PUCT.
 * Run with: make trainer && ./bin/trainer
 */

#include "../src/game.h"
#include "../src/mcts.h"
#include "../src/nn.h"
#include "../src/params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// TRAINING CONFIGURATION (from params.h)
// =============================================================================
// Hyperparameters are defined in params.h:
//   NN_NUM_ITERATIONS, NN_GAMES_PER_ITERATION, NN_BATCH_SIZE, 
//   NN_LEARNING_RATE, NN_MAX_SAMPLES, NN_CHECKPOINT_INTERVAL, NN_MOMENTUM

#define WEIGHTS_FILE        "bin/nn_weights.bin"

// =============================================================================
// SELF-PLAY DATA GENERATION
// =============================================================================

/**
 * Play one game with MCTS+PUCT and collect training data.
 * 
 * @param weights       Current neural network weights
 * @param buffer        Output buffer for training samples
 * @param buffer_offset Current position in buffer
 * @param max_samples   Maximum samples to collect
 * @return              Number of samples collected from this game
 */
static int play_self_play_game(NNWeights *weights, TrainingSample *buffer, 
                               int buffer_offset, int max_samples) {
    // Temporary storage for game positions
    GameState game_states[MAX_GAME_TURNS];
    float game_policies[MAX_GAME_TURNS][NN_OUTPUT_SIZE];
    int game_length = 0;
    
    // Initialize game
    GameState state;
    init_game(&state);
    
    // Setup MCTS config with PUCT enabled
    MCTSConfig config = {
        .ucb1_c = UCB1_C,
        .rollout_epsilon = DEFAULT_ROLLOUT_EPSILON,
        .draw_score = DRAW_SCORE,
        .expansion_threshold = EXPANSION_THRESHOLD,
        .verbose = 0,
        .use_lookahead = DEFAULT_USE_LOOKAHEAD,
        .use_tree_reuse = 0,  // Disable for training
        .use_ucb1_tuned = 0,
        .use_tt = 0,
        .use_solver = 0,
        .use_progressive_bias = 0,
        .use_fpu = 1,
        .fpu_value = FPU_VALUE,
        .use_decaying_reward = 0,
        .weights = {
            .w_capture = W_CAPTURE,
            .w_promotion = W_PROMOTION,
            .w_advance = W_ADVANCE,
            .w_center = W_CENTER,
            .w_edge = W_EDGE,
            .w_base = W_BASE,
            .w_threat = W_THREAT,
            .w_lady_activity = W_LADY_ACTIVITY
        },
        .use_puct = 1,
        .puct_c = PUCT_C,
        .nn_weights = weights
    };
    
    // Play game loop
    while (game_length < MAX_GAME_TURNS) {
        // Generate moves to check if game is over
        MoveList moves;
        generate_moves(&state, &moves);
        
        if (moves.count == 0) {
            break;  // Game over - current player loses
        }
        
        // Check for draw
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            break;  // Draw
        }
        
        // Store current state
        game_states[game_length] = state;
        
        // Clear policy array
        memset(game_policies[game_length], 0, sizeof(game_policies[game_length]));
        
        // Run MCTS search
        Arena arena;
        arena_init(&arena, ARENA_SIZE_TUNER);
        Node *root = mcts_create_root(state, &arena, config);
        Move best = mcts_search(root, &arena, TIME_TUNER, config, NULL, NULL);
        
        // Extract visit counts as policy target (normalized)
        if (root->visits > 0 && root->num_children > 0) {
            for (int i = 0; i < root->num_children; i++) {
                int idx = nn_move_to_index(&root->children[i]->move_from_parent);
                if (idx >= 0 && idx < NN_OUTPUT_SIZE) {
                    game_policies[game_length][idx] = 
                        (float)root->children[i]->visits / (float)root->visits;
                }
            }
        }
        
        game_length++;
        apply_move(&state, &best);
        arena_free(&arena);
    }
    
    // Determine game result
    // The last player to move won if opponent has no moves
    MoveList final_moves;
    generate_moves(&state, &final_moves);
    
    float result;
    if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
        result = 0.0f;  // Draw
    } else if (final_moves.count == 0) {
        // Current player has no moves = lost
        // Last move was by opponent, so from perspective of last mover: +1
        result = 1.0f;  // Last player to move won
    } else {
        result = 0.0f;  // Shouldn't happen, but treat as draw
    }
    
    // Store samples with alternating perspective
    int samples_stored = 0;
    for (int i = game_length - 1; i >= 0 && buffer_offset + samples_stored < max_samples; i--) {
        buffer[buffer_offset + samples_stored].state = game_states[i];
        memcpy(buffer[buffer_offset + samples_stored].target_policy, 
               game_policies[i], NN_OUTPUT_SIZE * sizeof(float));
        buffer[buffer_offset + samples_stored].target_value = result;
        
        result = -result;  // Flip perspective for opponent's turn
        samples_stored++;
    }
    
    return samples_stored;
}

/**
 * Generate training data from multiple self-play games.
 * Sequential for reliability (rand() is not thread-safe).
 */
static int generate_self_play_data(NNWeights *weights, TrainingSample *buffer, int games) {
    int total_samples = 0;
    
    for (int g = 0; g < games && total_samples < NN_MAX_SAMPLES; g++) {
        int collected = play_self_play_game(weights, buffer, total_samples, NN_MAX_SAMPLES - total_samples);
        total_samples += collected;
        
        printf("    Game %d/%d: %d samples (total: %d)\n", g + 1, games, collected, total_samples);
    }
    
    return total_samples;
}

// =============================================================================
// TRAINING LOOP
// =============================================================================

/**
 * Shuffle training samples (Fisher-Yates)
 */
static void shuffle_samples(TrainingSample *buffer, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        TrainingSample temp = buffer[i];
        buffer[i] = buffer[j];
        buffer[j] = temp;
    }
}

/**
 * Train on buffer of samples, returns average loss
 */
extern float g_last_policy_loss, g_last_value_loss;  // From nn.c

static float train_on_buffer(NNWeights *weights, TrainingSample *buffer, int count) {
    shuffle_samples(buffer, count);
    
    int num_batches = count / NN_BATCH_SIZE;
    float total_loss = 0.0f;
    float total_policy_loss = 0.0f;
    float total_value_loss = 0.0f;
    
    for (int b = 0; b < num_batches; b++) {
        float batch_loss = nn_train_step(weights, &buffer[b * NN_BATCH_SIZE], NN_BATCH_SIZE, NN_LEARNING_RATE);
        total_loss += batch_loss;
        total_policy_loss += g_last_policy_loss;
        total_value_loss += g_last_value_loss;
    }
    
    float avg_loss = (num_batches > 0) ? total_loss / num_batches : 0.0f;
    float avg_policy = (num_batches > 0) ? total_policy_loss / num_batches : 0.0f;
    float avg_value = (num_batches > 0) ? total_value_loss / num_batches : 0.0f;
    
    printf("  Training: %d batches | Policy: %.3f | Value: %.3f | Total: %.3f\n", 
           num_batches, avg_policy, avg_value, avg_loss);
    
    return avg_loss;
}

// =============================================================================
// MAIN
// =============================================================================

int main(void) {
    printf("=== PUCT Neural Network Trainer ===\n\n");
    
    // Seed RNG
    srand((unsigned)time(NULL));
    
    // Initialize Zobrist keys
    zobrist_init();
    init_move_tables();
    
    // Initialize neural network
    NNWeights weights;
    nn_init(&weights, NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    printf("Network initialized: %d -> %d -> %d\n", 
           NN_INPUT_SIZE, NN_HIDDEN_SIZE, NN_OUTPUT_SIZE);
    
    // Try to load existing weights
    if (nn_load_weights(&weights, WEIGHTS_FILE) == 0) {
        printf("Loaded existing weights from %s\n", WEIGHTS_FILE);
    } else {
        printf("Starting with random weights\n");
    }
    
    // Allocate sample buffer
    TrainingSample *buffer = malloc(NN_MAX_SAMPLES * sizeof(TrainingSample));
    if (!buffer) {
        fprintf(stderr, "Failed to allocate sample buffer\n");
        return 1;
    }
    
    // Main training loop
    printf("\nStarting training...\n\n");
    
    for (int iter = 0; iter < NN_NUM_ITERATIONS; iter++) {
        printf("=== Iteration %d/%d ===\n", iter + 1, NN_NUM_ITERATIONS);
        
        // 1. Generate self-play data
        printf("Generating self-play data...\n");
        int num_samples = generate_self_play_data(&weights, buffer, NN_GAMES_PER_ITERATION);
        printf("  Collected %d samples\n", num_samples);
        
        // 2. Train on collected data
        printf("Training...\n");
        if (num_samples >= NN_BATCH_SIZE) {
            train_on_buffer(&weights, buffer, num_samples);
        } else {
            printf("  Skipping (not enough samples)\n");
        }
        
        // 3. Checkpoint
        if ((iter + 1) % NN_CHECKPOINT_INTERVAL == 0) {
            printf("Saving checkpoint...\n");
            if (nn_save_weights(&weights, WEIGHTS_FILE) == 0) {
                printf("  Saved to %s\n", WEIGHTS_FILE);
            }
        }
        
        printf("\n");
    }
    
    // Final save
    printf("Saving final weights...\n");
    nn_save_weights(&weights, WEIGHTS_FILE);
    
    // Cleanup
    free(buffer);
    nn_free(&weights);
    
    printf("\nTraining complete!\n");
    return 0;
}
