/**
 * cnn_overfit.c - Dedicated Single Batch Overfitting Test
 * 
 * Purpose: Verify that the network can memorize a small amount of data.
 * If this fails, there is a fundamental bug in the network or backprop code.
 * If this succeeds, the network capacity and learning algorithm are sound.
 */

#include "game.h"
#include "movegen.h"
#include "cnn.h"
#include "dataset.h"
#include "params.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define TRAIN_FILE      "data/train.bin"
#define BATCH_SIZE      128
#define NUM_EPOCHS      80
#define LEARNING_RATE   0.01f  // Faster convergence for clean batch

int main(void) {
    printf("=== CNN Overfit Test ===\n");
    printf("Goal: Loss -> 0 on a single fixed batch.\n\n");
    
    // Init random
    srand((unsigned)time(NULL));
    init_move_tables(); // Required for generate_moves
    
    // 1. Load Data
    FILE *f = fopen(TRAIN_FILE, "rb");
    if (!f) { printf("Error: No training data.\n"); return 1; }
    fclose(f);
    
    int total_samples = dataset_get_count(TRAIN_FILE);
    if (total_samples < BATCH_SIZE) {
        printf("Error: Need at least %d samples (found %d)\n", BATCH_SIZE, total_samples);
        return 1;
    }
    
    TrainingSample *all_data = malloc(total_samples * sizeof(TrainingSample));
    dataset_load(TRAIN_FILE, all_data, total_samples);
    
    // 2. Select FIXED Batch with Conflict Check
    TrainingSample *fixed_batch = malloc(BATCH_SIZE * sizeof(TrainingSample));
    int found = 0;
    
    printf("Selecting unique samples...\n");
    for (int i = 0; i < total_samples && found < BATCH_SIZE; i++) {
        
        // Check collision based ONLY on what CNN sees: Pieces + Player
        // Ignore move counters, history, etc.
        int conflict = 0;
        for (int k = 0; k < found; k++) {
            if (all_data[i].state.white_pieces == fixed_batch[k].state.white_pieces &&
                all_data[i].state.white_ladies == fixed_batch[k].state.white_ladies &&
                all_data[i].state.black_pieces == fixed_batch[k].state.black_pieces &&
                all_data[i].state.black_ladies == fixed_batch[k].state.black_ladies &&
                all_data[i].state.current_player == fixed_batch[k].state.current_player) {
                
                // FOUND DUPLICATE INPUT
                conflict = 1;
                break;
            }
        }
        
        if (!conflict) {
            fixed_batch[found++] = all_data[i];
        }
    }
    
    if (found < BATCH_SIZE) {
        printf("Warning: Could only find %d unique samples (wanted %d)\n", found, BATCH_SIZE);
        // Fill rest with copies of first to avoid crash, but warn
        for(int k=found; k<BATCH_SIZE; k++) fixed_batch[k] = fixed_batch[0];
    }
    
    printf("Selected 1 fixed batch of %d UNIQUE samples.\n", BATCH_SIZE);
    
    // 3. Init Network
    CNNWeights weights;
    cnn_init(&weights);
    
    // 4. Overfit Loop
    printf("Training for %d epochs (LR: %f)...\n", NUM_EPOCHS, LEARNING_RATE);
    printf("Epoch | Total Loss | P Loss (->0) | V Loss (->0)\n");
    printf("------|------------|--------------|--------------\n");
    
    float lr = LEARNING_RATE;
    
    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        float p_loss, v_loss;
        
        // Train step on the SAME batch every time
        // hardcode L1/L2 decay to 0.0f
        float loss = cnn_train_step(&weights, fixed_batch, BATCH_SIZE, lr, 0.0f, 0.0f, &p_loss, &v_loss);
        
        if (epoch % 10 == 0 || epoch == 1 || loss < 0.01f) {
            printf("%5d | %10.4f | %12.4f | %12.4f\n", epoch, loss, p_loss, v_loss);
        }
        
        if (loss < 0.002f) { // Aim for extremely low loss
            printf("\nSUCCESS: Network memorized the batch! (Loss < 0.002)\n");
            system("rm bin/cnn_overfit_weights.bin"); // Cleanup
            return 0;
        }
    }
    
    // 5. Verification: Visual & Illegal Move Check
    printf("\n=== VERIFICATION ===\n");
    
    // Pick first sample
    TrainingSample *s = &fixed_batch[0];
    CNNOutput out;
    cnn_forward(&weights, &s->state, &out);
    
    printf("\n[Sample 0] Value Pred: %.4f | Target: %.4f\n", out.value, s->target_value);
    
    printf("\n--- Policy Distribution (Top 5) ---\n");
    // Find top 5 in target and network
    typedef struct { int idx; float val; } Entry;
    Entry target_top[CNN_POLICY_SIZE], net_top[CNN_POLICY_SIZE];
    for(int i=0; i<CNN_POLICY_SIZE; i++) {
        target_top[i].idx = i; target_top[i].val = s->target_policy[i];
        net_top[i].idx = i;    net_top[i].val = out.policy[i];
    }
    // Sort (naive bubble sort for CNN_POLICY_SIZE is fine)
    for(int i=0; i<5; i++) {
        for(int j=i+1; j<CNN_POLICY_SIZE; j++) {
            if (target_top[j].val > target_top[i].val) { Entry t=target_top[i]; target_top[i]=target_top[j]; target_top[j]=t; }
            if (net_top[j].val > net_top[i].val)       { Entry t=net_top[i];    net_top[i]=net_top[j];    net_top[j]=t;    }
        }
    }
    
    printf("Target Top 5: ");
    for(int i=0; i<5; i++) printf("[%d]:%.4f ", target_top[i].idx, target_top[i].val);
    printf("\n");
    
    printf("Net    Top 5: ");
    for(int i=0; i<5; i++) printf("[%d]:%.4f ", net_top[i].idx, net_top[i].val);
    printf("\n");
    
    // Illegal Move Check
    printf("\n--- Illegal Move Check ---\n");
    MoveList legal_moves;
    legal_moves.count = 0;
    generate_moves(&s->state, &legal_moves);
    
    printf("Legal moves found: %d\n", legal_moves.count);
    
    float illegal_prob_sum = 0.0f;
    float max_illegal_prob = 0.0f;
    
    for (int i = 0; i < CNN_POLICY_SIZE; i++) {
        // Check if index 'i' corresponds to a legal move
        int is_legal = 0;
        for (int m = 0; m < legal_moves.count; m++) {
            int idx = cnn_move_to_index(&legal_moves.moves[m], s->state.current_player);
            if (idx == i) {
                is_legal = 1;
                break;
            }
        }
        
        if (!is_legal) {
            illegal_prob_sum += out.policy[i];
            if (out.policy[i] > max_illegal_prob) max_illegal_prob = out.policy[i];
            
            if (out.policy[i] > 0.01f) {
                printf("WARNING: High prob on illegal move index %d: %.4f\n", i, out.policy[i]);
            }
        }
    }
    
    printf("Total Probability on Illegal Moves: %.6f\n", illegal_prob_sum);
    printf("Max Probability on Single Illegal Move: %.6f\n", max_illegal_prob);
    
    if (illegal_prob_sum > 0.05f) {
        printf("FAILED: Illegal moves have too much probability (> 5%%)\n");
    } else {
        printf("PASSED: Illegal moves have negligible probability.\n");
    }
    
    printf("\nTest Finished.\n");
    
    // Cleanup
    free(all_data);
    free(fixed_batch);
    cnn_free(&weights);
    
    return 0;
}
