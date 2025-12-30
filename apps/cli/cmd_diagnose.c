/**
 * cmd_diagnose.c - Diagnostic tool for CNN training issues
 * 
 * Checks:
 * 1. Gradient magnitudes (are gradients flowing?)
 * 2. Activation distributions (dead neurons?)
 * 3. Policy output distribution (uniform vs peaked?)
 * 4. Target policy verification (are targets correct?)
 */

#include "dataset.h"
#include "cnn.h"
#include "game.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Forward declarations
extern void cnn_encode_sample(const TrainingSample *sample, float *tensor, float *player);

static void print_stats(const char *name, float *arr, int n) {
    float min = arr[0], max = arr[0], sum = 0, sum_sq = 0;
    int zeros = 0;
    
    for (int i = 0; i < n; i++) {
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
        sum += arr[i];
        sum_sq += arr[i] * arr[i];
        if (fabsf(arr[i]) < 1e-10f) zeros++;
    }
    
    float mean = sum / n;
    float var = sum_sq / n - mean * mean;
    float std = sqrtf(var > 0 ? var : 0);
    
    printf("  %-20s: min=%+.4f max=%+.4f mean=%+.6f std=%.6f zeros=%d/%d\n",
           name, min, max, mean, std, zeros, n);
}

int cmd_diagnose(int argc, char **argv) {
    printf("=== CNN Training Diagnostics ===\n\n");
    
    const char *data_path = "out/data/run_3h_master.dat";
    const char *weights_path = "out/models/run_3h_current.bin";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_path = argv[++i];
        else if (strcmp(argv[i], "--weights") == 0 && i+1 < argc) weights_path = argv[++i];
    }
    
    // 1. Load data
    printf("[1] Loading data from %s...\n", data_path);
    int count = dataset_get_count(data_path);
    if (count <= 0) {
        printf("ERROR: Cannot load data\n");
        return 1;
    }
    printf("    Loaded %d samples\n\n", count);
    
    TrainingSample *samples = malloc(count * sizeof(TrainingSample));
    dataset_load(data_path, samples, count);
    
    // 2. Check target policy distribution
    printf("[2] Target Policy Analysis (first 10 samples):\n");
    for (int s = 0; s < 10 && s < count; s++) {
        TrainingSample *sample = &samples[s];
        
        // Count non-zero entries
        int nz = 0;
        float max_p = 0;
        int max_idx = -1;
        float entropy = 0;
        
        for (int j = 0; j < 512; j++) {
            float p = sample->target_policy[j];
            if (p > 0.01f) nz++;
            if (p > max_p) { max_p = p; max_idx = j; }
            if (p > 1e-6f) entropy -= p * logf(p);
        }
        
        printf("    Sample %2d: player=%s nz=%2d max_p=%.3f max_idx=%3d entropy=%.2f value=%.2f\n",
               s, sample->state.current_player == 0 ? "W" : "B",
               nz, max_p, max_idx, entropy, sample->target_value);
    }
    printf("\n");
    
    // 3. Load weights and run forward pass
    printf("[3] Loading weights from %s...\n", weights_path);
    zobrist_init();
    init_move_tables();
    
    CNNWeights weights;
    cnn_init(&weights);
    if (cnn_load_weights(&weights, weights_path) != 0) {
        printf("    Using fresh random weights\n");
    } else {
        printf("    Loaded successfully\n");
    }
    printf("\n");
    
    // 4. Check weight statistics
    printf("[4] Weight Statistics:\n");
    print_stats("conv1_w", weights.conv1_w, 64 * CNN_INPUT_CHANNELS * 9);
    print_stats("conv4_w", weights.conv4_w, 64 * 64 * 9);
    print_stats("policy_w", weights.policy_w, 512 * 4097);
    print_stats("policy_b", weights.policy_b, 512);
    print_stats("bn1_gamma", weights.bn1_gamma, 64);
    print_stats("bn1_mean (running)", weights.bn1_mean, 64);
    print_stats("bn1_var (running)", weights.bn1_var, 64);
    printf("\n");
    
    // 5. Run forward pass and check activations
    printf("[5] Forward Pass Analysis (sample 0):\n");
    TrainingSample *s = &samples[0];
    
    float input[CNN_INPUT_CHANNELS * 64];
    float player;
    cnn_encode_sample(s, input, &player);
    
    // Check input encoding
    int input_nz = 0;
    for (int i = 0; i < CNN_INPUT_CHANNELS * 64; i++) {
        if (input[i] > 0.5f) input_nz++;
    }
    printf("    Input: %d non-zero (pieces on board)\n", input_nz);
    
    // Forward pass
    CNNOutput out;
    cnn_forward_sample(&weights, s, &out);
    
    // Check policy output
    float policy_min = out.policy[0], policy_max = out.policy[0];
    float policy_sum = 0;
    float policy_entropy = 0;
    int policy_max_idx = 0;
    
    for (int j = 0; j < 512; j++) {
        if (out.policy[j] < policy_min) policy_min = out.policy[j];
        if (out.policy[j] > policy_max) { policy_max = out.policy[j]; policy_max_idx = j; }
        policy_sum += out.policy[j];
        if (out.policy[j] > 1e-8f) policy_entropy -= out.policy[j] * logf(out.policy[j]);
    }
    
    printf("    Policy output: min=%.6f max=%.6f sum=%.4f entropy=%.2f max_idx=%d\n",
           policy_min, policy_max, policy_sum, policy_entropy, policy_max_idx);
    printf("    Value output: %.4f\n", out.value);
    
    // Compare with target
    int target_max_idx = 0;
    float target_max = 0;
    for (int j = 0; j < 512; j++) {
        if (s->target_policy[j] > target_max) {
            target_max = s->target_policy[j];
            target_max_idx = j;
        }
    }
    printf("    Target max: idx=%d prob=%.3f | Predicted at target idx: %.6f\n",
           target_max_idx, target_max, out.policy[target_max_idx]);
    printf("\n");
    
    // 6. Check if network output is uniform (entropy close to log(512))
    printf("[6] Uniformity Check:\n");
    float max_entropy = logf(512.0f);  // ~6.24
    printf("    Max possible entropy: %.2f\n", max_entropy);
    printf("    Actual entropy: %.2f (%.1f%% of max)\n", 
           policy_entropy, 100.0f * policy_entropy / max_entropy);
    
    if (policy_entropy > 0.95f * max_entropy) {
        printf("    WARNING: Network output is nearly UNIFORM! Softmax is saturating.\n");
        printf("    This suggests vanishing gradients or all logits are similar.\n");
    }
    
    // 7. Check cross-entropy loss for this sample
    float ce_loss = 0;
    for (int j = 0; j < 512; j++) {
        ce_loss -= s->target_policy[j] * logf(out.policy[j] + 1e-10f);
    }
    printf("\n[7] Sample Loss:\n");
    printf("    Cross-entropy: %.4f (random would be ~%.2f)\n", ce_loss, max_entropy);
    
    // 8. SINGLE BATCH OVERFITTING TEST (key sanity check!)
    printf("\n[8] Single Batch Overfitting Test (100 steps on 32 samples, LR=0.1):\n");
    printf("    If the network CAN'T overfit a tiny batch, something is fundamentally broken.\n\n");
    
    int batch_size = (count < 32) ? count : 32;
    float lr = 0.1f;  // Very high LR for overfitting test
    
    // Reinitialize weights for clean test
    CNNWeights test_weights;
    cnn_init(&test_weights);
    
    for (int step = 0; step < 100; step++) {
        float p_loss = 0, v_loss = 0;
        
        // Train on the same batch repeatedly
        for (int b = 0; b < batch_size; b++) {
            float p, v;
            cnn_train_step(&test_weights, &samples[b], 1, lr, lr * 0.02f, 0.0f, 0.0f, &p, &v);  // policy_lr=lr, value_lr=lr*0.02
            p_loss += p;
            v_loss += v;
        }
        p_loss /= batch_size;
        v_loss /= batch_size;
        
        if (step % 20 == 0 || step == 99) {
            // Check accuracy on this batch
            int correct = 0;
            for (int b = 0; b < batch_size; b++) {
                CNNOutput test_out;
                cnn_forward_sample(&test_weights, &samples[b], &test_out);
                
                int pred_idx = 0;
                float pred_max = test_out.policy[0];
                for (int j = 1; j < 512; j++) {
                    if (test_out.policy[j] > pred_max) {
                        pred_max = test_out.policy[j];
                        pred_idx = j;
                    }
                }
                
                int tgt_idx = 0;
                float tgt_max = samples[b].target_policy[0];
                for (int j = 1; j < 512; j++) {
                    if (samples[b].target_policy[j] > tgt_max) {
                        tgt_max = samples[b].target_policy[j];
                        tgt_idx = j;
                    }
                }
                
                if (pred_idx == tgt_idx) correct++;
            }
            
            printf("    Step %3d: p_loss=%.4f v_loss=%.4f acc=%d/%d (%.1f%%)\n", 
                   step, p_loss, v_loss, correct, batch_size, 100.0f * correct / batch_size);
        }
    }
    
    // 9. BatchNorm Running Stats Analysis
    printf("\n[9] BatchNorm Running Stats Analysis:\n");
    printf("    Comparing running stats (used in inference) vs batch stats (used in training)\n\n");
    
    // Compute batch stats from first 64 samples
    float input_batch[64][CNN_INPUT_CHANNELS * 64];
    float player_batch[64];
    int bn_batch = (count < 64) ? count : 64;
    
    for (int i = 0; i < bn_batch; i++) {
        cnn_encode_sample(&samples[i], input_batch[i], &player_batch[i]);
    }
    
    printf("    BN Layer 1:\n");
    printf("      Running mean range: [%.4f, %.4f]\n", 
           weights.bn1_mean[0], weights.bn1_mean[63]);
    printf("      Running var range:  [%.4f, %.4f]\n", 
           weights.bn1_var[0], weights.bn1_var[63]);
    
    // Check if running vars are suspiciously small or large
    float min_var = weights.bn1_var[0], max_var = weights.bn1_var[0];
    for (int i = 1; i < 64; i++) {
        if (weights.bn1_var[i] < min_var) min_var = weights.bn1_var[i];
        if (weights.bn1_var[i] > max_var) max_var = weights.bn1_var[i];
    }
    printf("      Variance range: [%.6f, %.6f]\n", min_var, max_var);
    
    if (min_var < 0.001f) {
        printf("      WARNING: Very small variance detected! This can cause exploding gradients.\n");
    }
    if (max_var > 10.0f) {
        printf("      WARNING: Very large variance detected! Activations may be unstable.\n");
    }
    
    // 10. Check gradients after overfitting test
    printf("\n[10] Gradient Magnitudes (after overfitting test):\n");
    print_stats("d_policy_w", test_weights.d_policy_w, 512 * 4097);
    print_stats("d_policy_b", test_weights.d_policy_b, 512);
    print_stats("d_conv4_w", test_weights.d_conv4_w, 64 * 64 * 9);
    print_stats("d_conv1_w", test_weights.d_conv1_w, 64 * CNN_INPUT_CHANNELS * 9);
    
    cnn_free(&test_weights);
    
    printf("\n=== Diagnostics Complete ===\n");
    
    free(samples);
    cnn_free(&weights);
    return 0;
}
