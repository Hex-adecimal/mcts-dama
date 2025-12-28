/**
 * cnn_trainer.c - CNN Training Tool
 * 
 * Loads pre-generated dataset and trains the CNN.
 * Run data generation first: make datagen && ./bin/datagen
 * Then train: make cnn_trainer && ./bin/cnn_trainer
 */

#include "game.h"
#include "movegen.h"
#include "cnn.h"
#include "dataset.h"
#include "params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <locale.h>

// =============================================================================
// CONFIGURATION & HYPERPARAMETERS
// =============================================================================

#define TRAIN_FILE      "data/training/train.bin"
#define VAL_FILE        "data/training/val.bin"
#define WEIGHTS_FILE    "models/cnn_weights.bin"

#define NUM_EPOCHS          10              
#define BATCH_SIZE          4096            
#define LEARNING_RATE       0.f          
#define MOMENTUM            0.9f
#define L1_DECAY            0.0f
#define L2_DECAY            0.00001f        
#define MAX_PATIENCE        3               

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv) {
    setlocale(LC_NUMERIC, "");  // Enable thousand separators
    printf("=== CNN Trainer ===\n\n");
    
    char *train_file = (argc > 1) ? argv[1] : TRAIN_FILE;
    char *val_file = (argc > 2) ? argv[2] : NULL; // Optional
    
    srand((unsigned)time(NULL));
    zobrist_init();
    init_move_tables();
    
    // Check for dataset
    FILE *f = fopen(train_file, "rb");
    if (!f) {
        printf("ERROR: No training data found at '%s'\n", train_file);
        return 1;
    }
    fclose(f);
    
    printf("Loading dataset from: %s\n", train_file);
    
    // Get sample counts
    int total_count = dataset_get_count(train_file);
    if (total_count <= 0) {
        printf("ERROR: Failed to get training data count.\n");
        return 1;
    }
    
    // Allocate memory
    TrainingSample *all_data = malloc(total_count * sizeof(TrainingSample));
    if (!all_data) {
        printf("ERROR: Failed to allocate memory.\n");
        return 1;
    }
    
    // Load dataset
    total_count = dataset_load(train_file, all_data, total_count);
    
    // Split Train/Val
    TrainingSample *train_data = NULL;
    TrainingSample *val_data = NULL;
    size_t train_count = 0;
    size_t val_count = 0;
    
    if (val_file) {
        // Load separate validation file
        int v_cnt = dataset_get_count(val_file);
        if (v_cnt > 0) {
            val_data = malloc(v_cnt * sizeof(TrainingSample));
            dataset_load(val_file, val_data, v_cnt);
            val_count = v_cnt;
        }
        train_data = all_data;
        train_count = total_count;
    } else {
        // Auto-split 90/10
        printf("Auto-splitting dataset (90%% Train / 10%% Val)...\n");
        dataset_split(all_data, total_count, 0.90f, 
                      &train_data, &val_data, 
                      &train_count, &val_count);
    }
    
    printf("  Train: %'zu samples\n", train_count);
    printf("  Val:   %'zu samples\n", val_count);

    // Initialize CNN
    CNNWeights weights;
    cnn_init(&weights);
    
    // Try to load existing weights
    if (cnn_load_weights(&weights, WEIGHTS_FILE) == 0) {
        printf("\n✓ Loaded existing weights from %s\n", WEIGHTS_FILE);
    } else {
        printf("\n⚠ No existing weights, starting fresh\n");
    }
    
    float learning_rate = LEARNING_RATE;

    // --- INDEXING FOR BALANCED SAMPLING (TRAIN) ---
    printf("\nStructuring balanced batches (40%% W / 40%% L / 20%% D)...\n");
    int *w_idxs = malloc(train_count * sizeof(int));
    int *l_idxs = malloc(train_count * sizeof(int));
    int *d_idxs = malloc(train_count * sizeof(int));
    int w_cnt = 0, l_cnt = 0, d_cnt = 0;
    
    for (int i = 0; i < (int)train_count; i++) {
        if (train_data[i].target_value > 0.1f) w_idxs[w_cnt++] = i;
        else if (train_data[i].target_value < -0.1f) l_idxs[l_cnt++] = i;
        else                                     d_idxs[d_cnt++] = i;
    }
    printf("  Natural Dist: Wins %'d (%.1f%%) | Losses %'d (%.1f%%) | Draws %'d (%.1f%%)\n", 
           w_cnt, (float)w_cnt/train_count*100, 
           l_cnt, (float)l_cnt/train_count*100, 
           d_cnt, (float)d_cnt/train_count*100);

    // Calc Target Distribution
    float nat_draw_pct = (float)d_cnt / train_count * 100;
    float target_draw_pct = (nat_draw_pct < 10.0f) ? nat_draw_pct : 10.0f; // Cap draws at 10%
    float target_wl_pct = (100.0f - target_draw_pct) / 2.0f;

    printf("  Target Dist : Wins %.1f%% | Losses %.1f%% | Draws %.1f%%\n", 
           target_wl_pct, target_wl_pct, target_draw_pct);
           
    // Explain Strategy
    if (nat_draw_pct > 10.0f) {
        printf("  -> Strategy : DOWNSAMPLING Draws (%.1fx less freq) | OVERSAMPLING W/L (%.1fx more freq)\n",
               nat_draw_pct / target_draw_pct, 
               target_wl_pct / ((float)w_cnt/train_count*100));
    } else {
        printf("  -> Strategy : Natural Distribution (No Rebalancing needed)\n");
    }

    // --- INDEXING FOR BALANCED SAMPLING (VAL) ---
    int *val_w_idxs = malloc(val_count * sizeof(int));
    int *val_l_idxs = malloc(val_count * sizeof(int));
    int *val_d_idxs = malloc(val_count * sizeof(int));
    int val_w_cnt = 0, val_l_cnt = 0, val_d_cnt = 0;
    
    if (val_data && val_count > 0) {
        for (int i = 0; i < (int)val_count; i++) {
            if (val_data[i].target_value > 0.1f) val_w_idxs[val_w_cnt++] = i;
            else if (val_data[i].target_value < -0.1f) val_l_idxs[val_l_cnt++] = i;
            else                                     val_d_idxs[val_d_cnt++] = i;
        }
        printf("  Val   Dist: Wins %'d | Losses %'d | Draws %'d\n", val_w_cnt, val_l_cnt, val_d_cnt);
    }

    // Prepare batch buffer
    TrainingSample *batch_buffer = malloc(BATCH_SIZE * sizeof(TrainingSample));

    // Print network architecture size
    int conv1_params = CNN_CONV1_CHANNELS * CNN_INPUT_CHANNELS * CNN_KERNEL_SIZE * CNN_KERNEL_SIZE + CNN_CONV1_CHANNELS;
    int conv2_params = CNN_CONV2_CHANNELS * CNN_CONV1_CHANNELS * CNN_KERNEL_SIZE * CNN_KERNEL_SIZE + CNN_CONV2_CHANNELS;
    int conv3_params = CNN_CONV3_CHANNELS * CNN_CONV2_CHANNELS * CNN_KERNEL_SIZE * CNN_KERNEL_SIZE + CNN_CONV3_CHANNELS;
    int conv4_params = CNN_CONV4_CHANNELS * CNN_CONV3_CHANNELS * CNN_KERNEL_SIZE * CNN_KERNEL_SIZE + CNN_CONV4_CHANNELS;
    int bn_params = 4 * (CNN_CONV1_CHANNELS + CNN_CONV2_CHANNELS + CNN_CONV3_CHANNELS + CNN_CONV4_CHANNELS);  // gamma, beta, mean, var
    int policy_params = CNN_POLICY_SIZE * CNN_FC_INPUT_SIZE + CNN_POLICY_SIZE;
    int value_params = CNN_VALUE_HIDDEN * CNN_FC_INPUT_SIZE + CNN_VALUE_HIDDEN + CNN_VALUE_HIDDEN + 1;
    int total_params = conv1_params + conv2_params + conv3_params + conv4_params + bn_params + policy_params + value_params;
    
    printf("\n=== Network Architecture (4-Layer CNN + BN) ===\n");
    printf("  Conv1: %d -> %d channels (%'d params)\n", CNN_INPUT_CHANNELS, CNN_CONV1_CHANNELS, conv1_params);
    printf("  Conv2: %d -> %d channels (%'d params)\n", CNN_CONV1_CHANNELS, CNN_CONV2_CHANNELS, conv2_params);
    printf("  Conv3: %d -> %d channels (%'d params)\n", CNN_CONV2_CHANNELS, CNN_CONV3_CHANNELS, conv3_params);
    printf("  Conv4: %d -> %d channels (%'d params)\n", CNN_CONV3_CHANNELS, CNN_CONV4_CHANNELS, conv4_params);
    printf("  BatchNorm: 4 layers (%'d params)\n", bn_params);
    printf("  Policy Head: %'d inputs -> %d outputs (%'d params)\n", CNN_FC_INPUT_SIZE, CNN_POLICY_SIZE, policy_params);
    printf("  Value Head: %'d inputs -> %d hidden -> 1 (%'d params)\n", CNN_FC_INPUT_SIZE, CNN_VALUE_HIDDEN, value_params);
    printf("  TOTAL PARAMETERS: %'d (%.2f MB)\n", total_params, (float)(total_params * 4) / (1024 * 1024));

    printf("\nTraining for %d epochs (LR: %.4f, Batch: %'d)\n\n", NUM_EPOCHS, learning_rate, BATCH_SIZE);
    printf("+-------+---------------------------+---------------------------+------------+\n");
    printf("| Epoch |        Train Loss         |         Val Loss          |   Status   |\n");
    printf("|       |  Total  | Policy  | Value |  Total  | Policy  | Value |            |\n");
    printf("+-------+---------+---------+-------+---------+---------+-------+------------+\n");
    
    // Track best loss
    float global_min_val_loss = 1e9f;
    int patience = 0;

    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        float epoch_policy_loss = 0.0f;
        float epoch_value_loss = 0.0f;
        float epoch_total_loss = 0.0f;
        
        // Number of batches per epoch (cover dataset roughly once)
        int num_batches = train_count / BATCH_SIZE;
        
        for (int b = 0; b < num_batches; b++) {
            // Fill batch with balanced sampling
            for (int k = 0; k < BATCH_SIZE; k++) {
                float r = (float)rand() / RAND_MAX;
                int src_idx;
                
                // Balanced sampling: 45% Win, 45% Loss, 10% Draw (or natural ratio if fewer draws)
                // Cap draw probability at actual ratio to avoid excessive duplication
                float draw_ratio = (float)d_cnt / (w_cnt + l_cnt + d_cnt);
                float draw_prob = (draw_ratio < 0.10f) ? draw_ratio : 0.10f;
                float wl_prob = (1.0f - draw_prob) / 2.0f;
                
                if (r < wl_prob && w_cnt > 0)           src_idx = w_idxs[rand() % w_cnt];
                else if (r < 2*wl_prob && l_cnt > 0)    src_idx = l_idxs[rand() % l_cnt];
                else if (d_cnt > 0)                     src_idx = d_idxs[rand() % d_cnt];
                else                                    src_idx = rand() % train_count;
                
                batch_buffer[k] = train_data[src_idx];
            }
            
        // Calculate LR for this batch (Linear Warmup or Plateau)
            float effective_lr = learning_rate;
            
            // Linear Warmup (First Epoch only)
            if (epoch == 1) {
                float warmup_pct = (float)(b + 1) / num_batches;
                effective_lr = learning_rate * warmup_pct;
            }

            float p_loss, v_loss;
            float loss = cnn_train_step(&weights, batch_buffer, BATCH_SIZE, effective_lr, L1_DECAY, L2_DECAY, &p_loss, &v_loss);
            
            epoch_total_loss += loss;
            epoch_policy_loss += p_loss;
            epoch_value_loss += v_loss;
            
            // Progress bar within epoch
            if ((b+1) % 10 == 0) {
                 int pct = (b+1) * 100 / num_batches;
                 int bar_len = 20;
                 int filled = pct * bar_len / 100;
                 printf("\r  [Epoch %2d] [", epoch);
                 for (int x = 0; x < bar_len; x++) printf(x < filled ? "#" : ".");
                 printf("] %3d%% | Loss: %.4f (P:%.3f V:%.3f) | LR: %.5f   ",
                        pct, loss, p_loss, v_loss, effective_lr);
                 fflush(stdout);
            }
        }
        
        // Average training losses
        if (num_batches > 0) {
            epoch_total_loss /= num_batches;
            epoch_policy_loss /= num_batches;
            epoch_value_loss /= num_batches;
        }
        
        // --- VALIDATION (Balanced Pass) ---
        float val_total = 0, val_p = 0, val_v = 0;
        int val_num_batches = val_count / BATCH_SIZE; // Use full val count but sample balanced
        if (val_num_batches < 10) val_num_batches = 10; // Ensure enough batches for stability

        if (val_count > 0) {
             for (int b = 0; b < val_num_batches; b++) {
                 // Fill VALIDATION batch with balanced sampling
                for (int k = 0; k < BATCH_SIZE; k++) {
                    float r = (float)rand() / RAND_MAX;
                    int src_idx;
                    // 40% W, 40% L, 20% D
                    if (r < 0.4f && val_w_cnt > 0)      src_idx = val_w_idxs[rand() % val_w_cnt];
                    else if (r < 0.8f && val_l_cnt > 0) src_idx = val_l_idxs[rand() % val_l_cnt];
                    else if (val_d_cnt > 0)             src_idx = val_d_idxs[rand() % val_d_cnt];
                    else                                src_idx = rand() % val_count;
                    
                    batch_buffer[k] = val_data[src_idx];
                }

                CNNOutput out;
                float batch_p = 0, batch_v = 0;
                
                // Parallelized validation forward pass
                #pragma omp parallel for reduction(+:batch_p, batch_v) private(out)
                for (int j = 0; j < BATCH_SIZE; j++) {
                    cnn_forward_sample(&weights, &batch_buffer[j], &out);
                    
                    // Policy Loss (with fmaxf for -ffast-math safety)
                    float local_p = 0;
                    for (int k = 0; k < CNN_POLICY_SIZE; k++) {
                        if (batch_buffer[j].target_policy[k] > 0) {
                            float p = fmaxf(out.policy[k], 1e-4f);
                            local_p -= batch_buffer[j].target_policy[k] * logf(p);
                        }
                    }
                    batch_p += local_p;
                    
                    // Value Loss
                    float diff = out.value - batch_buffer[j].target_value;
                    batch_v += diff * diff;
                }
                val_p += batch_p / BATCH_SIZE;
                val_v += batch_v / BATCH_SIZE;
            }
            
            val_total = (val_p + val_v) / val_num_batches;
            val_p /= val_num_batches;
            val_v /= val_num_batches;
        }

        // Save weights
        printf("\r                                                                                    \r"); // Clear line
        
        if (val_total < global_min_val_loss) {
            global_min_val_loss = val_total;
            cnn_save_weights(&weights, WEIGHTS_FILE);
            patience = 0;
            printf("|  %3d  | %7.4f | %7.4f | %5.3f | %7.4f | %7.4f | %5.3f |   *BEST*   |\n",
                   epoch, epoch_total_loss, epoch_policy_loss, epoch_value_loss, val_total, val_p, val_v);
        } else {
            patience++;
            printf("|  %3d  | %7.4f | %7.4f | %5.3f | %7.4f | %7.4f | %5.3f |  wait %d/%d |\n",
                   epoch, epoch_total_loss, epoch_policy_loss, epoch_value_loss, val_total, val_p, val_v, patience, MAX_PATIENCE);
        }
        
        // LR Decay on Plateau (ReduceLROnPlateau)
        if (patience >= MAX_PATIENCE) {
            float new_lr = learning_rate * 0.1f; // Factor 0.1
            
            if (new_lr < 1e-6f) {
                printf("\n!!! Converged: LR dropped below 1e-6. Stopping. !!!\n");
                break;
            }
            
            printf("\n*** Plateau Detected: No improvement for %d epochs. Decay LR: %.6f -> %.6f ***\n\n", 
                   MAX_PATIENCE, learning_rate, new_lr);
            
            learning_rate = new_lr;
            patience = 0; // Reset patience
        }
        
        // Checkpoint
        if (epoch % 10 == 0) {
            char path[64];
            sprintf(path, "models/cnn_checkpoint_%d.bin", epoch);
            cnn_save_weights(&weights, path);
        }
    }
    
    printf("+-------+---------+---------+-------+---------+---------+-------+------------+\n\n");
    printf("   TRAINING COMPLETE\n");
    printf("   Best validation loss: %.4f\n", global_min_val_loss);
    
    // Save final weights to separate file
    cnn_save_weights(&weights, "models/cnn_weights_final.bin");
    printf("   📁 Final weights: models/cnn_weights_final.bin\n");
    printf("   📁 Best weights:  %s\n", WEIGHTS_FILE);
    
    // Cleanup
    free(all_data);
    if (val_file && val_data) free(val_data); // Only free if allocated separately
    cnn_training_cleanup();  // Free thread-local convolution buffers
    cnn_free(&weights);
    
    return 0;
}
