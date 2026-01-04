/**
 * training_pipeline.c - High level training loop implementation
 */

#include "dama/training/training_pipeline.h"
#include "dama/training/dataset.h"
#include "dama/common/rng.h"
#include "dama/common/params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// HELPERS
// =============================================================================

static void shuffle_samples(TrainingSample *arr, int n, RNG *rng) {
    for (int i = n - 1; i > 0; i--) {
        int j = rng_u32(rng) % (i + 1);
        TrainingSample tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// Validation loop - returns separate policy and value loss
static void run_validation(CNNWeights *w, const TrainingSample *samples, int count, 
                           float *out_p_loss, float *out_v_loss, float *out_acc) {
    double total_p_loss = 0, total_v_loss = 0;
    int correct_moves = 0;
    
    #pragma omp parallel for reduction(+:total_p_loss, total_v_loss, correct_moves)
    for (int i = 0; i < count; i++) {
        float policy[CNN_POLICY_SIZE];
        float value;
        CNNOutput out;
        cnn_forward_with_history(w, &samples[i].state, 
                   &samples[i].history[0], 
                   &samples[i].history[1], 
                   &out);
        memcpy(policy, out.policy, sizeof(policy));
        value = out.value;
        
        // Policy loss (cross-entropy)
        double p_loss = 0;
        for (int j = 0; j < CNN_POLICY_SIZE; j++) {
            p_loss -= samples[i].target_policy[j] * logf(policy[j] + 1e-10f);
        }
        // Value loss (MSE)
        double v_loss = (value - samples[i].target_value) * (value - samples[i].target_value);
        total_p_loss += p_loss;
        total_v_loss += v_loss;
        
        // Accuracy (top 1 move)
        int best_pred = 0;
        float max_p = -1;
        for (int j = 0; j < CNN_POLICY_SIZE; j++) {
            if (policy[j] > max_p) { max_p = policy[j]; best_pred = j; }
        }
        
        // Check if prediction matches target
        if (samples[i].target_policy[best_pred] > 0.0f) {
            float max_t = -1;
            int best_t = 0;
            for(int j=0; j<CNN_POLICY_SIZE; j++) {
                if(samples[i].target_policy[j] > max_t) { max_t=samples[i].target_policy[j]; best_t=j; }
            }
            if(best_pred == best_t) correct_moves++;
        }
    }
    
    *out_p_loss = (float)(total_p_loss / count);
    *out_v_loss = (float)(total_v_loss / count);
    *out_acc = (float)correct_moves / count;
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void training_run(CNNWeights *weights, const TrainingPipelineConfig *cfg) {
    // 1. Load Data
    int count = dataset_get_count(cfg->data_path);
    if (count <= 0) {
        // Error or empty. 
        // We can't easily return error code from void, maybe add logging callback or return int.
        // Assuming caller checks file existence.
        if (cfg->on_init) cfg->on_init(0, 0);
        return;
    }
    
    TrainingSample *samples = malloc(count * sizeof(TrainingSample));
    if (!samples) return; // OOM
    
    dataset_load(cfg->data_path, samples, count);
    
    // Shuffle
    RNG rng;
    rng_seed(&rng, time(NULL));
    shuffle_samples(samples, count, &rng);
    
    // Split 90/10
    int n_val = count / 10;
    if (n_val < 100) n_val = (count > 100) ? 100 : count/5; // Ensure some validation
    int n_train = count - n_val;
    TrainingSample *train_data = samples;
    TrainingSample *val_data = samples + n_train;
    
    if (cfg->on_init) cfg->on_init(n_train, n_val);
    
    // Setup - separate LRs for policy and value heads
    // If cfg->learning_rate is set (>0), use it as base for Policy LR and scale Value LR
    float base_lr = (cfg->learning_rate > 1e-6f) ? cfg->learning_rate : CNN_POLICY_LR;
    float policy_lr = base_lr;
    float value_lr = base_lr * (CNN_VALUE_LR / CNN_POLICY_LR); // Maintain ratio defined in params.h
    float best_val_loss = 1e9f;
    int patience_counter = 0;
    int best_epoch = -1;
    
    // Threads
    int threads = (cfg->num_threads > 0) ? cfg->num_threads : 1;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    // Epochs
    for (int epoch = 1; epoch <= cfg->epochs; epoch++) {
        // Use geometric mean for display (represents backbone LR)
        float display_lr = sqrtf(policy_lr * value_lr);
        if (cfg->on_epoch_start) cfg->on_epoch_start(epoch, cfg->epochs, display_lr);
        
        // Shuffle train data
        shuffle_samples(train_data, n_train, &rng);
        
        // Training Batches
        int num_batches = (n_train + cfg->batch_size - 1) / cfg->batch_size;
        float epoch_p_loss = 0, epoch_v_loss = 0;
        
        for (int b = 0; b < num_batches; b++) {
            int start = b * cfg->batch_size;
            int end = start + cfg->batch_size;
            if (end > n_train) end = n_train;
            int size = end - start;
            
            // Linear LR Warmup for first epoch (applies to both LRs)
            float effective_policy_lr = policy_lr;
            float effective_value_lr = value_lr;
            if (epoch <= CNN_LR_WARMUP_EPOCHS) {
                float warmup_progress = (float)(b + 1) / (float)num_batches;
                effective_policy_lr = policy_lr * warmup_progress;
                effective_value_lr = value_lr * warmup_progress;
            }
            
            float p_loss, v_loss;
            cnn_train_step(weights, &train_data[start], size, effective_policy_lr, effective_value_lr, 0.0f, cfg->l2_decay, &p_loss, &v_loss);
            
            epoch_p_loss += p_loss * size;
            epoch_v_loss += v_loss * size;
            
            if (cfg->on_batch_log && b % 10 == 0) {
                cfg->on_batch_log(b, num_batches, p_loss, v_loss, start + size);
            }
        }
        epoch_p_loss /= n_train;
        epoch_v_loss /= n_train;
        
        // Validation
        float val_p_loss, val_v_loss, val_acc;
        run_validation(weights, val_data, n_val, &val_p_loss, &val_v_loss, &val_acc);
        float total_val_loss = val_p_loss + val_v_loss;
        
        // Improvement check (based on total validation loss)
        int improved = 0;
        const char *saved_path = NULL;
        if (total_val_loss < best_val_loss) {
            best_val_loss = total_val_loss;
            best_epoch = epoch;
            improved = 1;
            patience_counter = 0;
            
            if (cfg->model_path) {
                cnn_save_weights(weights, cfg->model_path);
                saved_path = cfg->model_path;
            }
            if (cfg->backup_path) {
                cnn_save_weights(weights, cfg->backup_path);
            }
        } else {
            patience_counter++;
            // Decay both LRs if stuck (using params.h constants)
            if (patience_counter >= CNN_LR_DECAY_PATIENCE) {
                policy_lr *= CNN_LR_DECAY_FACTOR; 
                value_lr *= CNN_LR_DECAY_FACTOR;
                if (policy_lr < 1e-6f) policy_lr = 1e-6f;
                if (value_lr < 1e-6f) value_lr = 1e-6f;
            }
        }
        
        if (cfg->on_epoch_end) 
            cfg->on_epoch_end(epoch, epoch_p_loss, epoch_v_loss, val_p_loss, val_v_loss, val_acc, improved, saved_path);
            
        // Early stopping
        if (patience_counter >= cfg->patience) {
            break;
        }
    }
    
    // Cleanup
    cnn_training_cleanup(); // Important for freeing thread-local buffers
    free(samples);
    
    if (cfg->on_complete) cfg->on_complete(best_val_loss, best_epoch);
}
