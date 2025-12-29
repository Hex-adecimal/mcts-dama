/**
 * training_pipeline.c - High level training loop implementation
 */

#include "training_pipeline.h"
#include "dataset.h"
#include "core/rng.h"
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

// Validation loop
static void run_validation(CNNWeights *w, const TrainingSample *samples, int count, float *out_loss, float *out_acc) {
    double total_loss = 0;
    int correct_moves = 0;
    
    #pragma omp parallel for reduction(+:total_loss, correct_moves)
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
                   
        // Note: cnn_forward signature in cnn.h uses GameState*.
        // Need to check specific signature if it accepts pointer or what.
        
        // Loss
        double p_loss = 0;
        for (int j = 0; j < CNN_POLICY_SIZE; j++) {
            p_loss -= samples[i].target_policy[j] * logf(policy[j] + 1e-10f);
        }
        double v_loss = (value - samples[i].target_value) * (value - samples[i].target_value);
        total_loss += (p_loss + v_loss);
        
        // Accuracy (top 1 move)
        int best_pred = 0;
        float max_p = -1;
        for (int j = 0; j < CNN_POLICY_SIZE; j++) {
            if (policy[j] > max_p) { max_p = policy[j]; best_pred = j; }
        }
        
        // Is best_pred in target? target is probability distribution.
        if (samples[i].target_policy[best_pred] > 0.0f) {
            // Check if it was the "best" in target too
            float max_t = -1;
            int best_t = 0;
            for(int j=0; j<CNN_POLICY_SIZE; j++) {
                if(samples[i].target_policy[j] > max_t) { max_t=samples[i].target_policy[j]; best_t=j; }
            }
            if(best_pred == best_t) correct_moves++;
        }
    }
    
    *out_loss = (float)(total_loss / count);
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
    
    // Setup
    float lr = cfg->learning_rate;
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
        if (cfg->on_epoch_start) cfg->on_epoch_start(epoch, cfg->epochs, lr);
        
        // Shuffle train data
        shuffle_samples(train_data, n_train, &rng);
        
        // Training Batches
        int num_batches = (n_train + cfg->batch_size - 1) / cfg->batch_size;
        float epoch_loss = 0;
        
        for (int b = 0; b < num_batches; b++) {
            int start = b * cfg->batch_size;
            int end = start + cfg->batch_size;
            if (end > n_train) end = n_train;
            int size = end - start;
            
            float p_loss, v_loss;
            float batch_loss = cnn_train_step(weights, &train_data[start], size, lr, 0.0f, cfg->l2_decay, &p_loss, &v_loss);
            
            epoch_loss += batch_loss * size;
            
            if (cfg->on_batch_log && b % 10 == 0) {
                // Approximate samples processed
                cfg->on_batch_log(b, num_batches, batch_loss, start + size);
            }
        }
        epoch_loss /= n_train;
        
        // Validation
        float val_loss, val_acc;
        run_validation(weights, val_data, n_val, &val_loss, &val_acc);
        
        // Improvement check
        int improved = 0;
        const char *saved_path = NULL;
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
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
            // Decay LR if stuck
            if (patience_counter > cfg->patience / 2) {
                lr *= 0.8f; 
                // Don't let LR go too low?
                if (lr < 1e-6f) lr = 1e-6f;
            }
        }
        
        if (cfg->on_epoch_end) 
            cfg->on_epoch_end(epoch, epoch_loss, val_loss, val_acc, improved, saved_path);
            
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
