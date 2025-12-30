#ifndef TRAINING_PIPELINE_H
#define TRAINING_PIPELINE_H

#include "cnn.h"

typedef struct {
    int epochs;
    int batch_size;
    float learning_rate;
    float l2_decay;
    float momentum;
    int patience;
    const char *data_path;
    const char *model_path; // Path to save best model (or NULL)
    const char *backup_path; // Path to save backup model (or NULL)
    int num_threads;
    
    // Callbacks
    void (*on_init)(int total_samples, int validation_samples);
    void (*on_epoch_start)(int epoch, int total_epochs, float lr);
    void (*on_batch_log)(int batch, int total_batches, float p_loss, float v_loss, int samples_processed);
    void (*on_epoch_end)(int epoch, float train_p_loss, float train_v_loss, 
                         float val_p_loss, float val_v_loss, float val_acc, 
                         int improved, const char *saved_path);
    void (*on_complete)(float best_loss, int best_epoch);
} TrainingPipelineConfig;

/**
 * Runs the training loop.
 * Loads data, splits validation, runs epochs with SGD and validation.
 */
void training_run(CNNWeights *weights, const TrainingPipelineConfig *cfg);

#endif // TRAINING_PIPELINE_H
