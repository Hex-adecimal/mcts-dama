#ifndef DATASET_H
#define DATASET_H

#include "../core/game.h"
#include <stddef.h>
#include <stdint.h>

// =============================================================================
// TRAINING SAMPLE
// =============================================================================

#define CNN_POLICY_SIZE 512  // Must match cnn.h
#define CNN_HISTORY_T   3    // Must match cnn.h

typedef struct {
    GameState state;                        // Current state (T=0)
    GameState history[CNN_HISTORY_T - 1];   // Previous states (T=-1, T=-2)
    float target_policy[CNN_POLICY_SIZE];
    float target_value;
} TrainingSample;

// =============================================================================
// DATASET FILE FORMAT
// =============================================================================
// Binary file structure:
//   [Header: 12 bytes]
//     - Magic: "DAMA" (4 bytes)
//     - Version: uint32_t (4 bytes)
//     - Sample count: uint32_t (4 bytes)
//   [Samples: N × sizeof(TrainingSample)]

#define DATASET_MAGIC   "DAMA"
#define DATASET_VERSION 1

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t num_samples;
} DatasetHeader;

// =============================================================================
// DATASET API
// =============================================================================

/**
 * Save training samples to binary file.
 * @param filename Path to output file.
 * @param samples Array of training samples.
 * @param count Number of samples.
 * @return 0 on success, -1 on error.
 */
int dataset_save(const char *filename, const TrainingSample *samples, size_t count);

/**
 * Append training samples to an existing dataset file.
 * Updates the header sample count automatically.
 * Creates the file if it doesn't exist.
 */
int dataset_save_append(const char *filename, const TrainingSample *samples, size_t count);

/**
 * Load training samples from binary file.
 * @param filename Path to input file.
 * @param samples Output array (caller allocates).
 * @param max_samples Maximum samples to load.
 * @return Number of samples loaded, or -1 on error.
 */
int dataset_load(const char *filename, TrainingSample *samples, size_t max_samples);

/**
 * Get sample count from dataset file without loading.
 * @param filename Path to dataset file.
 * @return Number of samples, or -1 on error.
 */
int dataset_get_count(const char *filename);

/**
 * Load dataset with automatic memory allocation.
 * @param filename Path to dataset file.
 * @param out_count Output: number of samples loaded.
 * @return Allocated array of samples (caller must free), or NULL on error.
 */
TrainingSample* dataset_load_alloc(const char *filename, int *out_count);

/**
 * Shuffle samples in-place (Fisher-Yates).
 */
void dataset_shuffle(TrainingSample *samples, size_t count);

/**
 * Split dataset into train/validation sets.
 * @param samples Full dataset (will be shuffled).
 * @param count Total sample count.
 * @param train_ratio Fraction for training (e.g., 0.8).
 * @param train_out Output: training samples.
 * @param val_out Output: validation samples.
 * @param train_count Output: number of training samples.
 * @param val_count Output: number of validation samples.
 */
void dataset_split(TrainingSample *samples, size_t count, float train_ratio,
                   TrainingSample **train_out, TrainingSample **val_out,
                   size_t *train_count, size_t *val_count);

#endif // DATASET_H
