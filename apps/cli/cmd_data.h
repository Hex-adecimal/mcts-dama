/**
 * cmd_data.h - Data loading and balanced sampling utilities
 * Header-only implementation for simplicity
 */

#ifndef CMD_DATA_H
#define CMD_DATA_H

#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// DATA STRUCTURES
// =============================================================================

typedef struct {
    TrainingSample *train_data;
    TrainingSample *val_data;
    size_t train_count;
    size_t val_count;
} DatasetSplit;

typedef struct {
    int *w_idxs;
    int *l_idxs;
    int *d_idxs;
    int w_cnt;
    int l_cnt;
    int d_cnt;
} BalancedIndex;

// =============================================================================
// DATA LOADING (inline implementations)
// =============================================================================

static inline TrainingSample* load_dataset_file(const char *path, int *out_count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("ERROR: No data found at '%s'\n", path);
        *out_count = 0;
        return NULL;
    }
    fclose(f);
    
    int count = dataset_get_count(path);
    if (count <= 0) {
        printf("ERROR: Failed to get data count from '%s'\n", path);
        *out_count = 0;
        return NULL;
    }
    
    TrainingSample *data = malloc(count * sizeof(TrainingSample));
    if (!data) {
        printf("ERROR: Failed to allocate memory\n");
        *out_count = 0;
        return NULL;
    }
    
    *out_count = dataset_load(path, data, count);
    return data;
}

static inline DatasetSplit split_dataset(TrainingSample *all_data, int total_count, 
                                          const char *val_file, float train_ratio) {
    DatasetSplit split = {0};
    
    if (val_file) {
        int v_cnt = 0;
        split.val_data = load_dataset_file(val_file, &v_cnt);
        split.val_count = v_cnt;
        split.train_data = all_data;
        split.train_count = total_count;
    } else {
        printf("Auto-splitting (%.0f%% Train / %.0f%% Val)...\n", 
               train_ratio * 100, (1 - train_ratio) * 100);
        dataset_split(all_data, total_count, train_ratio, 
                      &split.train_data, &split.val_data, 
                      &split.train_count, &split.val_count);
    }
    
    printf("  Train: %'zu samples | Val: %'zu samples\n", split.train_count, split.val_count);
    return split;
}

// =============================================================================
// BALANCED SAMPLING (inline implementations)
// =============================================================================

static inline BalancedIndex build_balanced_index(TrainingSample *data, size_t count) {
    BalancedIndex idx = {0};
    
    idx.w_idxs = malloc(count * sizeof(int));
    idx.l_idxs = malloc(count * sizeof(int));
    idx.d_idxs = malloc(count * sizeof(int));
    
    for (int i = 0; i < (int)count; i++) {
        if (data[i].target_value > 0.1f) idx.w_idxs[idx.w_cnt++] = i;
        else if (data[i].target_value < -0.1f) idx.l_idxs[idx.l_cnt++] = i;
        else idx.d_idxs[idx.d_cnt++] = i;
    }
    
    return idx;
}

static inline void free_balanced_index(BalancedIndex *idx) {
    free(idx->w_idxs);
    free(idx->l_idxs);
    free(idx->d_idxs);
}

static inline int sample_balanced_index(BalancedIndex *idx, int total_count) {
    float draw_ratio = (float)idx->d_cnt / total_count;
    float draw_prob = (draw_ratio < 0.10f) ? draw_ratio : 0.10f;
    float wl_prob = (1.0f - draw_prob) / 2.0f;
    
    float r = (float)rand() / RAND_MAX;
    
    if (r < wl_prob && idx->w_cnt > 0)           return idx->w_idxs[rand() % idx->w_cnt];
    else if (r < 2*wl_prob && idx->l_cnt > 0)    return idx->l_idxs[rand() % idx->l_cnt];
    else if (idx->d_cnt > 0)                     return idx->d_idxs[rand() % idx->d_cnt];
    else                                         return rand() % total_count;
}

static inline void fill_balanced_batch(TrainingSample *batch, int batch_size,
                                        TrainingSample *data, BalancedIndex *idx, int total_count) {
    for (int k = 0; k < batch_size; k++) {
        batch[k] = data[sample_balanced_index(idx, total_count)];
    }
}

#endif // CMD_DATA_H
