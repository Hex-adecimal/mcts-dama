/**
 * dataset_analysis.c - Dataset statistics implementation
 */

#include "dama/training/dataset_analysis.h"
#include "dama/training/dataset.h"
#include "dama/engine/game.h"
#include "dama/neural/cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper for popcount
static inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

// Comparator for duplicate detection
static int compare_u64(const void *a, const void *b) {
    uint64_t ua = *(const uint64_t*)a;
    uint64_t ub = *(const uint64_t*)b;
    return (ua > ub) - (ua < ub);
}

int dataset_analyze(const char *path, DatasetStats *stats) {
    memset(stats, 0, sizeof(DatasetStats));
    
    int count = dataset_get_count(path);
    if (count <= 0) return 1;
    
    stats->count = count;
    stats->file_size_mb = (float)(count * sizeof(TrainingSample)) / (1024*1024);
    
    TrainingSample *samples = malloc(count * sizeof(TrainingSample));
    if (!samples) return 2; // Out of memory
    
    dataset_load(path, samples, count);
    
    // Init min/max
    stats->val_min = 1.0f;
    stats->val_max = -1.0f;
    stats->min_pieces = 24;
    stats->max_pieces = 0;
    stats->max_entropy = logf(512); // Theoretical max for policy
    
    float val_sum = 0;
    float total_moves = 0;
    float total_entropy = 0;
    float total_max_prob = 0;
    int total_pieces = 0;
    int total_ladies = 0;
    int total_pawns = 0;
    
    // Duplicate checking prep
    uint64_t *hashes = NULL;
    int check_duplicates = (count <= 5000000); // Increased limit (safe for modern RAM)
    if (check_duplicates) {
        hashes = malloc(count * sizeof(uint64_t));
        if (!hashes) check_duplicates = 0;
    }
    stats->duplicates_checked = check_duplicates;
    
    for (int i = 0; i < count; i++) {
        TrainingSample *s = &samples[i];
        float v = s->target_value;
        
        // Value stats
        val_sum += v;
        if (v < stats->val_min) stats->val_min = v;
        if (v > stats->val_max) stats->val_max = v;
        
        if (v > 0.1f) stats->wins++;
        else if (v < -0.1f) stats->losses++;
        else stats->draws++;
        
        // Policy stats
        int moves = 0;
        float max_p = 0; 
        float entropy = 0;
        
        for (int j = 0; j < CNN_POLICY_SIZE; j++) {
            float p = s->target_policy[j];
            if (p > 0.01f) moves++;
            if (p > max_p) max_p = p;
            if (p > 1e-6f) entropy -= p * logf(p);
        }
        total_moves += moves;
        total_max_prob += max_p;
        total_entropy += entropy;
        if (max_p > 0.5f) stats->sharp_policies++;
        
        // Board occupancy
        GameState *gs = &s->state;
        int w_pawns = popcount64(gs->piece[WHITE][PAWN]);
        int w_ladies = popcount64(gs->piece[WHITE][LADY]);
        int b_pawns = popcount64(gs->piece[BLACK][PAWN]);
        int b_ladies = popcount64(gs->piece[BLACK][LADY]);
        
        int pieces = w_pawns + w_ladies + b_pawns + b_ladies;
        int ladies = w_ladies + b_ladies;
        int pawns = w_pawns + b_pawns;
        
        total_pieces += pieces;
        total_ladies += ladies;
        total_pawns += pawns;
        
        if (pieces < stats->min_pieces) stats->min_pieces = pieces;
        if (pieces > stats->max_pieces) stats->max_pieces = pieces;
        
        // Histogram
        if (pieces >= 0 && pieces <= 24) stats->piece_histogram[pieces]++;
        
        // Phase
        if (pieces >= 20) stats->phase_opening++;
        else if (pieces >= 10) stats->phase_midgame++;
        else stats->phase_endgame++;
        
        // Hashes
        if (check_duplicates) hashes[i] = gs->hash;
    }
    
    // Averages
    stats->val_mean = val_sum / count;
    stats->avg_moves = total_moves / count;
    stats->avg_entropy = total_entropy / count;
    stats->entropy_ratio = stats->avg_entropy / stats->max_entropy;
    stats->avg_max_prob = total_max_prob / count;
    stats->sharp_ratio_pct = (float)stats->sharp_policies / count * 100.0f;
    
    stats->avg_pieces = (float)total_pieces / count;
    stats->avg_pawns = (float)total_pawns / count;
    stats->avg_ladies = (float)total_ladies / count;
    stats->lady_ratio_pct = (float)total_ladies / (total_pieces > 0 ? total_pieces : 1) * 100.0f;
    
    // Histogram buckets
    for (int i = 1; i <= 24; i++) {
        stats->buckets[(i-1) / 4] += stats->piece_histogram[i];
    }
    
    // Duplicates
    if (check_duplicates && hashes) {
        qsort(hashes, count, sizeof(uint64_t), compare_u64);
        for (int i = 1; i < count; i++) {
            if (hashes[i] == hashes[i-1]) stats->duplicates++;
        }
        free(hashes);
        stats->duplicate_ratio_pct = (float)stats->duplicates / count * 100.0f;
    }
    
    free(samples);
    return 0;
}
