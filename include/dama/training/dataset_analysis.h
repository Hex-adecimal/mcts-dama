/**
 * dataset_analysis.h - Dataset statistics and analysis logic
 */

#ifndef DATASET_ANALYSIS_H
#define DATASET_ANALYSIS_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int count;
    float file_size_mb;
    
    // Value stats
    int wins;
    int losses; 
    int draws;
    float val_mean;
    float val_min;
    float val_max;
    
    // Policy stats
    float avg_moves;
    float avg_entropy;
    float max_entropy;
    float entropy_ratio;
    float avg_max_prob;
    int sharp_policies; // > 50%
    float sharp_ratio_pct;
    
    // Occupancy
    float avg_pieces;
    float avg_pawns;
    float avg_ladies;
    int min_pieces;
    int max_pieces;
    float lady_ratio_pct;
    
    // Phase
    int phase_opening;
    int phase_midgame;
    int phase_endgame;
    
    // Histograms
    int piece_histogram[25];  // 1-24
    int buckets[6];           // Grouped histogram: 1-4, 5-8...
    
    // Duplicates
    int duplicates;
    int duplicates_checked; // boolean
    float duplicate_ratio_pct;
} DatasetStats;

/**
 * Analyze a dataset file and populate the stats structure.
 * Returns 0 on success, non-zero on error.
 */
int dataset_analyze(const char *path, DatasetStats *stats);

#endif // DATASET_ANALYSIS_H
