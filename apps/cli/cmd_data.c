/**
 * cmd_data.c - Data utilities and loading functions
 * 
 * Contains:
 * - Data loading and balanced sampling (used by cmd_train)
 * - inspect/merge commands (used by CLI)
 */

#include "dama/training/dataset.h"
#include "dama/engine/game.h"
#include "dama/common/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <locale.h>
#include <dirent.h>
#include "dama/training/dataset_analysis.h"
#include "dama/common/cli_view.h"

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
// DATA LOADING (wrapper for backward compat)
// =============================================================================

static inline TrainingSample* load_dataset_file(const char *path, int *out_count) {
    return dataset_load_alloc(path, out_count);
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
// BALANCED SAMPLING
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

// Helper functions (popcount, compare_u64) moved to dataset_analysis.c or removed as unused

// =============================================================================
// INSPECT - Show dataset statistics
// =============================================================================

static int data_inspect(const char *path) {
    DatasetStats stats;
    int res = dataset_analyze(path, &stats);
    
    if (res != 0) {
        printf("ERROR: Cannot read file or empty (code %d).\n", res);
        return 1;
    }
    
    // Map Logic Stats to View Stats
    DatasetStatsView view = {
        .path = path,
        .count = stats.count,
        .file_size_mb = stats.file_size_mb,
        .wins = stats.wins, .losses = stats.losses, .draws = stats.draws,
        .val_mean = stats.val_mean, .val_min = stats.val_min, .val_max = stats.val_max,
        
        .avg_moves = stats.avg_moves,
        .avg_entropy = stats.avg_entropy,
        .max_entropy = stats.max_entropy,
        .entropy_ratio_pct = stats.entropy_ratio * 100.0f,
        .avg_max_prob_pct = stats.avg_max_prob * 100.0f,
        .sharp_policies = stats.sharp_policies,
        .sharp_ratio_pct = stats.sharp_ratio_pct,
        
        .avg_pieces = stats.avg_pieces,
        .avg_pawns = stats.avg_pawns,
        .avg_ladies = stats.avg_ladies,
        .min_pieces = stats.min_pieces, .max_pieces = stats.max_pieces,
        .lady_ratio_pct = stats.lady_ratio_pct,
        
        .phase_opening = stats.phase_opening,
        .phase_midgame = stats.phase_midgame,
        .phase_endgame = stats.phase_endgame,
        
        .buckets = stats.buckets,
        
        .duplicates = stats.duplicates,
        .duplicates_checked = stats.duplicates_checked,
        .duplicate_ratio_pct = stats.duplicate_ratio_pct
    };
    
    cli_view_print_dataset_stats(&view);
    return 0;
}

// =============================================================================
// MERGE - Combine multiple dataset files
// =============================================================================

static int data_merge(int file_count, char **files, const char *output) {
    printf("=== Dataset Merger ===\n\n");
    printf("Merging %d files into: %s\n\n", file_count, output);
    
    int total = 0;
    for (int i = 0; i < file_count; i++) {
        int n = dataset_get_count(files[i]);
        if (n > 0) {
            printf("  [%d] %s: %'d samples\n", i+1, files[i], n);
            total += n;
        } else {
            printf("  [%d] %s: SKIP\n", i+1, files[i]);
        }
    }
    
    if (total == 0) {
        printf("\nERROR: No valid samples found.\n");
        return 1;
    }
    
    printf("\nTotal: %'d samples\n", total);
    
    TrainingSample *all = malloc(total * sizeof(TrainingSample));
    if (!all) {
        printf("ERROR: Out of memory.\n");
        return 1;
    }
    
    int offset = 0;
    for (int i = 0; i < file_count; i++) {
        int n = dataset_get_count(files[i]);
        if (n > 0) {
            dataset_load(files[i], all + offset, n);
            offset += n;
        }
    }
    
    printf("\nSaving to %s... ", output);
    fflush(stdout);
    
    remove(output);
    dataset_save_append(output, all, total);
    
    printf("Done! (%.2f MB)\n", (float)(total * sizeof(TrainingSample)) / (1024*1024));
    
    free(all);
    return 0;
}

// =============================================================================
// GLOB - Find files matching pattern
// =============================================================================

static int data_glob(const char *dir, const char *pattern, char ***out_files, int *out_count) {
    DIR *d = opendir(dir);
    if (!d) {
        printf("ERROR: Cannot open directory: %s\n", dir);
        return 1;
    }
    
    int capacity = 64;
    char **files = malloc(capacity * sizeof(char*));
    int count = 0;
    
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        if (strstr(entry->d_name, pattern)) {
            if (count >= capacity) {
                capacity *= 2;
                files = realloc(files, capacity * sizeof(char*));
            }
            size_t len = strlen(dir) + strlen(entry->d_name) + 2;
            files[count] = malloc(len);
            snprintf(files[count], len, "%s/%s", dir, entry->d_name);
            count++;
        }
    }
    closedir(d);
    
    *out_files = files;
    *out_count = count;
    return 0;
}

// =============================================================================
// DEDUPE - Remove duplicate positions
// =============================================================================

typedef struct {
    uint64_t hash;
    int index;
} HashIndex;

static int compare_hash_index(const void *a, const void *b) {
    uint64_t ha = ((const HashIndex*)a)->hash;
    uint64_t hb = ((const HashIndex*)b)->hash;
    return (ha > hb) - (ha < hb);
}

static int data_dedupe(const char *input, const char *output) {
    printf("=== Dataset Deduplicator ===\n\n");
    printf("Input:  %s\n", input);
    printf("Output: %s\n\n", output);
    
    int count = dataset_get_count(input);
    if (count <= 0) {
        printf("ERROR: Cannot read file or empty.\n");
        return 1;
    }
    
    TrainingSample *samples = malloc(count * sizeof(TrainingSample));
    HashIndex *hashes = malloc(count * sizeof(HashIndex));
    if (!samples || !hashes) {
        printf("ERROR: Out of memory.\n");
        return 1;
    }
    
    dataset_load(input, samples, count);
    
    // Build hash index
    for (int i = 0; i < count; i++) {
        hashes[i].hash = samples[i].state.hash;
        hashes[i].index = i;
    }
    
    // Sort by hash
    qsort(hashes, count, sizeof(HashIndex), compare_hash_index);
    
    // Mark unique samples (keep first occurrence of each hash)
    int *keep = calloc(count, sizeof(int));
    int unique_count = 0;
    
    for (int i = 0; i < count; i++) {
        int is_dup = (i > 0 && hashes[i].hash == hashes[i-1].hash);
        if (!is_dup) {
            keep[hashes[i].index] = 1;
            unique_count++;
        }
    }
    
    int removed = count - unique_count;
    printf("Original:   %'d samples\n", count);
    printf("Duplicates: %'d (%.1f%%)\n", removed, 100.0f * removed / count);
    printf("Unique:     %'d samples\n\n", unique_count);
    
    // Copy unique samples
    TrainingSample *unique = malloc(unique_count * sizeof(TrainingSample));
    if (!unique) {
        printf("ERROR: Out of memory.\n");
        return 1;
    }
    
    int j = 0;
    for (int i = 0; i < count; i++) {
        if (keep[i]) {
            unique[j++] = samples[i];
        }
    }
    
    // Save
    printf("Saving to %s... ", output);
    fflush(stdout);
    
    remove(output);
    dataset_save_append(output, unique, unique_count);
    
    printf("Done! (%.2f MB)\n", (float)(unique_count * sizeof(TrainingSample)) / (1024*1024));
    
    free(samples);
    free(hashes);
    free(keep);
    free(unique);
    return 0;
}

// =============================================================================
// TRIM - Keep only last N samples
// =============================================================================

static int data_trim(const char *file, int keeping) {
    int count = dataset_get_count(file);
    if (count <= keeping) {
        printf("Dataset size (%d) <= limit (%d). No trim needed.\n", count, keeping);
        return 0;
    }

    printf("Trimming dataset from %d to %d samples...\n", count, keeping);
    
    TrainingSample *samples = malloc(count * sizeof(TrainingSample));
    if (!samples) {
        printf("ERROR: Out of memory loading for trim.\n");
        return 1;
    }
    
    // Load ALL (inefficient but safe for now)
    // Optimization: seek and load only last N if 'dataset_load' supports partial/seek
    // Current API loads from start. 
    dataset_load(file, samples, count);
    
    int start_idx = count - keeping;
    if (start_idx < 0) start_idx = 0;
    
    // Save Overwrite
    // We create a temp file then rename to be safe
    char tmp_path[256];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", file);
    
    remove(tmp_path);
    if (dataset_save(tmp_path, &samples[start_idx], keeping) != 0) {
        printf("ERROR: Failed to save trim file.\n");
        free(samples);
        return 1;
    }
    
    free(samples);
    
    if (rename(tmp_path, file) != 0) {
        printf("ERROR: Failed to replace original file.\n");
        return 1;
    }
    
    printf("Trim complete.\n");
    return 0;
}

// =============================================================================
// CMD_DATA - Entry point
// =============================================================================

int cmd_data(int argc, char **argv) {
    setlocale(LC_NUMERIC, "");
    
    if (argc < 2) {
        printf("Usage: dama data <subcommand> [args]\n\n");
        printf("Subcommands:\n");
        printf("  inspect <file>              Show dataset statistics\n");
        printf("  dedupe <input> [-o <out>]   Remove duplicate positions\n");
        printf("  merge <file1> <file2> ...   Merge files\n");
        printf("  merge -d <dir> -p <pattern> Merge matching files\n");
        printf("  merge -o <output> ...       Specify output file\n");
        return 1;
    }
    
    const char *subcmd = argv[1];
    
    if (strcmp(subcmd, "inspect") == 0) {
        if (argc < 3) {
            printf("Usage: dama data inspect <file.bin>\n");
            return 1;
        }
        return data_inspect(argv[2]);
    }
    
    if (strcmp(subcmd, "merge") == 0) {
        if (argc < 4) {
            printf("Usage: dama data merge <file1.bin> <file2.bin> ...\n");
            return 1;
        }
        
        const char *output = "out/data/merged.bin";
        char **files = NULL;
        int file_count = 0;
        
        int argi = 2;
        while (argi < argc) {
            if (strcmp(argv[argi], "-o") == 0 && argi+1 < argc) {
                output = argv[++argi];
                argi++;
            } else if (strcmp(argv[argi], "-d") == 0 && argi+1 < argc) {
                const char *dir = argv[++argi];
                argi++;
                const char *pattern = ".bin";
                if (argi < argc && strcmp(argv[argi], "-p") == 0 && argi+1 < argc) {
                    pattern = argv[++argi];
                    argi++;
                }
                data_glob(dir, pattern, &files, &file_count);
                break;
            } else {
                file_count = argc - argi;
                files = &argv[argi];
                break;
            }
        }
        
        if (file_count < 1) {
            printf("ERROR: No input files.\n");
            return 1;
        }
        
        return data_merge(file_count, files, output);
    }
    
    if (strcmp(subcmd, "dedupe") == 0) {
        if (argc < 3) {
            printf("Usage: dama data dedupe <input.bin> [-o <output.bin>]\n");
            return 1;
        }
        const char *input = argv[2];
        const char *output = "out/data/deduped.bin";
        
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "-o") == 0 && i+1 < argc) {
                output = argv[++i];
            }
        }
        return data_dedupe(input, output);
    }

    if (strcmp(subcmd, "trim") == 0) {
        if (argc < 4) {
             printf("Usage: dama data trim <file.bin> <max_samples>\n");
             return 1;
        }
        return data_trim(argv[2], atoi(argv[3]));
    }
    
    printf("Unknown subcommand: %s\n", subcmd);
    return 1;
}
