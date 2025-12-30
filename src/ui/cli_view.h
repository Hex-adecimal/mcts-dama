/**
 * cli_view.h - CLI UI View Classes
 * 
 * Contains:
 * - Format helpers (format_num, format_time)
 * - View structures for different screens
 * - View rendering functions
 */

#ifndef CLI_VIEW_H
#define CLI_VIEW_H

#include <stdio.h>
#include <string.h>

// =============================================================================
// FORMAT HELPERS (inline)
// =============================================================================

static inline const char* format_num(long long n) {
    static char buffers[4][64];
    static int buf_idx = 0;
    char *buf = buffers[buf_idx];
    buf_idx = (buf_idx + 1) % 4;
    
    char temp[64];
    sprintf(temp, "%lld", n);
    int len = (int)strlen(temp);
    int out_idx = 0;
    for (int i = 0; i < len; i++) {
        if (i > 0 && (len - i) % 3 == 0 && temp[i] != '-') {
            buf[out_idx++] = ',';
        }
        buf[out_idx++] = temp[i];
    }
    buf[out_idx] = '\0';
    return buf;
}

static inline const char* format_time(double seconds) {
    static char buf[64];
    if (seconds < 60) {
        sprintf(buf, "%.1fs", seconds);
    } else if (seconds < 3600) {
        int m = (int)(seconds / 60);
        int s = (int)seconds % 60;
        sprintf(buf, "%dm %02ds", m, s);
    } else {
        int h = (int)(seconds / 3600);
        int m = ((int)seconds % 3600) / 60;
        sprintf(buf, "%dh %02dm", h, m);
    }
    return buf;
}

static inline const char* format_metric(double val) {
    static char buffers[4][64];
    static int buf_idx = 0;
    char *buf = buffers[buf_idx];
    buf_idx = (buf_idx + 1) % 4;
    
    if (val >= 1000000) {
        sprintf(buf, "%.1fM", val / 1000000.0);
    } else if (val >= 1000) {
        sprintf(buf, "%.1fk", val / 1000.0);
    } else {
        sprintf(buf, "%.0f", val);
    }
    return buf;
}

// =============================================================================
// VIEW STRUCTURES
// =============================================================================

typedef struct {
    const char *date_str;
    const char *output_file;
    int num_games;
    int mcts_nodes;
    int temp_threshold;
    int max_moves;
    int omp_threads;
    float temperature;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    float endgame_prob;
} SelfplayView;

typedef struct {
    int epochs;
    int batch_size;
    int patience;
    int omp_threads;
    int total_params;
    float learning_rate;
    float l2_decay;
} TrainingConfigView;

typedef struct {
    const char *weights_file;
    float best_loss;
    double training_time;
    double samples_per_sec;
} TrainingResultView;

typedef struct {
    const char *path;
    int count;
    float file_size_mb;
    
    // Value stats
    int wins, losses, draws;
    float val_mean, val_min, val_max;
    
    // Policy stats
    float avg_moves;
    float avg_entropy, max_entropy, entropy_ratio_pct;
    float avg_max_prob_pct;
    int sharp_policies; 
    float sharp_ratio_pct;
    
    // Occupancy
    float avg_pieces, avg_pawns, avg_ladies;
    int min_pieces, max_pieces;
    float lady_ratio_pct;
    
    // Phase
    int phase_opening, phase_midgame, phase_endgame;
    
    // Histogram buckets (grouped)
    const int *buckets; // Array of 6 ints
    
    // Duplicates
    int duplicates;
    int duplicates_checked;
    float duplicate_ratio_pct;
} DatasetStatsView;

typedef struct {
    int id;
    char name[32];
    int nodes;
    float explore_c;  // Either ucb1_c or puct_c depending on mode
    char features[64];
} TournamentPlayerInfo;

typedef struct {
    int count;
    const TournamentPlayerInfo *players;
} TournamentRosterView;

typedef struct {
    int rank;
    char name[32];
    double points;
    int wins, losses, draws;
    double elo;
    long long avg_iters;
    long long avg_nodes;
    double win_rate_pct;
} TournamentPlayerStats;

typedef struct {
    int count;
    const TournamentPlayerStats *players;
} TournamentLeaderboardView;

// =============================================================================
// VIEW RENDERING FUNCTIONS
// =============================================================================

void cli_view_print_selfplay(const SelfplayView *view);
void cli_view_print_training_config(const TrainingConfigView *view);
void cli_view_print_training_complete(const TrainingResultView *view);
void cli_view_print_dataset_stats(const DatasetStatsView *view);
void cli_view_print_tournament_roster(const TournamentRosterView *view);
void cli_view_print_tournament_leaderboard(const TournamentLeaderboardView *view);

#endif // CLI_VIEW_H
