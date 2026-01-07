/**
 * mcts_tree_stats.h - Tree Statistics Helpers
 * 
 * Reusable functions to compute and report MCTS tree statistics:
 * - Branching Factor
 * - Tree Depth (max/avg)
 * - TT Hit Rate
 * - Peak Memory (RSS)
 */

#ifndef MCTS_TREE_STATS_H
#define MCTS_TREE_STATS_H

#include "dama/search/mcts_config.h"
#include <stdio.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

// =============================================================================
// TREE STATISTICS STRUCT
// =============================================================================

typedef struct {
    double branching_factor;    // total_children / nodes_with_children
    int max_depth;
    double avg_depth;
    double tt_hit_rate;         // tt_hits / (tt_hits + tt_misses)
    size_t peak_rss_bytes;      // Peak Resident Set Size
} TreeStats;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Compute tree statistics from MCTSStats.
 */
static inline TreeStats compute_tree_stats(const MCTSStats *stats) {
    TreeStats ts = {0};
    
    // Branching factor
    if (stats->nodes_with_children > 0) {
        ts.branching_factor = (double)stats->total_children_expanded / stats->nodes_with_children;
    }
    
    // Depth (we have total_depth and total_expansions)
    if (stats->total_expansions > 0) {
        ts.avg_depth = (double)stats->total_depth / stats->total_expansions;
    }
    
    // TT hit rate
    long tt_total = stats->tt_hits + stats->tt_misses;
    if (tt_total > 0) {
        ts.tt_hit_rate = (double)stats->tt_hits / tt_total;
    }
    
    // Peak RSS
    ts.peak_rss_bytes = stats->peak_memory_bytes;
    
    return ts;
}

/**
 * Get current process peak RSS (macOS only, returns 0 on other platforms).
 */
static inline size_t get_peak_rss(void) {
#ifdef __APPLE__
    struct task_basic_info info;
    mach_msg_type_number_t size = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &size) == KERN_SUCCESS) {
        return info.resident_size;
    }
#endif
    return 0;
}

/**
 * Print tree statistics to stdout in a formatted table.
 */
static inline void print_tree_stats(const TreeStats *ts) {
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║                 TREE STATISTICS                      ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║ Branching Factor:        %10.2f                 ║\n", ts->branching_factor);
    printf("║ Average Depth:           %10.2f                 ║\n", ts->avg_depth);
    printf("║ TT Hit Rate:             %10.2f%%                ║\n", ts->tt_hit_rate * 100.0);
    printf("║ Peak Memory (RSS):       %10.2f MB              ║\n", ts->peak_rss_bytes / (1024.0 * 1024.0));
    printf("╚══════════════════════════════════════════════════════╝\n");
}

/**
 * Print tree statistics as a one-liner for tournament/match output.
 */
static inline void print_tree_stats_oneline(const TreeStats *ts) {
    printf("[TreeStats] BF=%.2f  AvgDepth=%.1f  TT_Hit=%.1f%%  RSS=%.1fMB\n",
           ts->branching_factor, ts->avg_depth, ts->tt_hit_rate * 100.0,
           ts->peak_rss_bytes / (1024.0 * 1024.0));
}

#endif // MCTS_TREE_STATS_H
