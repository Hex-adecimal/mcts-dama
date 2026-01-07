/**
 * cli_view.c - CLI UI View Class Implementation
 * 
 * Contains box-style headers and formatted output for:
 * - Self-play generation
 * - Training configuration
 * - Training results
 */

#include "dama/common/cli_view.h"
#include "dama/common/logging.h"
#include <stdio.h>

// =============================================================================
// SELFPLAY HEADER
// =============================================================================

void cli_view_print_selfplay(const SelfplayView *view) {
    log_printf("\n┌────────────────────────────────────────────────────────────────────┐\n");
    log_printf("│                       SELF-PLAY GENERATION                        │\n");
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  Date         : %-50s │\n", view->date_str);
    log_printf("│  Games        : %-6d                                             │\n", view->num_games);
    log_printf("│  Output       : %-50s │\n", view->output_file);
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  MCTS Nodes   : %-4d (Symmetric)                                   │\n", 
               view->mcts_nodes);
    log_printf("│  Dirichlet    : α=%.2f, ε=%.2f (first 30 moves)                    │\n", 
               view->dirichlet_alpha, view->dirichlet_epsilon);
    log_printf("│  Temperature  : %.1f → 0.05 (after move %d)                         │\n", 
               view->temperature, view->temp_threshold);
    char openings_buf[64];
    if (view->endgame_prob > 0.0f) {
        snprintf(openings_buf, sizeof(openings_buf), "Mixed (%.0f%% End, %.0f%% Rand)", 
                 view->endgame_prob * 100.0f, (1.0f - view->endgame_prob) * 100.0f);
    } else {
        strcpy(openings_buf, "2 Random Moves");
    }
    log_printf("│  Openings     : %-50s │\n", openings_buf);
    log_printf("│  Opponent     : Self-Play (Current Model)                          │\n");
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  Adjudication : Mercy Rule (Intermediate)                          │\n");
    log_printf("│  Soft Rewards : Checkmate ±1.0 │ Mercy ±0.7 │ Draw 0.0             │\n");
    log_printf("│  Max Moves    : %d (or 150 with mercy)                             │\n", view->max_moves);
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  OMP Threads  : %-2d                                                 │\n", view->omp_threads);
    log_printf("│  Backend      : Apple Accelerate (BLAS/vDSP)                       │\n");
    log_printf("└────────────────────────────────────────────────────────────────────┘\n\n");
}

// =============================================================================
// TRAINING CONFIG HEADER
// =============================================================================

void cli_view_print_training_config(const TrainingConfigView *view) {
    log_printf("\n┌────────────────────────────────────────────────────────────────────┐\n");
    log_printf("│                        CNN TRAINING CONFIG                         │\n");
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  Network      : 4 Conv (64ch) + Policy (512) + Value (256→1)       │\n");
    log_printf("│  Parameters   : %s (~%.1f MB)                               │\n", 
               format_num(view->total_params), view->total_params * 4.0f / 1024 / 1024);
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  Epochs       : %-4d            Batch Size : %-4d                  │\n", 
               view->epochs, view->batch_size);
    log_printf("│  Learning Rate: %.6f        L2 Decay   : %.1e                │\n", 
               view->learning_rate, view->l2_decay);
    log_printf("│  LR Warmup    : Epoch 1 ramp   Early Stop : %d epochs patience     │\n", view->patience);
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  Rewards      : Checkmate ±1.0 │ Mercy ±0.7 │ Draw 0.0             │\n");
    log_printf("│  Canonical    : Board flipped for Black (always \"my turn\")         │\n");
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  OMP Threads  : %-2d               Backend: Apple Accelerate         │\n", view->omp_threads);
    log_printf("└────────────────────────────────────────────────────────────────────┘\n\n");
    
    log_printf("+-------+---------------------------+---------------------------+------------+\n");
    log_printf("| Epoch |        Train Loss         |         Val Loss          |   Status   |\n");
    log_printf("|       |  Total  | Policy  | Value |  Total  | Policy  | Value |            |\n");
    log_printf("+-------+---------+---------+-------+---------+---------+-------+------------+\n");
}

// =============================================================================
// TRAINING COMPLETE
// =============================================================================

void cli_view_print_training_complete(const TrainingResultView *view) {
    log_printf("+-------+---------+---------+-------+---------+---------+-------+------------+\n");
    log_printf("\n┌────────────────────────────────────────────────────────────────────┐\n");
    log_printf("│                       TRAINING COMPLETE                            │\n");
    log_printf("├────────────────────────────────────────────────────────────────────┤\n");
    log_printf("│  Best Loss    : %.4f                                              │\n", view->best_loss);
    log_printf("│  Weights      : %-50s │\n", view->weights_file);
    log_printf("│  Time         : %-20s                               │\n", format_time(view->training_time));
    log_printf("│  Throughput   : %s samples/sec                             │\n", 
               format_num((long long)view->samples_per_sec));
    log_printf("└────────────────────────────────────────────────────────────────────┘\n");
}

// =============================================================================
// DATASET STATS
// =============================================================================

void cli_view_print_dataset_stats(const DatasetStatsView *view) {
    log_printf("=== Dataset Inspector ===\n\n");
    log_printf("File: %s\n\n", view->path);
    
    log_printf("Samples:    %s\n", format_num(view->count));
    log_printf("File size:  %.2f MB\n\n", view->file_size_mb);
    
    log_printf("=== Value Distribution ===\n");
    log_printf("  Wins:   %s (%.1f%%)\n", format_num(view->wins), 100.0f * view->wins / view->count);
    log_printf("  Losses: %s (%.1f%%)\n", format_num(view->losses), 100.0f * view->losses / view->count);
    log_printf("  Draws:  %s (%.1f%%)\n", format_num(view->draws), 100.0f * view->draws / view->count);
    log_printf("\n  Mean: %.4f | Min: %.4f | Max: %.4f\n", view->val_mean, view->val_min, view->val_max);
    
    log_printf("\n=== Policy Stats ===\n");
    log_printf("  Avg moves/sample: %.1f\n", view->avg_moves);
    
    log_printf("  Avg entropy:      %.2f (max=%.2f, ratio=%.1f%%)\n", 
           view->avg_entropy, view->max_entropy, view->entropy_ratio_pct);
    log_printf("  Avg max prob:     %.1f%%\n", view->avg_max_prob_pct);
    log_printf("  Sharp policies:   %s (%.1f%%) [max_prob > 50%%]\n", 
           format_num(view->sharp_policies), view->sharp_ratio_pct);
    
    if (view->entropy_ratio_pct > 90.0f) {
        log_printf("  ⚠️  High entropy suggests flat/uninformative policy targets!\n");
    }
    
    log_printf("\n=== Board Occupancy ===\n");
    log_printf("  Avg pieces/board:  %.1f\n", view->avg_pieces);
    log_printf("  Piece range:       %d - %d\n", view->min_pieces, view->max_pieces);
    log_printf("  Avg pawns:         %.1f\n", view->avg_pawns);
    log_printf("  Avg ladies:        %.1f\n", view->avg_ladies);
    log_printf("  Lady ratio:        %.1f%%\n", view->lady_ratio_pct);
    
    log_printf("\n=== Game Phase Distribution ===\n");
    log_printf("  Opening  (20-24 pcs): %s (%.1f%%)\n", 
               format_num(view->phase_opening), 100.0f * view->phase_opening / view->count);
    log_printf("  Midgame  (10-19 pcs): %s (%.1f%%)\n", 
               format_num(view->phase_midgame), 100.0f * view->phase_midgame / view->count);
    log_printf("  Endgame  ( 1-9  pcs): %s (%.1f%%)\n", 
               format_num(view->phase_endgame), 100.0f * view->phase_endgame / view->count);
    
    // Mini histogram (grouped)
    log_printf("\n  Piece count histogram:\n");
    const char *bucket_labels[] = {"1-4", "5-8", "9-12", "13-16", "17-20", "21-24"};
    int max_bucket = 1;
    for (int i = 0; i < 6; i++) if (view->buckets[i] > max_bucket) max_bucket = view->buckets[i];
    
    for (int i = 0; i < 6; i++) {
        int bar_len = (view->buckets[i] * 30) / max_bucket;
        log_printf("    %6s: ", bucket_labels[i]);
        for (int j = 0; j < bar_len; j++) log_printf("█");
        log_printf(" %s\n", format_num(view->buckets[i]));
    }
    
    log_printf("\n=== Duplicate Detection ===\n");
    if (view->duplicates_checked) {
        log_printf("  Duplicate positions: %s (%.2f%%)\n", 
                   format_num(view->duplicates), view->duplicate_ratio_pct);
        if (view->duplicate_ratio_pct > 10.0f) {
            log_printf("  ⚠️  High duplicate rate may indicate overfitting risk\n");
        }
    } else {
        log_printf("  Skipped (dataset too large, > 200k samples)\n");
    }
}

// =============================================================================
// TOURNAMENT VIEWS
// =============================================================================

void cli_view_print_tournament_roster(const TournamentRosterView *view) {
    log_printf("\n");
    log_printf("┌────┬──────────────────┬────────┬───────┬──────────────────────┐\n");
    log_printf("│ ID │ Model Name       │ Nodes  │ Exp-C │ Features             │\n");
    log_printf("├────┼──────────────────┼────────┼───────┼──────────────────────┤\n");
    for (int i = 0; i < view->count; i++) {
        const TournamentPlayerInfo *p = &view->players[i];
        log_printf("│ %-2d │ %-16s │ %-6d │ %-5.2f │ %-20s │\n", 
               p->id, p->name, p->nodes, p->explore_c, p->features);
    }
    log_printf("└────┴──────────────────┴────────┴───────┴──────────────────────┘\n");
    log_printf("\n");
}

void cli_view_print_tournament_leaderboard(const TournamentLeaderboardView *view) {
    log_printf("\n\n");
    log_printf("┌──────┬────────────────────────┬────────┬──────┬──────┬──────┬──────┬────────┬────────┬────────┬──────┬──────┬──────┬──────────┐\n");
    log_printf("│ Rank │ Name                   │ Points │ Wins │ Loss │ Draw │ ELO  │ IPS    │ NPS    │ Depth  │ BF   │ Eff%% │ Mem  │ Win Rate │\n");
    log_printf("├──────┼────────────────────────┼────────┼──────┼──────┼──────┼──────┼────────┼────────┼────────┼──────┼──────┼──────┼──────────┤\n");
    
    for (int i = 0; i < view->count; i++) {
        const TournamentPlayerStats *p = &view->players[i];
        log_printf("│ %-4d │ %-22s │ %-6.1f │ %-4d │ %-4d │ %-4d │ %-4.0f │ %-6s │ %-6s │ %-6.1f │ %-4.1f │ %3.0f%% │ %-4s │ %5.1f%%   │\n", 
               p->rank, p->name, p->points, p->wins, p->losses, p->draws, p->elo, 
               format_metric(p->ips), format_metric(p->nps), p->avg_depth, p->avg_bf, 
               p->efficiency * 100.0, format_metric(p->peak_mem_mb * 1024 * 1024), p->win_rate_pct);
    }
    log_printf("└──────┴────────────────────────┴────────┴──────┴──────┴──────┴──────┴────────┴────────┴────────┴──────┴──────┴──────┴──────────┘\n");
}
