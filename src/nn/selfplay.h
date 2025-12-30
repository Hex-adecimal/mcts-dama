#ifndef SELFPLAY_H
#define SELFPLAY_H

#include "mcts.h"
#include "cnn.h"

// =============================================================================
// CONFIGURATION
// =============================================================================

typedef struct {
    float threshold;      // e.g. 0.95 or -0.95 to trigger mercy
    int check_interval;   // Check every N moves
} MercyConfig;

typedef struct {
    int games;
    int max_moves;
    double time_limit;    // Per move
    float temp;           // Temperature
    const char *output_file;
    int parallel_threads;
    
    MercyConfig mercy;
    int overwrite_data;   // If 1, overwrites output file; else appends
    float endgame_prob;   // Probability of starting from endgame (0.0-1.0)
    
    // Callbacks
    void (*on_start)(int total_games);
    void (*on_game_complete)(int game_idx, int total_games, int winner, int moves, int reason);
    void (*on_progress)(int completed, int total, int w, int l, int d);
} SelfplayConfig;

// =============================================================================
// API
// =============================================================================

/**
 * Runs self-play data generation.
 * @param sp_cfg Self-play configuration
 * @param mcts_cfg MCTS configuration (used for both players)
 * @param cnn_weights CNN weights (optional, can be NULL)
 */
void selfplay_run(const SelfplayConfig *sp_cfg, const MCTSConfig *mcts_cfg);

#endif // SELFPLAY_H
