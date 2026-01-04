#ifndef TOURNAMENT_H
#define TOURNAMENT_H

#include "dama/search/mcts.h"

// =============================================================================
// TYPES
// =============================================================================

typedef struct {
    char name[32];
    char desc[64]; // Human-readable configuration details
    MCTSConfig config;
    // Runtime stats
    double elo;
    int wins, losses, draws;
    double points;
    long long total_iters, total_nodes, total_moves;
} TournamentPlayer;

typedef struct {
    int p1_idx;
    int p2_idx;
    int result; // 1 = p1 wins, -1 = p2 wins, 0 = draw
    int moves;
    double duration;
    double duration_p1;
    double duration_p2;
    MCTSStats s1;
    MCTSStats s2;
} TournamentGameResult;

typedef struct {
    int num_players;
    TournamentPlayer *players;
    int games_per_pair;
    double time_limit;
    int parallel_games; // 1 = serial
    
    // Callbacks
    void (*on_start)(int total_matches);
    void (*on_match_start)(int p1, int p2, const char *n1, const char *n2);
    void (*on_game_complete)(const TournamentGameResult *res);
    void (*on_match_end)(int p1, int p2, int score1, int score2, int draws); //, const TournamentMatchStats *stats);
    void (*on_tournament_end)(TournamentPlayer *players, int count);
} TournamentSystemConfig;

// =============================================================================
// API
// =============================================================================

/**
 * Runs a Round-Robin tournament.
 * Updates players[i].elo and stats in-place.
 */
void tournament_run(TournamentSystemConfig *cfg);

/**
 * Calculates ELO ratings based on match matrix.
 * Helper exposed if needed, but tournament_run calls it automatically.
 */
void tournament_calculate_elo(TournamentPlayer *players, int n, int **results_matrix);

#endif // TOURNAMENT_H
