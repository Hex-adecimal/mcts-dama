/**
 * cmd_tournament.c - Tournament Controller
 */

#include "dama/common/logging.h"
#include "dama/common/cli_view.h"
#include "dama/search/tournament.h"
#include "dama/engine/movegen.h"
#include "dama/neural/cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =============================================================================
// CALLBACKS
// =============================================================================

static TournamentPlayer *g_players = NULL;

static void on_start(int total) {
    printf("\nStarting Round-Robin Tournament: %d matches scheduled\n", total);
    printf("Legend: ips = Iterations/sec (Speed) | nps = Nodes/sec (Search Volume)\n");
}

static void on_match_start(int i, int j, const char *n1, const char *n2) {
    (void)i; (void)j;
    printf("\n[%s vs %s]\n", n1, n2);
}

static void on_match_end(int i, int j, int s1, int s2, int d) {
    (void)i; (void)j;
    // Simple inline result print. Could use View if complex.
    printf("Result: %d - %d (Draws: %d)\n", s1, s2, d);
}

static void on_game_complete(const TournamentGameResult *r) {
    if (!g_players) return;
    const char *n1 = g_players[r->p1_idx].name;
    const char *n2 = g_players[r->p2_idx].name;
    
    const char *winner = "Draw";
    if (r->result == 1) winner = n1;
    else if (r->result == -1) winner = n2;
    
    double dur = r->duration;

    printf("  > Game: %-15s vs %-15s -> %s (%d moves, %.2fs)\n", 
           n1, n2, winner, r->moves, dur);
           
    // P1 Stats (with restored KB and Depth)
    double d1 = r->duration_p1;
    int moves1 = r->s1.total_moves > 0 ? r->s1.total_moves : 1;
    double kb1 = (double)r->s1.total_memory / moves1 / 1024.0;
    double depth1 = (double)r->s1.total_depth / moves1;
    double ips1 = (d1 > 0.001) ? (double)r->s1.total_iterations / d1 : 0.0;
    printf("    [%-15s]: %s iters, %s nodes | %.1f KB, D%.1f | %s ips (%.2fs)\n",
           n1, format_num(r->s1.total_iterations), format_num(r->s1.total_nodes), 
           kb1, depth1, format_metric(ips1), d1);

    // P2 Stats (with restored KB and Depth)
    double d2 = r->duration_p2;
    int moves2 = r->s2.total_moves > 0 ? r->s2.total_moves : 1;
    double kb2 = (double)r->s2.total_memory / moves2 / 1024.0;
    double depth2 = (double)r->s2.total_depth / moves2;
    double ips2 = (d2 > 0.001) ? (double)r->s2.total_iterations / d2 : 0.0;
    printf("    [%-15s]: %s iters, %s nodes | %.1f KB, D%.1f | %s ips (%.2fs)\n",
           n2, format_num(r->s2.total_iterations), format_num(r->s2.total_nodes), 
           kb2, depth2, format_metric(ips2), d2);
}

static void on_tournament_end(TournamentPlayer *players, int count) {
    // Map to View
    TournamentPlayerStats *stats = malloc(count * sizeof(TournamentPlayerStats));
    
    // Sort logic is handled by sort before print? No, Tournament view expects sorted or sorts itself?
    // cli_view logic was just printing.
    // The previous cmd_tournament logic did sort. 
    // We should sort here before passing to View.
    
    // Sort pointers
    for (int k = 0; k < count; k++) {
        TournamentPlayer *p = &players[k];
        int total = p->wins + p->losses + p->draws;
        double wr = (total > 0) ? 100.0 * p->wins / total : 0;
        
        // We need to map to stats struct
        // But we want to sort. 
        // Let's populate first, then qsort the stats array.
        stats[k].rank = 0; // Filled after sort
        strncpy(stats[k].name, p->name, 31);
        stats[k].points = p->points;
        stats[k].wins = p->wins;
        stats[k].losses = p->losses;
        stats[k].draws = p->draws;
        stats[k].elo = p->elo;
        stats[k].avg_iters = (p->total_moves>0) ? p->total_iters/p->total_moves : 0;
        stats[k].avg_nodes = (p->total_moves>0) ? p->total_nodes/p->total_moves : 0;
        stats[k].win_rate_pct = wr;
    }
    
    // Bubble sort or qsort stats array
    for (int check=0; check<count-1; check++) {
        for (int k=0; k<count-check-1; k++) {
            if (stats[k].points < stats[k+1].points) {
                TournamentPlayerStats tmp = stats[k];
                stats[k] = stats[k+1];
                stats[k+1] = tmp;
            }
        }
    }
    
    for(int k=0; k<count; k++) stats[k].rank = k+1;
    
    TournamentLeaderboardView view = { .count = count, .players = stats };
    cli_view_print_tournament_leaderboard(&view);
    
    free(stats);
}

// =============================================================================
// SETUP
// =============================================================================

static int setup_roster(TournamentPlayer *players, CNNWeights *v3, CNNWeights *active, int nodes) {
    int n = 0;
    
    // 1. CNN-V3 (Benchmark)
    MCTSConfig cnn3 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO); 
    cnn3.max_nodes = nodes; 
    cnn3.cnn_weights = v3;
    players[n++] = (TournamentPlayer){.name="CNN-V3", .desc="Benchmark (Stable)", .config=cnn3, .elo=1200};

    // 2. Grandmaster (Hybrid with V3)
    MCTSConfig gm = mcts_get_preset(MCTS_PRESET_GRANDMASTER); 
    gm.max_nodes = nodes;
    gm.cnn_weights = v3; 
    players[n++] = (TournamentPlayer){.name="Grandmaster-V3", .desc="Hybrid Heuristic", .config=gm, .elo=1200};
    
    // 3. CNN-New (Active Model)
    MCTSConfig cnn_new = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO); 
    cnn_new.max_nodes = nodes; 
    cnn_new.cnn_weights = active;
    players[n++] = (TournamentPlayer){.name="CNN-New", .desc="Active Training", .config=cnn_new, .elo=1200};
    
    return n;
}

// =============================================================================
// MAIN
// =============================================================================

int cmd_tournament(int argc, char **argv) {
    int games = 10; // per pairing
    double time_limit = 0.2;  // 200ms per move (time-based)
    int nodes = 0;  // 0 = no node limit, use time only
    int use_parallel = 1;

    char *p1_path = NULL;
    char *p2_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: dama tournament [options]\n");
            printf("Options:\n");
            printf("  -g <n>      Games per pair (default: 10)\n");
            printf("  -n <n>      MCTS nodes (default: 800)\n");
            printf("  -t <sec>    Time limit per move (default: 1.0)\n");
            printf("  --serial    Run games serially (default: parallel)\n");
            printf("  --p1-path <file>  Custom weights for Player 1 (Candidate)\n");
            printf("  --p2-path <file>  Custom weights for Player 2 (Opponent)\n");
            return 0;
        }
        else if (strcmp(argv[i], "-g") == 0 && i+1 < argc) games = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) nodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) time_limit = atof(argv[++i]);
        else if (strcmp(argv[i], "--serial") == 0) use_parallel = 0;
        else if (strcmp(argv[i], "--p1-path") == 0 && i+1 < argc) p1_path = argv[++i];
        else if (strcmp(argv[i], "--p2-path") == 0 && i+1 < argc) p2_path = argv[++i];
        else if (strcmp(argv[i], "--p1-type") == 0 && i+1 < argc) { i++; /* Ignore legacy arg */ }
        else if (strcmp(argv[i], "--p2-type") == 0 && i+1 < argc) { i++; /* Ignore legacy arg */ }
        else if (strcmp(argv[i], "--timeout") == 0 && i+1 < argc) time_limit = atof(argv[++i]); /* Alias for -t */
    }
    
    // Init Deps
    zobrist_init();
    init_move_tables();
    srand(time(NULL));
    
    // Load Weights
    CNNWeights w3, w_active;
    cnn_init(&w3); cnn_init(&w_active);
    
    TournamentPlayer players[16];
    int n = 0;

    if (p1_path && p2_path) {
        printf("Running Dual Tournament (Candidate vs Defender)...\n");
        if(cnn_load_weights(&w_active, p1_path) != 0) { printf("Error loading P1: %s\n", p1_path); return 1; }
        if(cnn_load_weights(&w3, p2_path) != 0) { printf("Error loading P2: %s\n", p2_path); return 1; }

        // Player 1: Candidate
        MCTSConfig c1 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO); 
        c1.max_nodes = nodes; c1.cnn_weights = &w_active;
        players[n++] = (TournamentPlayer){.name="Candidate", .desc="Training Candidate", .config=c1, .elo=1200};

        // Player 2: Best (Defender)
        MCTSConfig c2 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO); 
        c2.max_nodes = nodes; c2.cnn_weights = &w3; 
        players[n++] = (TournamentPlayer){.name="Best", .desc="Current Best", .config=c2, .elo=1200};
    } 
    else {
        // Default Three-Way
        if(cnn_load_weights(&w3, "out/models/cnn_weights_v3.bin") != 0) printf("Warning: V3 missing\n");
        if(cnn_load_weights(&w_active, "out/models/cnn_weights.bin") != 0) printf("Warning: Active model missing\n");
        n = setup_roster(players, &w3, &w_active, nodes);
    }
    
    // Run
    TournamentSystemConfig cfg = {
        .num_players = n,
        .players = players,
        .games_per_pair = games,
        .time_limit = time_limit,
        .parallel_games = use_parallel,
        .on_start = on_start,
        .on_match_start = on_match_start,
        .on_match_end = on_match_end,
        .on_game_complete = on_game_complete,
        .on_tournament_end = on_tournament_end
    };
    
    // Print Roster
    TournamentPlayerInfo infos[16];
    for(int i=0; i<n; i++) {
        infos[i].id = i+1;
        strncpy(infos[i].name, players[i].name, 31);
        infos[i].nodes = players[i].config.max_nodes;
        // Show whichever exploration constant is active (puct_c or ucb1_c)
        infos[i].explore_c = players[i].config.use_puct ? players[i].config.puct_c : players[i].config.ucb1_c;
        strncpy(infos[i].features, players[i].desc, 63);
    }
    TournamentRosterView rv = { .count=n, .players=infos };
    cli_view_print_tournament_roster(&rv);
    
    g_players = players;
    g_players = players;
    tournament_run(&cfg);
    g_players = NULL;

    // Helper for script parsing
    if (n == 2) {
        printf("Match Analysis: Player 1 (%s) Wins: %d\n", players[0].name, players[0].wins);
    }
    g_players = NULL;
    
    // Cleanup
    cnn_free(&w3); cnn_free(&w_active);
    
    return 0;
}
