/**
 * cmd_tournament.c - Tournament Controller
 */

#include "logging.h"
#include "../../src/ui/cli_view.h"
#include "../../src/mcts/tournament.h"
#include "../../src/core/movegen.h"
#include "cnn.h"
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

    printf("  > Game: %s vs %s -> %s (%d moves, %.2fs)\n", 
           n1, n2, winner, r->moves, dur);
           
    // P1 Stats
    double d1 = r->duration_p1;
    double nps1 = (d1 > 0.001) ? (double)r->s1.total_nodes / d1 : 0.0;
    double ips1 = (d1 > 0.001) ? (double)r->s1.total_iterations / d1 : 0.0;
    printf("    [%s]: %s iters, %s nodes | %s ips, %s nps (%.2fs)\n",
           n1, format_num(r->s1.total_iterations), format_num(r->s1.total_nodes), 
           format_metric(ips1), format_metric(nps1), d1);

    // P2 Stats
    double d2 = r->duration_p2;
    double nps2 = (d2 > 0.001) ? (double)r->s2.total_nodes / d2 : 0.0;
    double ips2 = (d2 > 0.001) ? (double)r->s2.total_iterations / d2 : 0.0;
    printf("    [%s]: %s iters, %s nodes | %s ips, %s nps (%.2fs)\n",
           n2, format_num(r->s2.total_iterations), format_num(r->s2.total_nodes), 
           format_metric(ips2), format_metric(nps2), d2);
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

static int setup_roster(TournamentPlayer *players, CNNWeights *v1, CNNWeights *v2, CNNWeights *v3, int nodes) {
    int n = 0;
    
    // Presets
    MCTSConfig pure = mcts_get_preset(MCTS_PRESET_PURE_VANILLA); pure.max_nodes = nodes;
    MCTSConfig van = mcts_get_preset(MCTS_PRESET_VANILLA); van.max_nodes = nodes;
    MCTSConfig gm = mcts_get_preset(MCTS_PRESET_GRANDMASTER); gm.max_nodes = nodes;
    MCTSConfig cnn1 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO); cnn1.max_nodes = nodes; cnn1.cnn_weights = v1;
    MCTSConfig cnn2 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO); cnn2.max_nodes = nodes; cnn2.cnn_weights = v2;
    MCTSConfig cnn3 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO); cnn3.max_nodes = nodes; cnn3.cnn_weights = v3;
    
    // Add players
    players[n++] = (TournamentPlayer){.name="PureVanilla", .desc="Random Rollouts", .config=pure, .elo=1200};
    players[n++] = (TournamentPlayer){.name="Vanilla", .desc="Std MCTS (Lookahead)", .config=van, .elo=1200};
    players[n++] = (TournamentPlayer){.name="Grandmaster", .desc="Hand-Tuned Heuristic + PUCT", .config=gm, .elo=1200};
    players[n++] = (TournamentPlayer){.name="CNN-V1", .desc="CNN (v1.bin)", .config=cnn1, .elo=1200};
    players[n++] = (TournamentPlayer){.name="CNN-V2", .desc="CNN (v2.bin)", .config=cnn2, .elo=1200};
    players[n++] = (TournamentPlayer){.name="CNN-New", .desc="CNN (v3.bin)", .config=cnn3, .elo=1200};
    
    return n;
}

// =============================================================================
// MAIN
// =============================================================================

int cmd_tournament(int argc, char **argv) {
    int games = 10; // per pairing
    double time_limit = 1.0;
    int nodes = 800;
    int use_parallel = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: dama tournament [options]\n");
            printf("Options:\n");
            printf("  -g <n>      Games per pair (default: 10)\n");
            printf("  -n <n>      MCTS nodes (default: 800)\n");
            printf("  -t <sec>    Time limit per move (default: 1.0)\n");
            printf("  --serial    Run games serially (default: parallel)\n");
            return 0;
        }
        else if (strcmp(argv[i], "-g") == 0 && i+1 < argc) games = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) nodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) time_limit = atof(argv[++i]);
        else if (strcmp(argv[i], "--serial") == 0) use_parallel = 0;
    }
    
    // Init Deps
    zobrist_init();
    init_move_tables();
    srand(time(NULL));
    
    // Load Weights
    CNNWeights w1, w2, w3;
    cnn_init(&w1); cnn_init(&w2); cnn_init(&w3);
    cnn_load_weights(&w1, "out/models/cnn_weights_v1.bin");
    cnn_load_weights(&w2, "out/models/cnn_weights_v2.bin");
    cnn_load_weights(&w3, "out/models/cnn_weights_v3.bin");
    
    // Setup Players
    TournamentPlayer players[16];
    int n = setup_roster(players, &w1, &w2, &w3, nodes);
    
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
        infos[i].puct = players[i].config.use_puct ? players[i].config.puct_c : 0;
        strncpy(infos[i].features, players[i].desc, 63);
    }
    TournamentRosterView rv = { .count=n, .players=infos };
    cli_view_print_tournament_roster(&rv);
    
    g_players = players;
    tournament_run(&cfg);
    g_players = NULL;
    
    // Cleanup
    cnn_free(&w1); cnn_free(&w2); cnn_free(&w3);
    
    return 0;
}
