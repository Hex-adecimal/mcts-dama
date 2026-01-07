/**
 * cmd_tournament.c - Tournament Controller
 */

#include "dama/common/logging.h"
#include "dama/common/cli_view.h"
#include "dama/tournament/tournament.h"
#include "dama/engine/movegen.h"
#include "dama/engine/zobrist.h"
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
    printf("Legend: ips = Iters/s | nps = Nodes/s | D = Avg Depth | BF = Avg Branching Factor | Eff%% = TT Hit Rate | Mem = Peak Memory\n");
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
    
    printf("  > Game: %-15s vs %-15s -> %s (%d moves, %.2fs)\n", 
           n1, n2, winner, r->moves, r->duration);
           
    // P1 Stats
    double d1 = r->duration_p1;
    int moves1 = r->s1.total_moves > 0 ? r->s1.total_moves : 1;
    double depth1 = (double)r->s1.total_depth / moves1;
    double ips1 = (d1 > 0.001) ? (double)r->s1.total_iterations / d1 : 0.0;
    double nps1 = (d1 > 0.001) ? (double)r->s1.total_nodes / d1 : 0.0;
    long avg_iters1 = r->s1.total_iterations / moves1;
    double bf1 = (r->s1.nodes_with_children > 0) ? 
        (double)r->s1.total_children_expanded / r->s1.nodes_with_children : 0.0;
    double eff1 = (r->s1.tt_hits + r->s1.tt_misses > 0) ? 
        (double)r->s1.tt_hits * 100.0 / (r->s1.tt_hits + r->s1.tt_misses) : 0.0;
    
    printf("    [%-15s]: %s iters (%s/mv), %s nodes | D%2.1f, BF%.1f | %s ips, %s nps | Eff%2.0f%% | Mem: %s\n",
           n1, format_num(r->s1.total_iterations), format_num(avg_iters1), format_num(r->s1.total_nodes), 
           depth1, bf1, format_metric(ips1), format_metric(nps1), eff1,
           format_metric(r->s1.peak_memory_bytes));

    // P2 Stats
    double d2 = r->duration_p2;
    int moves2 = r->s2.total_moves > 0 ? r->s2.total_moves : 1;
    double depth2 = (double)r->s2.total_depth / moves2;
    double ips2 = (d2 > 0.001) ? (double)r->s2.total_iterations / d2 : 0.0;
    double nps2 = (d2 > 0.001) ? (double)r->s2.total_nodes / d2 : 0.0;
    long avg_iters2 = r->s2.total_iterations / moves2;
    double bf2 = (r->s2.nodes_with_children > 0) ? 
        (double)r->s2.total_children_expanded / r->s2.nodes_with_children : 0.0;
    double eff2 = (r->s2.tt_hits + r->s2.tt_misses > 0) ? 
        (double)r->s2.tt_hits * 100.0 / (r->s2.tt_hits + r->s2.tt_misses) : 0.0;
    
    printf("    [%-15s]: %s iters (%s/mv), %s nodes | D%2.1f, BF%.1f | %s ips, %s nps | Eff%2.0f%% | Mem: %s\n",
           n2, format_num(r->s2.total_iterations), format_num(avg_iters2), format_num(r->s2.total_nodes), 
           depth2, bf2, format_metric(ips2), format_metric(nps2), eff2,
           format_metric(r->s2.peak_memory_bytes));
}

static void on_tournament_end(TournamentPlayer *players, int count) {
    // Map to View
    TournamentPlayerStats *stats = malloc(count * sizeof(TournamentPlayerStats));
    
    // Sort logic
    for (int k = 0; k < count; k++) {
        TournamentPlayer *p = &players[k];
        double total_games = p->wins + p->losses + p->draws;
        double wr = (total_games > 0) ? 100.0 * p->wins / total_games : 0;
        
        strncpy(stats[k].name, p->name, 31);
        stats[k].points = p->points;
        stats[k].wins = p->wins;
        stats[k].losses = p->losses;
        stats[k].draws = p->draws;
        stats[k].elo = p->elo;
        stats[k].win_rate_pct = wr;

        // Cumulative Metrics
        double dur = (p->total_duration > 0.001) ? p->total_duration : 1.0;
        
        stats[k].ips = (double)p->total_iters / dur;
        stats[k].nps = (double)p->total_nodes / dur;
        stats[k].avg_depth = (p->total_moves > 0) ? (double)p->total_depth / p->total_moves : 0;
        
        // Exact Branching Factor: Children / NodesExpanded
        stats[k].avg_bf = (p->total_expansions > 0) ? 
            (double)p->total_children_expanded / p->total_expansions : 0;
        
        stats[k].peak_mem_mb = (double)p->peak_memory / (1024.0 * 1024.0);
        
        long total_tt = p->tt_hits + p->tt_misses;
        stats[k].efficiency = (total_tt > 0) ? (double)p->tt_hits / total_tt : 0;
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
    
    // 1. Classical Baselines
    MCTSConfig pure = mcts_get_preset(MCTS_PRESET_PURE_VANILLA);
    pure.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="Pure-Vanilla", .desc="No Lookahead, No Reuse", .config=pure, .elo=1200};

    MCTSConfig vanilla = mcts_get_preset(MCTS_PRESET_VANILLA);
    vanilla.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="Vanilla-Base", .desc="Basic Lookahead+Reuse", .config=vanilla, .elo=1200};

    // 2. CNN-Based Models
    MCTSConfig cnn3 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO); 
    cnn3.max_nodes = nodes; 
    cnn3.cnn_weights = v3;
    players[n++] = (TournamentPlayer){.name="AlphaZero-V3", .desc="Benchmark CNN (Stable)", .config=cnn3, .elo=1200};

    MCTSConfig gm = mcts_get_preset(MCTS_PRESET_GRANDMASTER); 
    gm.max_nodes = nodes;
    gm.cnn_weights = v3; 
    players[n++] = (TournamentPlayer){.name="Grandmaster-V3", .desc="Hybrid Heuristic+CNN", .config=gm, .elo=1200};
    
    MCTSConfig cnn_new = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO); 
    cnn_new.max_nodes = nodes; 
    cnn_new.cnn_weights = active;
    cnn_new.use_tt = 1; // TT active for the contender
    players[n++] = (TournamentPlayer){.name="AlphaZero-New", .desc="Active Model + TT", .config=cnn_new, .elo=1200};

    // 3. Feature-Specific Variants (Ablation Study)
    MCTSConfig tt = mcts_get_preset(MCTS_PRESET_TT_ONLY);
    tt.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="TT-Only", .desc="Vanilla + Transposition", .config=tt, .elo=1200};

    MCTSConfig solver = mcts_get_preset(MCTS_PRESET_SOLVER_ONLY);
    solver.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="Solver-Only", .desc="Checkmate Solver", .config=solver, .elo=1200};

    MCTSConfig tuned = mcts_get_preset(MCTS_PRESET_TUNED_ONLY);
    tuned.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="UCB1-Tuned", .desc="Variance-Aware Search", .config=tuned, .elo=1200};

    MCTSConfig fpu = mcts_get_preset(MCTS_PRESET_FPU_ONLY);
    fpu.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="FPU-Only", .desc="First Play Urgency", .config=fpu, .elo=1200};

    MCTSConfig decay = mcts_get_preset(MCTS_PRESET_DECAY_ONLY);
    decay.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="Decay-Only", .desc="Reward Decay Factor", .config=decay, .elo=1200};

    MCTSConfig smart = mcts_get_preset(MCTS_PRESET_SMART_ROLLOUTS);
    smart.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="Smart-Rollouts", .desc="Heuristic Rollouts", .config=smart, .elo=1200};

    MCTSConfig weights = mcts_get_preset(MCTS_PRESET_WEIGHTS_ONLY);
    weights.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="Weights-Only", .desc="Fixed Heuristics", .config=weights, .elo=1200};

    MCTSConfig pb = mcts_get_preset(MCTS_PRESET_PROG_BIAS_ONLY);
    pb.max_nodes = nodes;
    players[n++] = (TournamentPlayer){.name="Prog-Bias", .desc="Progressive Bias", .config=pb, .elo=1200};

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
    movegen_init();
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
    tournament_run(&cfg);
    g_players = NULL;

    // Helper for script parsing
    if (n == 2) {
        printf("Match Analysis: Player 1 (%s) Wins: %d\n", players[0].name, players[0].wins);
    }
    
    // Cleanup
    cnn_free(&w3); cnn_free(&w_active);
    
    return 0;
}
