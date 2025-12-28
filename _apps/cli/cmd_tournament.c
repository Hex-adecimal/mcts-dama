/**
 * tournament.c - MCTS Tournament System
 * 
 * Runs round-robin tournament between different MCTS configurations.
 * Refactored for modularity and readability.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "game.h"
#include "movegen.h"
#include "mcts.h"
#include "cnn.h"
#include "params.h"

// =============================================================================
// CONFIGURATION
// =============================================================================

#define DEFAULT_GAMES_PER_PAIRING   10
#define DEFAULT_TIME_LIMIT          TIME_HIGH
#define MAX_PLAYERS                 16
#define INITIAL_ELO                 1200.0

// =============================================================================
// TYPES
// =============================================================================

typedef struct {
    const char *name;
    MCTSConfig config;
    double elo;
    int wins, losses, draws;
    double points;
    long long total_iters, tree_nodes, total_moves;
    double total_time;
} Player;

typedef struct {
    int wins, losses, draws;
} MatchResult;

// =============================================================================
// UTILITY: Number formatting
// =============================================================================

static const char* fmt_num(long long n) {
    static char bufs[4][32];
    static int idx = 0;
    char *buf = bufs[idx++ % 4];
    char tmp[32];
    sprintf(tmp, "%lld", n);
    int len = strlen(tmp), out = 0;
    for (int i = 0; i < len; i++) {
        if (i > 0 && (len - i) % 3 == 0) buf[out++] = '.';
        buf[out++] = tmp[i];
    }
    buf[out] = '\0';
    return buf;
}

// =============================================================================
// ELO CALCULATION
// =============================================================================

static void calculate_elo(Player *players, int n, MatchResult results[MAX_PLAYERS][MAX_PLAYERS]) {
    // Initialize
    for (int i = 0; i < n; i++) players[i].elo = INITIAL_ELO;
    
    // Iterative solver
    for (int iter = 0; iter < 10000; iter++) {
        double new_elos[MAX_PLAYERS];
        double max_diff = 0.0;
        
        for (int i = 0; i < n; i++) {
            double actual = 0, expected = 0;
            
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                int games = results[i][j].wins + results[i][j].losses + results[i][j].draws;
                if (games == 0) continue;
                
                actual += results[i][j].wins + 0.5 * results[i][j].draws;
                double diff = fmin(fmax(players[j].elo - players[i].elo, -800), 800);
                expected += games / (1.0 + pow(10.0, diff / 400.0));
            }
            
            new_elos[i] = players[i].elo + (actual - expected) * 0.1;
        }
        
        // Recenter around 1200
        double sum = 0;
        for (int i = 0; i < n; i++) sum += new_elos[i];
        double shift = INITIAL_ELO - sum / n;
        
        for (int i = 0; i < n; i++) {
            new_elos[i] += shift;
            double d = fabs(new_elos[i] - players[i].elo);
            if (d > max_diff) max_diff = d;
            players[i].elo = new_elos[i];
        }
        
        if (max_diff < 0.001) break;
    }
}

// =============================================================================
// SINGLE GAME
// =============================================================================

typedef struct {
    int wins_a, wins_b, draws;
    MCTSStats stats_a, stats_b;
} GameResult;

static GameResult play_single_game(Player *pA, Player *pB, int a_is_white, double time_limit) {
    GameResult result = {0};
    GameState state;
    init_game(&state);
    
    Arena arena_a, arena_b;
    arena_init(&arena_a, ARENA_SIZE_TOURNAMENT);
    arena_init(&arena_b, ARENA_SIZE_TOURNAMENT);
    
    GameState history[2] = {0};
    int history_valid = 0;
    int move_count = 0;
    
    while (1) {
        arena_reset(&arena_a);
        arena_reset(&arena_b);
        
        MoveList moves;
        generate_moves(&state, &moves);
        
        // Terminal conditions
        if (moves.count == 0) {
            int winner = (state.current_player == WHITE) ? BLACK : WHITE;
            if ((winner == WHITE && a_is_white) || (winner == BLACK && !a_is_white))
                result.wins_a++;
            else
                result.wins_b++;
            break;
        }
        
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES || move_count > 200) {
            result.draws++;
            break;
        }
        
        // Determine whose turn
        int is_a_turn = (state.current_player == WHITE) == a_is_white;
        Player *current = is_a_turn ? pA : pB;
        Arena *arena = is_a_turn ? &arena_a : &arena_b;
        MCTSStats *stats = is_a_turn ? &result.stats_a : &result.stats_b;
        
        Node *root = mcts_create_root(state, arena, current->config);
        
        // History injection for CNN
        if (current->config.cnn_weights && history_valid >= 1) {
            Node *h1 = arena_alloc(arena, sizeof(Node));
            memset(h1, 0, sizeof(Node));
            pthread_mutex_init(&h1->lock, NULL);
            h1->state = history[0];
            root->parent = h1;
            if (history_valid >= 2) {
                Node *h2 = arena_alloc(arena, sizeof(Node));
                memset(h2, 0, sizeof(Node));
                pthread_mutex_init(&h2->lock, NULL);
                h2->state = history[1];
                h1->parent = h2;
            }
        }
        
        Move best = mcts_search(root, arena, time_limit, current->config, stats, NULL);
        
        // Update history
        history[1] = history[0];
        history[0] = state;
        if (history_valid < 2) history_valid++;
        
        apply_move(&state, &best);
        move_count++;
    }
    
    arena_free(&arena_a);
    arena_free(&arena_b);
    return result;
}

// =============================================================================
// MATCH (Multiple Games)
// =============================================================================

// =============================================================================
// MATCH (Multiple Games - Legacy Style Output)
// =============================================================================

static void play_match(Player *pA, Player *pB, int games, double time_limit,
                       MatchResult *out_result) {
    int wins_a = 0, wins_b = 0, draws = 0;
    long long iters_a = 0, nodes_a = 0, moves_a = 0, mem_a = 0, depth_a = 0;
    long long iters_b = 0, nodes_b = 0, moves_b = 0, mem_b = 0, depth_b = 0;
    
    #pragma omp parallel for reduction(+:wins_a, wins_b, draws, iters_a, nodes_a, moves_a, mem_a, depth_a, iters_b, nodes_b, moves_b, mem_b, depth_b)
    for (int g = 0; g < games; g++) {
        int a_is_white = (g % 2 == 0);
        GameResult r = play_single_game(pA, pB, a_is_white, time_limit);
        
        wins_a += r.wins_a;
        wins_b += r.wins_b;
        draws += r.draws;
        
        iters_a += r.stats_a.total_iterations;
        nodes_a += r.stats_a.total_nodes;
        moves_a += r.stats_a.total_moves;
        mem_a += r.stats_a.total_memory;
        depth_a += r.stats_a.total_depth;
        
        iters_b += r.stats_b.total_iterations;
        nodes_b += r.stats_b.total_nodes;
        moves_b += r.stats_b.total_moves;
        mem_b += r.stats_b.total_memory;
        depth_b += r.stats_b.total_depth;
    }
    
    // Update player stats
    pA->wins += wins_a;
    pA->losses += wins_b;
    pA->draws += draws;
    pA->points += wins_a + 0.5 * draws;
    pA->total_iters += iters_a;
    pA->tree_nodes += nodes_a;
    pA->total_moves += moves_a;
    
    pB->wins += wins_b;
    pB->losses += wins_a;
    pB->draws += draws;
    pB->points += wins_b + 0.5 * draws;
    pB->total_iters += iters_b;
    pB->tree_nodes += nodes_b;
    pB->total_moves += moves_b;
    
    out_result->wins = wins_a;
    out_result->losses = wins_b;
    out_result->draws = draws;
    
    printf("\n");
    printf("   > %-18s Stats: %s iter/move, %s nodes, %.1f KB mem, Depth %.1f\n", 
           pA->name, 
           (moves_a > 0) ? fmt_num(iters_a / moves_a) : "0",
           (moves_a > 0) ? fmt_num(nodes_a / moves_a) : "0",
           (moves_a > 0) ? (double)mem_a / moves_a / 1024.0 : 0,
           (moves_a > 0) ? (double)depth_a / moves_a : 0);
    printf("   > %-18s Stats: %s iter/move, %s nodes, %.1f KB mem, Depth %.1f\n", 
           pB->name, 
           (moves_b > 0) ? fmt_num(iters_b / moves_b) : "0", 
           (moves_b > 0) ? fmt_num(nodes_b / moves_b) : "0",
           (moves_b > 0) ? (double)mem_b / moves_b / 1024.0 : 0,
           (moves_b > 0) ? (double)depth_b / moves_b : 0);
           
    printf("Result: %s %d - %d %s (Draws: %d)\n", pA->name, wins_a, wins_b, pB->name, draws);
}

// =============================================================================
// LEADERBOARD
// =============================================================================

static int cmp_players(const void *a, const void *b) {
    const Player *pa = a, *pb = b;
    if (pa->points != pb->points) return (pb->points > pa->points) ? 1 : -1;
    return (pb->elo > pa->elo) ? 1 : -1;
}

static void print_leaderboard(Player *players, int n) {
    qsort(players, n, sizeof(Player), cmp_players);
    
    printf("\n\n");
    printf("┌──────┬────────────────────────┬────────┬──────┬──────┬──────┬──────┬────────────┬────────────┬──────────┐\n");
    printf("│ Rank │ Name                   │ Points │ Wins │ Loss │ Draw │ ELO  │ iter/Move  │ Nodes/Move │ Win Rate │\n");
    printf("├──────┼────────────────────────┼────────┼──────┼──────┼──────┼──────┼────────────┼────────────┼──────────┤\n");
    
    for (int i = 0; i < n; i++) {
        Player *p = &players[i];
        int total_games = p->wins + p->losses + p->draws;
        double win_rate = (total_games > 0) ? 100.0 * p->wins / total_games : 0.0;
        
        long long avg_iters = (p->total_moves > 0) ? p->total_iters / p->total_moves : 0;
        long long avg_nodes = (p->total_moves > 0) ? p->tree_nodes / p->total_moves : 0;

        printf("│ %-4d │ %-22s │ %-6.1f │ %-4d │ %-4d │ %-4d │ %-4.0f │ %-10s │ %-10s │ %5.1f%%   │\n", 
               i+1, p->name, p->points, p->wins, p->losses, p->draws, p->elo, 
               fmt_num(avg_iters), fmt_num(avg_nodes), win_rate);
    }
    printf("└──────┴────────────────────────┴────────┴──────┴──────┴──────┴──────┴────────────┴────────────┴──────────┘\n");
}

// =============================================================================
// PLAYER ROSTER
// =============================================================================

static int setup_players(Player *players, CNNWeights *cnn_v1, CNNWeights *cnn_v2, CNNWeights *cnn_v3, int node_limit, int loaded[3]) {
    // Load separate CNN weights for different models
    cnn_init(cnn_v1);
    cnn_init(cnn_v2);
    cnn_init(cnn_v3);
    
    // Load 3 different weight versions
    loaded[0] = (cnn_load_weights(cnn_v1, "out/models/cnn_weights_v1.bin") == 0);
    loaded[1] = (cnn_load_weights(cnn_v2, "out/models/cnn_weights_v2.bin") == 0);
    loaded[2] = (cnn_load_weights(cnn_v3, "out/models/cnn_weights_v3.bin") == 0);
    
    printf("CNN Weights: V1: %s | V2: %s | V3: %s\n", 
           loaded[0] ? "✓" : "⚠", loaded[1] ? "✓" : "⚠", loaded[2] ? "✓" : "⚠");
    
    // Get presets
    MCTSConfig cfg_pure = mcts_get_preset(MCTS_PRESET_PURE_VANILLA);
    MCTSConfig cfg_van = mcts_get_preset(MCTS_PRESET_VANILLA);
    MCTSConfig cfg_gm = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
    MCTSConfig cfg_cnn1 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    MCTSConfig cfg_cnn2 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    MCTSConfig cfg_cnn3 = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    MCTSConfig cfg_tt = mcts_get_preset(MCTS_PRESET_TT_ONLY);
    MCTSConfig cfg_solver = mcts_get_preset(MCTS_PRESET_SOLVER_ONLY);
    MCTSConfig cfg_weights = mcts_get_preset(MCTS_PRESET_WEIGHTS_ONLY);
    MCTSConfig cfg_smart = mcts_get_preset(MCTS_PRESET_SMART_ROLLOUTS);
    MCTSConfig cfg_progbias = mcts_get_preset(MCTS_PRESET_PROG_BIAS_ONLY);
    
    // Assign different weights to different models
    cfg_gm.cnn_weights = cnn_v1;    // Grandmaster uses V1
    cfg_cnn1.cnn_weights = cnn_v1;  // CNN V1
    cfg_cnn2.cnn_weights = cnn_v2;  // CNN V2
    cfg_cnn3.cnn_weights = cnn_v3;  // CNN V3
    
    // Apply node limit
    MCTSConfig *cfgs[] = {&cfg_pure, &cfg_van, &cfg_gm, &cfg_cnn1, &cfg_cnn2, &cfg_cnn3, &cfg_tt, &cfg_solver, &cfg_weights, &cfg_smart, &cfg_progbias};
    for (int i = 0; i < 11; i++) cfgs[i]->max_nodes = node_limit;
    
    // Build roster
    int n = 0;
    players[n++] = (Player){.name="PureVanilla", .config=cfg_pure, .elo=INITIAL_ELO};
    players[n++] = (Player){.name="Vanilla", .config=cfg_van, .elo=INITIAL_ELO};
    players[n++] = (Player){.name="Grandmaster", .config=cfg_gm, .elo=INITIAL_ELO};
    players[n++] = (Player){.name=loaded[0] ? "CNN (V1)" : "CNN V1 (Random)", .config=cfg_cnn1, .elo=INITIAL_ELO};
    players[n++] = (Player){.name=loaded[1] ? "CNN (V2)" : "CNN V2 (Random)", .config=cfg_cnn2, .elo=INITIAL_ELO};
    players[n++] = (Player){.name=loaded[2] ? "CNN (V3)" : "CNN V3 (Random)", .config=cfg_cnn3, .elo=INITIAL_ELO};
    players[n++] = (Player){.name="TT-Only", .config=cfg_tt, .elo=INITIAL_ELO};
    players[n++] = (Player){.name="Solver", .config=cfg_solver, .elo=INITIAL_ELO};
    players[n++] = (Player){.name="Weights", .config=cfg_weights, .elo=INITIAL_ELO};
    players[n++] = (Player){.name="SmartRollout", .config=cfg_smart, .elo=INITIAL_ELO};
    players[n++] = (Player){.name="ProgBias", .config=cfg_progbias, .elo=INITIAL_ELO};
    
    // Print roster
    printf("\n");
    printf("┌────┬──────────────────┬────────┬───────┬──────────────────────┐\n");
    printf("│ ID │ Model Name       │ Nodes  │ PUCT  │ Features             │\n");
    printf("├────┼──────────────────┼────────┼───────┼──────────────────────┤\n");
    for (int i = 0; i < n; i++) {
        Player *p = &players[i];
        char features[64] = "";
        if (p->config.cnn_weights) strcat(features, "CNN ");
        if (p->config.use_tt) strcat(features, "TT ");
        if (p->config.use_solver) strcat(features, "Solver ");
        
        printf("│ %-2d │ %-16s │ %-6d │ %-5.2f │ %-20s │\n", 
               i+1, p->name, p->config.max_nodes, 
               p->config.use_puct ? p->config.puct_c : 0.0f,
               features);
    }
    printf("└────┴──────────────────┴────────┴───────┴──────────────────────┘\n");
    printf("\n");
    
    return n;
}

// =============================================================================
// CMD_TOURNAMENT - Entry point for unified CLI
// =============================================================================

int cmd_tournament(int argc, char **argv) {
    int games = DEFAULT_GAMES_PER_PAIRING;
    double time_limit = DEFAULT_TIME_LIMIT;
    int node_limit = 1000;
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-g") == 0 && i+1 < argc) games = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) time_limit = atof(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) node_limit = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: dama tournament [-g games] [-n nodes] [-t time]\n\n");
            printf("Options:\n");
            printf("  -g N   Games per pairing (default: %d)\n", DEFAULT_GAMES_PER_PAIRING);
            printf("  -n N   Node limit per move (default: 1000)\n");
            printf("  -t T   Time limit in seconds (default: %.1f)\n", DEFAULT_TIME_LIMIT);
            return 0;
        }
    }
    
    // if (node_limit > 0) time_limit = 600.0;  // Large fallback when using nodes
    
    printf("=== MCTS Tournament ===\n");
    printf("Games: %d | Nodes: %d | Time: %.1fs\n\n", games, node_limit, time_limit);
    
    srand(time(NULL));
    zobrist_init();
    init_move_tables();
    
    // Setup players with 3 weight versions
    CNNWeights cnn_v1, cnn_v2, cnn_v3;
    Player players[MAX_PLAYERS] = {0};
    int loaded[3];
    int n = setup_players(players, &cnn_v1, &cnn_v2, &cnn_v3, node_limit, loaded);
    
    // Match results matrix (Static to avoid stack overflow/corruption)
    static MatchResult results[MAX_PLAYERS][MAX_PLAYERS];
    memset(results, 0, sizeof(results));
    
    // Round-robin
    clock_t start = clock();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            printf("\n[%s vs %s]\n", players[i].name, players[j].name);
            play_match(&players[i], &players[j], games, time_limit, &results[i][j]);
            
            // Mirror results
            results[j][i].wins = results[i][j].losses;
            results[j][i].losses = results[i][j].wins;
            results[j][i].draws = results[i][j].draws;
        }
    }
    
    // Calculate ELO and print
    calculate_elo(players, n, results);
    print_leaderboard(players, n);
    
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("\nTotal time: %.1fs\n", elapsed);
    
    cnn_free(&cnn_v1);
    cnn_free(&cnn_v2);
    cnn_free(&cnn_v3);
    return 0;
}
