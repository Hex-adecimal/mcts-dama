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
#define DEFAULT_TIME_LIMIT          0.5
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

static void play_match(Player *pA, Player *pB, int games, double time_limit,
                       MatchResult *out_result) {
    int wins_a = 0, wins_b = 0, draws = 0;
    long long iters_a = 0, nodes_a = 0, moves_a = 0;
    long long iters_b = 0, nodes_b = 0, moves_b = 0;
    
    #pragma omp parallel for reduction(+:wins_a, wins_b, draws, iters_a, nodes_a, moves_a, iters_b, nodes_b, moves_b)
    for (int g = 0; g < games; g++) {
        int a_is_white = (g % 2 == 0);
        GameResult r = play_single_game(pA, pB, a_is_white, time_limit);
        
        wins_a += r.wins_a;
        wins_b += r.wins_b;
        draws += r.draws;
        
        iters_a += r.stats_a.total_iterations;
        nodes_a += r.stats_a.total_nodes;
        moves_a += r.stats_a.total_moves;
        
        iters_b += r.stats_b.total_iterations;
        nodes_b += r.stats_b.total_nodes;
        moves_b += r.stats_b.total_moves;
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
    
    printf("   %s %d-%d %s (D:%d) | %s: %s iter/mv | %s: %s iter/mv\n",
           pA->name, wins_a, wins_b, pB->name, draws,
           pA->name, moves_a ? fmt_num(iters_a / moves_a) : "0",
           pB->name, moves_b ? fmt_num(iters_b / moves_b) : "0");
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
    
    printf("\n");
    printf("┌──────┬────────────────────────┬────────┬──────┬──────┬──────┬──────┬────────────┐\n");
    printf("│ Rank │ Name                   │ Points │ Wins │ Loss │ Draw │  ELO │ Nodes/Move │\n");
    printf("├──────┼────────────────────────┼────────┼──────┼──────┼──────┼──────┼────────────┤\n");
    
    for (int i = 0; i < n; i++) {
        Player *p = &players[i];
        long long avg_nodes = p->total_moves > 0 ? p->tree_nodes / p->total_moves : 0;
        printf("│ %-4d │ %-22s │ %6.1f │ %4d │ %4d │ %4d │ %4.0f │ %10s │\n",
               i+1, p->name, p->points, p->wins, p->losses, p->draws, p->elo, fmt_num(avg_nodes));
    }
    printf("└──────┴────────────────────────┴────────┴──────┴──────┴──────┴──────┴────────────┘\n");
}

// =============================================================================
// PLAYER ROSTER
// =============================================================================

static int setup_players(Player *players, CNNWeights *cnn, int node_limit, int *weights_loaded) {
    // Load CNN weights
    cnn_init(cnn);
    *weights_loaded = (cnn_load_weights(cnn, "models/cnn_weights.bin") == 0 ||
                       cnn_load_weights(cnn, "models/cnn_weights_final.bin") == 0);
    
    if (*weights_loaded) printf("✓ CNN weights loaded\n");
    else printf("⚠ CNN using random weights\n");
    
    // Get presets
    MCTSConfig cfg_pure = mcts_get_preset(MCTS_PRESET_PURE_VANILLA);
    MCTSConfig cfg_van = mcts_get_preset(MCTS_PRESET_VANILLA);
    MCTSConfig cfg_gm = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
    MCTSConfig cfg_cnn = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    MCTSConfig cfg_tt = mcts_get_preset(MCTS_PRESET_TT_ONLY);
    MCTSConfig cfg_solver = mcts_get_preset(MCTS_PRESET_SOLVER_ONLY);
    MCTSConfig cfg_weights = mcts_get_preset(MCTS_PRESET_WEIGHTS_ONLY);
    MCTSConfig cfg_smart = mcts_get_preset(MCTS_PRESET_SMART_ROLLOUTS);
    
    cfg_cnn.cnn_weights = cnn;
    cfg_gm.cnn_weights = cnn;
    
    // Apply node limit
    MCTSConfig *cfgs[] = {&cfg_pure, &cfg_van, &cfg_gm, &cfg_cnn, &cfg_tt, &cfg_solver, &cfg_weights, &cfg_smart};
    for (int i = 0; i < 8; i++) cfgs[i]->max_nodes = node_limit;
    
    // Build roster
    int n = 0;
    players[n++] = (Player){.name = "PureVanilla", .config = cfg_pure, .elo = INITIAL_ELO};
    players[n++] = (Player){.name = "Vanilla", .config = cfg_van, .elo = INITIAL_ELO};
    players[n++] = (Player){.name = "Grandmaster", .config = cfg_gm, .elo = INITIAL_ELO};
    players[n++] = (Player){.name = *weights_loaded ? "CNN (Trained)" : "CNN (Random)", .config = cfg_cnn, .elo = INITIAL_ELO};
    players[n++] = (Player){.name = "TT-Only", .config = cfg_tt, .elo = INITIAL_ELO};
    players[n++] = (Player){.name = "Solver", .config = cfg_solver, .elo = INITIAL_ELO};
    players[n++] = (Player){.name = "Weights", .config = cfg_weights, .elo = INITIAL_ELO};
    players[n++] = (Player){.name = "SmartRollout", .config = cfg_smart, .elo = INITIAL_ELO};
    
    return n;
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char *argv[]) {
    int games = DEFAULT_GAMES_PER_PAIRING;
    double time_limit = DEFAULT_TIME_LIMIT;
    int node_limit = 1000;
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-g") == 0 && i+1 < argc) games = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) time_limit = atof(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) node_limit = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: tournament [-g games] [-n nodes] [-t time]\n");
            printf("  -g N   Games per pairing (default: %d)\n", DEFAULT_GAMES_PER_PAIRING);
            printf("  -n N   Node limit per move (default: 1000)\n");
            printf("  -t T   Time limit in seconds (default: %.1f)\n", DEFAULT_TIME_LIMIT);
            return 0;
        }
    }
    
    if (node_limit > 0) time_limit = 600.0;  // Large fallback when using nodes
    
    printf("=== MCTS Tournament ===\n");
    printf("Games: %d | Nodes: %d | Time: %.1fs\n\n", games, node_limit, time_limit);
    
    srand(time(NULL));
    zobrist_init();
    init_move_tables();
    
    // Setup players
    CNNWeights cnn;
    Player players[MAX_PLAYERS] = {0};
    int weights_loaded;
    int n = setup_players(players, &cnn, node_limit, &weights_loaded);
    
    // Match results matrix
    MatchResult results[MAX_PLAYERS][MAX_PLAYERS] = {0};
    
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
    
    cnn_free(&cnn);
    return 0;
}
