#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "game.h"
#include "movegen.h"
#include "mcts.h"
#include "cnn.h"
#include "params.h"

#define DEFAULT_GAMES_COMPARE   10
#define DEFAULT_TIME_COMPARE    TIME_LOW

// Helper to format numbers with thousand separators (e.g. 1.000.000)
const char* format_thousands(long long n) {
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
            buf[out_idx++] = '.';
        }
        buf[out_idx++] = temp[i];
    }
    buf[out_idx] = '\0';
    return buf;
}

// =============================================================================
// 1v1 COMPARISON BENCHMARK
// =============================================================================

int main(int argc, char *argv[]) {
    int games = DEFAULT_GAMES_COMPARE;
    double time_limit = DEFAULT_TIME_COMPARE;
    int node_limit = 0;
    
    int opt;
    while ((opt = getopt(argc, argv, "g:t:n:h")) != -1) {
        switch (opt) {
            case 'g': games = atoi(optarg); break;
            case 't': time_limit = atof(optarg); break;
            case 'n': node_limit = atoi(optarg); break;
            case 'h':
            default:
                printf("Usage: %s [-g games] [-t time_limit] [-n node_limit]\n", argv[0]);
                printf("  -g: Games to play (default: %d)\n", DEFAULT_GAMES_COMPARE);
                printf("  -t: Time limit per move in seconds (default: %.2fs)\n", DEFAULT_TIME_COMPARE);
                printf("  -n: Node limit per move (0 = use time limit, default: %d)\n", 0);
                return 0;
        }
    }

    zobrist_init();
    init_move_tables();
    srand(time(NULL));
    
    printf("┌──────────────────────────────────────────────────────────┐\n");
    printf("│ 1v1 COMPARISON: AlphaZero (CNN) vs Vanilla                   │\n");
    printf("└──────────────────────────────────────────────────────────┘\n");
    
    if (node_limit > 0) {
        printf("Mode: Intelligence (Nodes: %d)\n", node_limit);
    } else {
        printf("Mode: Performance (Time: %.2fs)\n", time_limit);
    }
    printf("Games: %d\n\n", games);

    // --- WEIGHT LOADING ---
    CNNWeights cnn_weights;
    cnn_init(&cnn_weights);
    const char *weight_paths[] = {
        "bin/cnn_weights.bin",
        "cnn_weights.bin",
        "models/cnn_weights.bin",
        "bin/cnn_weights_final.bin"
    };
    int weights_loaded = 0;
    for (int i = 0; i < 4; i++) {
        if (cnn_load_weights(&cnn_weights, weight_paths[i]) == 0) {
            printf("✓ Loaded CNN weights from: %s (TRAINED)\n", weight_paths[i]);
            weights_loaded = 1;
            break;
        }
    }
    if (!weights_loaded) {
        printf("⚠️  CNN using random weights (BRAINLESS)\n");
    }

    MCTSConfig cfg_vanilla = mcts_get_preset(MCTS_PRESET_VANILLA);
    MCTSConfig cfg_cnn = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    if (weights_loaded) cfg_cnn.cnn_weights = &cnn_weights;
    
    cfg_vanilla.max_nodes = node_limit;
    cfg_cnn.max_nodes = node_limit;

    int wins_gm = 0, wins_vanilla = 0, draws = 0;
    long long total_nodes = 0, total_moves = 0;
    
    double start_perf = (double)clock();
    
    #pragma omp parallel for reduction(+:wins_gm, wins_vanilla, draws, total_nodes, total_moves) schedule(dynamic)
    for (int game = 0; game < games; game++) {
        Arena arena_A, arena_B;
        arena_init(&arena_A, ARENA_SIZE_TOURNAMENT);
        arena_init(&arena_B, ARENA_SIZE_TOURNAMENT);
        
        GameState state;
        init_game(&state);
        
        int turn = 0;
        int gm_is_white = (game % 2 == 0);
        
        while (1) {
            arena_reset(&arena_A);
            arena_reset(&arena_B);
            
            MoveList list;
            generate_moves(&state, &list);
            
            if (list.count == 0) {
                int winner = (state.current_player == WHITE) ? BLACK : WHITE;
                if ((winner == WHITE && gm_is_white) || (winner == BLACK && !gm_is_white)) {
                    wins_gm++;
                } else {
                    wins_vanilla++;
                }
                break;
            }
            
            if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES || turn > MAX_GAME_TURNS) {
                draws++;
                break;
            }
            
            MCTSConfig *cfg;
            Arena *arena;
            const char *pname;
            
            if ((state.current_player == WHITE && gm_is_white) || 
                (state.current_player == BLACK && !gm_is_white)) {
                cfg = &cfg_cnn;
                arena = &arena_A;
                pname = "AlphaZero";
            } else {
                cfg = &cfg_vanilla;
                arena = &arena_B;
                pname = "Vanilla";
            }
            
            MCTSStats stats = {0};
            Node *root = mcts_create_root(state, arena, *cfg);
            Move best = mcts_search(root, arena, time_limit, *cfg, &stats, &root);
            
            total_nodes += stats.total_iterations;
            total_moves++;

            if (game == 0) {
                #pragma omp critical
                {
                    double ucb = mcts_get_avg_root_ucb(root, *cfg);
                    printf("\r[G0] Turn %-3d | %-12s | Nodes: %-7s | Avg UCB: %5.2f  ", 
                           turn, pname, format_thousands(stats.total_iterations), ucb);
                    fflush(stdout);
                }
            }

            apply_move(&state, &best);
            turn++;
        }
        
        arena_free(&arena_A);
        arena_free(&arena_B);
        
        // Progress indicator
        if ((game + 1) % (games / 10 > 0 ? games / 10 : 1) == 0) {
            #pragma omp critical
            printf("\n  Completed %d/%d games...\n", game + 1, games);
        }
    }
    
    double total_time = ((double)clock() - start_perf) / CLOCKS_PER_SEC;
    
    printf("\n┌─────────────────────────────┬────────┬──────────┐\n");
    printf("│ Player                      │ Wins   │ Rate     │\n");
    printf("├─────────────────────────────┼────────┼──────────┤\n");
    printf("│ AlphaZero   %-15s │ %-6d │ %5.1f%%   │\n", weights_loaded ? "(TRAINED)" : "(BRAINLESS)", wins_gm, 100.0 * wins_gm / games);
    printf("│ Vanilla                     │ %-6d │ %5.1f%%   │\n", wins_vanilla, 100.0 * wins_vanilla / games);
    printf("│ Draws                       │ %-6d │ %5.1f%%   │\n", draws, 100.0 * draws / games);
    printf("├─────────────────────────────┴────────┴──────────┤\n");
    printf("│ Avg Nodes/Move: %-31s │\n", format_thousands(total_moves > 0 ? total_nodes / total_moves : 0));
    printf("└─────────────────────────────────────────────────┘\n");
    printf("Total Wall Time: %.2f seconds\n", total_time);
    
    cnn_free(&cnn_weights);
    return 0;
}
