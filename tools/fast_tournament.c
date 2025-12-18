#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "game.h"
#include "mcts.h"
#include "params.h"

// =============================================================================
// FAST 1v1 TOURNAMENT vs VANILLA
// =============================================================================
// Quick benchmark: Tests Grandmaster (TT + Solver) against Vanilla baseline.

int main() {
    zobrist_init();
    init_move_tables();
    srand(time(NULL));
    
    printf("=== FAST 1v1 TOURNAMENT: Grandmaster vs Vanilla ===\n");
    printf("Games: %d | Time per move: %.2fs\n\n", GAMES_FAST, TIME_TOURNAMENT);

#ifdef _OPENMP
    printf("Parallel Execution: %d Threads\n\n", omp_get_max_threads());
#endif

    // Vanilla (Baseline: Pure Random Playouts)
    MCTSConfig cfg_vanilla = { 
        .ucb1_c = UCB1_C, 
        .rollout_epsilon = ROLLOUT_EPSILON_RANDOM, 
        .draw_score = DRAW_SCORE, 
        .expansion_threshold = EXPANSION_THRESHOLD,
        .use_tree_reuse = 0, 
        .use_ucb1_tuned = 0, 
        .use_tt = 0, 
        .use_solver = 0, 
        .use_progressive_bias = 0,
        .weights = { W_CAPTURE, W_PROMOTION, W_ADVANCE, W_CENTER, W_EDGE, W_BASE, W_THREAT, W_LADY_ACTIVITY }
    };

    // Grandmaster (TT + Solver)
    MCTSConfig cfg_grandmaster = cfg_vanilla;
    cfg_grandmaster.use_tt = 1;
    cfg_grandmaster.use_solver = 1;
    cfg_grandmaster.use_ucb1_tuned = 1;
    cfg_grandmaster.rollout_epsilon = ROLLOUT_EPSILON_RANDOM;

    int wins_gm = 0, wins_vanilla = 0, draws = 0;
    
    double start_time = (double)clock();
    
    #pragma omp parallel for reduction(+:wins_gm, wins_vanilla, draws) schedule(dynamic)
    for (int game = 0; game < GAMES_FAST; game++) {
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
            
            if ((state.current_player == WHITE && gm_is_white) || 
                (state.current_player == BLACK && !gm_is_white)) {
                cfg = &cfg_grandmaster;
                arena = &arena_A;
            } else {
                cfg = &cfg_vanilla;
                arena = &arena_B;
            }
            
            Node *root = mcts_create_root(state, arena, *cfg);
            Move best = mcts_search(root, arena, TIME_TOURNAMENT, *cfg, NULL, NULL);
            apply_move(&state, &best);
            turn++;
        }
        
        arena_free(&arena_A);
        arena_free(&arena_B);
        
        // Progress indicator
        if ((game + 1) % 10 == 0) {
            printf("  Completed %d/%d games...\n", game + 1, GAMES_FAST);
        }
    }
    
    double total_time = ((double)clock() - start_time) / CLOCKS_PER_SEC;
    
    printf("\n========================================\n");
    printf("           RESULTS                     \n");
    printf("========================================\n");
    printf("Grandmaster: %d wins (%.1f%%)\n", wins_gm, 100.0 * wins_gm / GAMES_FAST);
    printf("Vanilla:     %d wins (%.1f%%)\n", wins_vanilla, 100.0 * wins_vanilla / GAMES_FAST);
    printf("Draws:       %d (%.1f%%)\n", draws, 100.0 * draws / GAMES_FAST);
    printf("========================================\n");
    printf("Total Time: %.2f seconds\n", total_time);
    
    return 0;
}
