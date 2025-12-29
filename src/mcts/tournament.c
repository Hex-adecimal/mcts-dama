/**
 * tournament.c - Tournament Logic
 */

#include "tournament.h"
#include "core/game.h"
#include "core/movegen.h"
#include "mcts.h" // mcts_create_root etc
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// ELO CALCULATION
// =============================================================================

#define INITIAL_ELO 1200.0

typedef struct {
    int wins, losses, draws;
} MatchPairResult;

static void compute_elos(TournamentPlayer *players, int n, MatchPairResult *results) {
    // Reset ELOs to initial if needed? Or refine?
    // Assuming starting from INITIAL. 
    // Actually, preserving prior ELOs is good for ongoing tourneys.
    // But usually reset 1200.
    for(int i=0; i<n; i++) players[i].elo = INITIAL_ELO;

    for (int iter = 0; iter < 10000; iter++) {
        double new_elos[100]; // Assuming max 100 players
        double max_diff = 0.0;
        
        for (int i = 0; i < n; i++) {
            double actual = 0, expected = 0;
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                MatchPairResult res = results[i * n + j];
                int games = res.wins + res.losses + res.draws;
                if (games == 0) continue;
                
                actual += res.wins + 0.5 * res.draws;
                double diff = fmin(fmax(players[j].elo - players[i].elo, -800), 800);
                expected += games / (1.0 + pow(10.0, diff / 400.0));
            }
            new_elos[i] = players[i].elo + (actual - expected) * 0.1;
        }
        
        // Recenter
        double sum = 0;
        for(int i=0; i<n; i++) sum += new_elos[i];
        double shift = INITIAL_ELO - sum/n;
        for(int i=0; i<n; i++) {
            new_elos[i] += shift;
            double d = fabs(new_elos[i] - players[i].elo);
            if(d > max_diff) max_diff = d;
            players[i].elo = new_elos[i];
        }
        if (max_diff < 0.001) break;
    }
}

// =============================================================================
// GAME LOGIC
// =============================================================================

static int play_single_game(TournamentPlayer *pA, TournamentPlayer *pB, int a_is_white, double time_limit, 
                            MCTSStats *sA, MCTSStats *sB, int *out_game_moves, double *out_durA, double *out_durB) {
    GameState state;
    init_game(&state);
    
    Arena arenaA, arenaB;
    arena_init(&arenaA, 50*1024*1024);
    arena_init(&arenaB, 50*1024*1024);
    
    GameState history[2] = {0};
    int moves = 0;
    
    // Result: 1 (A wins), -1 (B wins), 0 (Draw)
    int result = 0;
    
    double durA = 0, durB = 0;
    
    while (1) {
        MoveList list;
        generate_moves(&state, &list);
        if (list.count == 0) {
            int winner = (state.current_player == WHITE) ? BLACK : WHITE;
            if ((winner == WHITE && a_is_white) || (winner == BLACK && !a_is_white)) result = 1;
            else result = -1;
            break;
        }
        
        if (state.moves_without_captures >= 40 || moves > 200) {
            result = 0;
            break;
        }
        
        int is_a_turn = (state.current_player == WHITE) == a_is_white;
        TournamentPlayer *cur = is_a_turn ? pA : pB;
        Arena *arena = is_a_turn ? &arenaA : &arenaB;
        MCTSStats *stats = is_a_turn ? sA : sB;
        
        // Search
        Node *root = mcts_create_root(state, arena, cur->config);
        
        // History injection
        // MCTS root creation doesn't add parent history automatically?
        // In cmd_tournament it was manually added.
        if (cur->config.cnn_weights && moves >= 1) {
             // Add logic here if needed. 
             // mcts_create_root just sets state.
             // If cnn uses parent pointer for history, we need to mock it.
             // For simplicity in this logic, we assume mcts engine handles it 
             // OR we do minimal mock:
             Node *h1 = arena_alloc(arena, sizeof(Node));
             memset(h1, 0, sizeof(Node));
             h1->state = history[0]; // Previous state
             root->parent = h1;
             if (moves >= 2) {
                 Node *h2 = arena_alloc(arena, sizeof(Node));
                 memset(h2, 0, sizeof(Node));
                 h2->state = history[1];
                 h1->parent = h2;
             }
        }
        
        // Reset arena for other player? No, arena reset happens at start of YOUR turn search.
        // Actually mcts_search resets arena usually? No, caller resets.
        arena_reset(arena); // Reset before search to clear previous tree
        // Re-create root after reset? Yes!
        root = mcts_create_root(state, arena, cur->config);
        // Add history again... repeated code logic. 
        if (cur->config.cnn_weights && moves >= 1) {
             Node *h1 = arena_alloc(arena, sizeof(Node));
             memset(h1, 0, sizeof(Node));
             h1->state = history[0];
             root->parent = h1;
             if (moves >= 2) {
                 Node *h2 = arena_alloc(arena, sizeof(Node));
                 memset(h2, 0, sizeof(Node));
                 h2->state = history[1];
                 h1->parent = h2;
             }
        }
        
        double t0;
        #ifdef _OPENMP
        t0 = omp_get_wtime();
        #else
        t0 = (double)clock() / CLOCKS_PER_SEC;
        #endif
        
        Move best = mcts_search(root, arena, time_limit, cur->config, stats, NULL);
        
        double t1;
        #ifdef _OPENMP
        t1 = omp_get_wtime();
        #else
        t1 = (double)clock() / CLOCKS_PER_SEC;
        #endif
        
        if (is_a_turn) durA += (t1 - t0);
        else durB += (t1 - t0);
        
        history[1] = history[0];
        history[0] = state;
        
        apply_move(&state, &best);
        moves++;
    }
    
    arena_free(&arenaA);
    arena_free(&arenaB);
    
    if (out_game_moves) *out_game_moves = moves;
    if (out_durA) *out_durA = durA;
    if (out_durB) *out_durB = durB;
    return result;
}

// =============================================================================
// MAIN RUNNER
// =============================================================================

void tournament_run(TournamentSystemConfig *cfg) {
    int n = cfg->num_players;
    int games = cfg->games_per_pair;
    
    // Matrix for ELO
    MatchPairResult *matrix = calloc(n * n, sizeof(MatchPairResult));
    
    // Total matches: n*(n-1)/2
    int total_matches = n * (n - 1) / 2;
    if (cfg->on_start) cfg->on_start(total_matches);
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (cfg->on_match_start) cfg->on_match_start(i, j, cfg->players[i].name, cfg->players[j].name);
            
            MatchPairResult pair_res = {0};
            
            // Run games
            int p1_wins = 0, p2_wins = 0, draws = 0;
            
            // OMP Parallel Games?
            int use_omp = (cfg->parallel_games);
            #ifdef _OPENMP
            // Need thread-safe stats aggregation
            #else
            use_omp = 0;
            #endif
            
            #pragma omp parallel for if(use_omp) reduction(+:p1_wins, p2_wins, draws)
            for (int g = 0; g < games; g++) {
                int a_is_white = (g % 2 == 0);
                MCTSStats s1 = {0}, s2 = {0};
                int moves_count = 0;
                double durA = 0, durB = 0;
                int res = play_single_game(&cfg->players[i], &cfg->players[j], a_is_white, cfg->time_limit, &s1, &s2, &moves_count, &durA, &durB);
                
                if (res == 1) p1_wins++;
                else if (res == -1) p2_wins++;
                else draws++;
                
                // Aggregate stats to P1 (i)
                #pragma omp atomic
                cfg->players[i].total_iters += s1.total_iterations;
                #pragma omp atomic
                cfg->players[i].total_nodes += s1.total_nodes;
                #pragma omp atomic
                cfg->players[i].total_moves += s1.total_moves;

                // Aggregate stats to P2 (j)
                #pragma omp atomic
                cfg->players[j].total_iters += s2.total_iterations;
                #pragma omp atomic
                cfg->players[j].total_nodes += s2.total_nodes;
                #pragma omp atomic
                cfg->players[j].total_moves += s2.total_moves;
                
                if (cfg->on_game_complete) {
                    TournamentGameResult gr = {
                        .p1_idx=i, .p2_idx=j, .result=res, .moves=moves_count, .duration=durA+durB, .duration_p1=durA, .duration_p2=durB, .s1=s1, .s2=s2
                    };
                    #pragma omp critical
                    cfg->on_game_complete(&gr);
                }
            }
            
            pair_res.wins = p1_wins;
            pair_res.losses = p2_wins;
            pair_res.draws = draws;
            
            // Store symmetry
            matrix[i*n + j] = pair_res;
            matrix[j*n + i] = (MatchPairResult){ .wins=p2_wins, .losses=p1_wins, .draws=draws };
            
            // Update player cumulative
            cfg->players[i].wins += p1_wins;
            cfg->players[i].losses += p2_wins;
            cfg->players[i].draws += draws;
            cfg->players[i].points += p1_wins + 0.5*draws;
            
            cfg->players[j].wins += p2_wins;
            cfg->players[j].losses += p1_wins;
            cfg->players[j].draws += draws;
            cfg->players[j].points += p2_wins + 0.5*draws;
            
            if (cfg->on_match_end) cfg->on_match_end(i, j, p1_wins, p2_wins, draws);
        }
    }
    
    compute_elos(cfg->players, n, matrix);
    
    if (cfg->on_tournament_end) cfg->on_tournament_end(cfg->players, n);
    
    free(matrix);
}

void tournament_calculate_elo(TournamentPlayer *players, int n, int **results_matrix) {
    (void)players; (void)n; (void)results_matrix;
}
