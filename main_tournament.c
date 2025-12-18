#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "game.h"
#include "mcts.h"
#include "params.h"

// ================================================================================================
//  ELO RATING SYSTEM
// ================================================================================================

#define INITIAL_ELO 1500.0
#define K_FACTOR 32.0

/**
 * Calculates expected score for player A vs player B.
 * @param elo_a ELO rating of player A.
 * @param elo_b ELO rating of player B.
 * @return Expected score (0.0 to 1.0).
 */
double expected_score(double elo_a, double elo_b) {
    return 1.0 / (1.0 + pow(10.0, (elo_b - elo_a) / 400.0));
}

// ================================================================================================
//  TOURNAMENT LOGIC
// ================================================================================================

/**
 * Plays a single match between two configurations.
 * @param config_white Configuration for White player.
 * @param config_black Configuration for Black player.
 * @param arena Pointer to the arena allocator.
 * @param stats_white Statistics tracker for White player.
 * @param stats_black Statistics tracker for Black player.
 * @return Winner color (WHITE, BLACK, or -1 for draw).
 */
int play_match(MCTSConfig *config_white, MCTSConfig *config_black, Arena *arena,
               MCTSStats *stats_white, MCTSStats *stats_black) {
    GameState state;
    init_game(&state);
    
    int turn_count = 0;
    int max_turns = 150; // Prevent infinite games
    
    // Persistent roots for tree reuse
    Node *root_white = NULL;
    Node *root_black = NULL;
    Move last_move = {0}; // Track last move for tree reuse
    
    while (1) {
        // Check for game over
        MoveList list;
        generate_moves(&state, &list);
        
        if (list.count == 0) {
            return (state.current_player == WHITE) ? BLACK : WHITE;
        }
        
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            return -1; // Draw
        }
        
        if (turn_count >= max_turns) {
            return -1; // Draw (timeout)
        }
        
        // Select config and stats based on current player
        MCTSConfig *config = (state.current_player == WHITE) ? config_white : config_black;
        MCTSStats *stats = (state.current_player == WHITE) ? stats_white : stats_black;
        Node **my_root = (state.current_player == WHITE) ? &root_white : &root_black;
        Node **opponent_root = (state.current_player == WHITE) ? &root_black : &root_white;
        
        // Reset arena if too full (>80%)
        if (arena->offset > arena->size * 0.8) {
            arena_reset(arena);
            root_white = NULL;
            root_black = NULL;
            *my_root = NULL;
        }
        
        // MCTS search
        Node *root;
        
        // Attempt tree reuse: find opponent's move in our tree
        if (config->use_tree_reuse && *my_root != NULL && turn_count > 0) {
            Node *reused = find_child_by_move(*my_root, &last_move);
            if (reused) {
                root = reused; // Reuse opponent's subtree
            } else {
                // Opponent's move not in our tree, create new
                root = mcts_create_root(state, arena, *config);
            }
        } else {
            // First move or tree reuse disabled
            root = mcts_create_root(state, arena, *config);
        }
        
        double time_limit = (state.current_player == WHITE) ? TIME_WHITE : TIME_BLACK;
        
        Node *new_root = NULL;
        Move chosen_move = mcts_search(root, arena, time_limit, *config, stats, &new_root);
        
        // Save chosen move for opponent's tree reuse
        last_move = chosen_move;
        
        // Update opponent's root to our chosen child (for their next turn)
        if (config->use_tree_reuse && new_root) {
            *opponent_root = new_root;
        }
        
        // Apply move
        apply_move(&state, &chosen_move);
        turn_count++;
    }
}

// ================================================================================================
//  ROUND ROBIN TOURNAMENT
// ================================================================================================

typedef struct {
    char name[32];
    MCTSConfig config;
    double elo;
    int wins;
    int losses;
    int draws;
    double points; // 1.0 per win, 0.5 per draw
} TournamentPlayer;

// Comparison for Sorting (Desc by Points, then ELO)
int compare_players(const void *a, const void *b) {
    TournamentPlayer *pA = (TournamentPlayer *)a;
    TournamentPlayer *pB = (TournamentPlayer *)b;
    if (pA->points > pB->points) return -1;
    if (pA->points < pB->points) return 1;
    if (pA->elo > pB->elo) return -1;
    if (pA->elo < pB->elo) return 1;
    return 0;
}

int main() {
    zobrist_init(); // Init Zobrist (even if unused by some players)
    srand(time(NULL));
    
    printf("=== MCTS ROUND ROBIN TOURNAMENT ===\n");
    printf("Type: All-vs-All\n");
    
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    printf("Parallel Execution Enabled: %d Threads\n", max_threads);
#endif

    // --- CONFIGURATIONS ---
    
    // 1. Vanilla (Baseline: Pure Random Playouts)
    MCTSConfig cfg_vanilla = { .ucb1_c = UCB1_C, .rollout_epsilon = 1.0, .draw_score = DRAW_SCORE, .expansion_threshold = EXPANSION_THRESHOLD, 
                               .use_tree_reuse = 0, .use_ucb1_tuned = 0, .use_tt = 0, .use_solver = 0, .use_progressive_bias = 0,
                               .weights = { 10.0, 5.0, 0.5, 3.0, 2.0, 2.0 } };

    // 2. Tuned (UCB1-Tuned)
    MCTSConfig cfg_tuned = cfg_vanilla;
    cfg_tuned.use_ucb1_tuned = 1;
    cfg_tuned.rollout_epsilon = 0.1; // Smarter playouts

    // 3. TT (Transposition Table Only)
    MCTSConfig cfg_tt = cfg_vanilla;
    cfg_tt.use_tt = 1;

    // 4. Solver (Solver Only - No TT as requested)
    MCTSConfig cfg_solver = cfg_vanilla;
    cfg_solver.use_solver = 1;

    // 5. Bias (Progressive Bias Only - No TT as requested)
    // Bias implies using heuristics, so we also enable smart playouts
    MCTSConfig cfg_bias = cfg_vanilla;
    cfg_bias.use_progressive_bias = 1;
    cfg_bias.bias_constant = 3.0;
    cfg_bias.rollout_epsilon = 0.1;

    // 6. Ultimate (All Best Features Combined)
    MCTSConfig cfg_ultimate = cfg_vanilla;
    cfg_ultimate.use_tt = 1;
    cfg_ultimate.use_solver = 1;
    cfg_ultimate.use_progressive_bias = 1;
    cfg_ultimate.bias_constant = 3.0;
    cfg_ultimate.rollout_epsilon = 0.1;

    // 7. SPSA Tuned (Best Weights found)
    MCTSConfig cfg_spsa = cfg_ultimate; // Inherit features from Ultimate
    cfg_spsa.weights.w_capture = 10.1878;
    cfg_spsa.weights.w_promotion = 4.9642;
    cfg_spsa.weights.w_advance = 0.0000;
    cfg_spsa.weights.w_center = 3.1170;
    cfg_spsa.weights.w_edge = 1.9339;
    cfg_spsa.weights.w_base = 3.0680;

    // 8. Grandmaster (SPSA Weights + UCB1-Tuned + All Features)
    MCTSConfig cfg_grandmaster = cfg_spsa;
    cfg_grandmaster.use_ucb1_tuned = 1;

    // --- PLAYER ROSTER ---
    TournamentPlayer players[] = {
        { "Vanilla",  cfg_vanilla,  1200.0, 0, 0, 0, 0.0 },
        { "Tuned",    cfg_tuned,    1200.0, 0, 0, 0, 0.0 },
        { "TT",       cfg_tt,       1200.0, 0, 0, 0, 0.0 },
        { "Solver",   cfg_solver,   1200.0, 0, 0, 0, 0.0 },
        { "Bias",     cfg_bias,     1200.0, 0, 0, 0, 0.0 },
        { "Ultimate", cfg_ultimate, 1200.0, 0, 0, 0, 0.0 },
        { "SPSA",     cfg_spsa,     1200.0, 0, 0, 0, 0.0 },
        { "Grandmaster", cfg_grandmaster, 1200.0, 0, 0, 0, 0.0 }
    };
    int num_players = 8;
    
    int games_per_pairing = 30;
    #define TIME_PER_MOVE 0.5
    
    double start_time = (double)clock();
    
    // --- TOURNAMENT LOOP ---
    for (int i = 0; i < num_players; i++) {
        for (int j = i + 1; j < num_players; j++) {
            TournamentPlayer *pA = &players[i];
            TournamentPlayer *pB = &players[j];
            
            printf("\n--- Match: %s vs %s (%d games) ---\n", pA->name, pB->name, games_per_pairing);
            
            int wins_A = 0;
            int wins_B = 0;
            int draws = 0;
            
            long long total_nodes_A = 0, total_depth_A = 0, total_moves_A = 0;
            long long total_nodes_B = 0, total_depth_B = 0, total_moves_B = 0;
            
            #pragma omp parallel for reduction(+:wins_A, wins_B, draws, total_nodes_A, total_depth_A, total_moves_A, total_nodes_B, total_depth_B, total_moves_B) schedule(dynamic)
            for (int game = 0; game < games_per_pairing; game++) {
                GameState state;
                init_game(&state);
                
                // Stats per game (accumulated locally)
                MCTSStats stats_A = {0};
                MCTSStats stats_B = {0};
                
                int move_count = 0;
                
                // Arena per thread
                Arena arena_A, arena_B;
                arena_init(&arena_A, 1512 * 1024 * 1024); // 128MB Safe
                arena_init(&arena_B, 1512 * 1024 * 1024);
                
                // MCTS Trees (No persistence)
                Node *root_node_A = NULL;
                Node *root_node_B = NULL;
                
                // Alternate colors: Even games A=White, Odd games A=Black
                int A_is_white = (game % 2 == 0);
                
                while (1) {
                    // Start of Turn: ALWAYS RESET ARENA (Stateless MCTS)
                    arena_reset(&arena_A);
                    root_node_A = NULL;
                    
                    arena_reset(&arena_B);
                    root_node_B = NULL;

                    // Check Terminal State
                    MoveList legal_moves;
                    generate_moves(&state, &legal_moves);
                    
                    if (legal_moves.count == 0) {
                        // Current player has no moves -> LOSS
                        int winner = (state.current_player == WHITE) ? BLACK : WHITE;
                        if (winner == WHITE) {
                            if (A_is_white) wins_A++; else wins_B++;
                        } else {
                            if (A_is_white) wins_B++; else wins_A++;
                        }
                        break;
                    }
                    
                    if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES || move_count > 200) {
                        draws++;
                        break;
                    }
                    
                    Move best_move;
                    int player = state.current_player; // WHITE or BLACK

                    
                    if ((player == WHITE && A_is_white) || (player == BLACK && !A_is_white)) {
                        // Player A's turn
                        if (!root_node_A) root_node_A = mcts_create_root(state, &arena_A, pA->config);
                        
                        best_move = mcts_search(root_node_A, &arena_A, TIME_PER_MOVE, pA->config, &stats_A, &root_node_A);
                        
                        // Update opponent's tree (B) - Disabled but kept for structure
                        if (root_node_B && pB->config.use_tree_reuse) {
                            Node *next = find_child_by_move(root_node_B, &best_move);
                            root_node_B = next; 
                        }
                    } else {
                        // Player B's turn
                        if (!root_node_B) root_node_B = mcts_create_root(state, &arena_B, pB->config);
                        
                        best_move = mcts_search(root_node_B, &arena_B, TIME_PER_MOVE, pB->config, &stats_B, &root_node_B);
                        
                        // Update opponent's tree (A)
                        if (root_node_A && pA->config.use_tree_reuse) {
                            Node *next = find_child_by_move(root_node_A, &best_move);
                            root_node_A = next;
                        }
                    }
                    
                    apply_move(&state, &best_move);
                    move_count++;
                }
                
                // Aggregate game stats to reduction variables
                total_nodes_A += stats_A.total_iterations;
                total_depth_A += stats_A.total_depth;
                total_moves_A += stats_A.total_moves;
                
                total_nodes_B += stats_B.total_iterations;
                total_depth_B += stats_B.total_depth;
                total_moves_B += stats_B.total_moves;
                
                arena_free(&arena_A);
                arena_free(&arena_B);
            }
            
            // --- PRINT MATCH STATISTICS ---
            printf("   > %s Stats: %.0f Nodes/Move, Depth %.1f\n", 
                   pA->name, 
                   (total_moves_A > 0) ? (double)total_nodes_A / total_moves_A : 0, 
                   (total_moves_A > 0) ? (double)total_depth_A / total_moves_A : 0);
            
            printf("   > %s Stats: %.0f Nodes/Move, Depth %.1f\n", 
                   pB->name, 
                   (total_moves_B > 0) ? (double)total_nodes_B / total_moves_B : 0, 
                   (total_moves_B > 0) ? (double)total_depth_B / total_moves_B : 0);
            
            // --- UPDATE STATS ---
            pA->wins += wins_A;
            pA->losses += wins_B;
            pA->draws += draws;
            pA->points += wins_A + (0.5 * draws);
            
            pB->wins += wins_B;
            pB->losses += wins_A;
            pB->draws += draws;
            pB->points += wins_B + (0.5 * draws);
            
            // Sequential ELO Update (Approximate)
            double score_A = wins_A + 0.5 * draws;
            double Ea = 1.0 / (1.0 + pow(10.0, (pB->elo - pA->elo) / 400.0));
            double expected_score_A = Ea * games_per_pairing;
            
            // Update ELO
            double delta = (score_A - expected_score_A) * K_FACTOR; 
            
            pA->elo += delta;
            pB->elo -= delta;
            
            printf("Result: %s %d - %d %s (Draws: %d)\n", pA->name, wins_A, wins_B, pB->name, draws);
            printf("New ELO: %s (%.0f), %s (%.0f)\n", pA->name, pA->elo, pB->name, pB->elo);
        }
    }
    
    double total_time = ((double)clock() - start_time) / CLOCKS_PER_SEC;
    
    // --- PRINT LEADERBOARD ---
    qsort(players, num_players, sizeof(TournamentPlayer), compare_players);
    
    printf("\n\n");
    printf("========================================================================\n");
    printf("                           TOURNAMENT STANDINGS                         \n");
    printf("========================================================================\n");
    printf("| Rank | Name       | Points | Wins | Loss | Draw | ELO  | Win Rate |\n");
    printf("|------|------------|--------|------|------|------|------|----------|\n");
    
    for (int i = 0; i < num_players; i++) {
        TournamentPlayer *p = &players[i];
        int total_games = p->wins + p->losses + p->draws;
        double win_rate = (total_games > 0) ? 100.0 * p->wins / total_games : 0.0;
        
        printf("| %-4d | %-10s | %-6.1f | %-4d | %-4d | %-4d | %-4.0f | %5.1f%%   |\n", 
               i+1, p->name, p->points, p->wins, p->losses, p->draws, p->elo, win_rate);
    }
    printf("========================================================================\n");
    printf("Total Time: %.2f seconds\n", total_time);
    
    return 0;
}
