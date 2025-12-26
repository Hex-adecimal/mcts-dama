#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <unistd.h>
#include "game.h"
#include "movegen.h"
#include "mcts.h"
#include "cnn.h"
#include "params.h"

#define DEFAULT_GAMES_PER_PAIRING   10
#define DEFAULT_TIME_TOURNAMENT     TIME_LOW



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
               MCTSStats *stats_white, MCTSStats *stats_black, double time_limit) {
    GameState state;
    init_game(&state);
    
    int turn_count = 0;
    int max_turns = MAX_GAME_TURNS;
    
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
        
        Move chosen_move;
        
        // SPECIAL: Random Player Flag (max_nodes = -1)
        if (config->max_nodes == -1) {
            MoveList legals;
            generate_moves(&state, &legals);
            if (legals.count > 0) {
                chosen_move = legals.moves[rand() % legals.count];
            } else {
                chosen_move = (Move){0};
            }
        } else {
            Node *new_root = NULL;
            chosen_move = mcts_search(root, arena, time_limit, *config, stats, &new_root);
            
            // Update opponent's root to our chosen child (for their next turn)
            if (config->use_tree_reuse && new_root) {
                *opponent_root = new_root;
            }
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
    const char *name;
    MCTSConfig config;
    double elo;
    int wins;
    int losses;
    int draws;
    double points; // 1.0 per win, 0.5 per draw
    long long total_iters;      // Thread iterations
    long long tree_nodes;       // Unique tree nodes
    double total_time;
    long long total_moves;
} TournamentPlayer;

// Helper to format numbers with thousand separators (e.g. 1.000.000)
// Uses a pool of buffers to allow multiple calls in the same printf
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

#define MAX_PLAYERS 16

typedef struct {
    int wins;
    int losses;
    int draws;
} MatchResult;

void calculate_elo_batch(TournamentPlayer *players, int num_players, MatchResult results[MAX_PLAYERS][MAX_PLAYERS]) {
    for (int i=0; i<num_players; i++) {
        players[i].elo = 1200.0;
    }
    
    int iterations = 10000;
    for (int iter = 0; iter < iterations; iter++) {
        double max_diff = 0.0;
        double new_elos[MAX_PLAYERS];
        
        for (int i = 0; i < num_players; i++) {
            double actual_score = 0;
            double expected_sum = 0;
            
            for (int j = 0; j < num_players; j++) {
                if (i == j) continue;
                
                int games = results[i][j].wins + results[i][j].losses + results[i][j].draws;
                if (games == 0) continue;
                
                actual_score += results[i][j].wins + 0.5 * results[i][j].draws;
                
                // Clamp difference to avoid pow() explosion
                double diff = players[j].elo - players[i].elo;
                if (diff > 800.0) diff = 800.0;
                if (diff < -800.0) diff = -800.0;
                
                double E = 1.0 / (1.0 + pow(10.0, diff / 400.0));
                expected_sum += E * games;
            }
            
            // Step size damping (Newton-like would be better, but this is stable)
            double delta = (actual_score - expected_sum) * 0.1; 
            new_elos[i] = players[i].elo + delta;
        }
        
        // Re-center around 1200
        double sum = 0;
        for(int k=0; k<num_players; k++) sum += new_elos[k];
        double avg = sum / num_players;
        double shift = 1200.0 - avg;
        
        max_diff = 0.0;
        for (int i = 0; i < num_players; i++) {
            new_elos[i] += shift;
            double d = fabs(new_elos[i] - players[i].elo);
            if (d > max_diff) max_diff = d;
            players[i].elo = new_elos[i];
        }
        
        if (max_diff < 0.001) break;
    }
    
    printf("\n[Elo Solver] Batch results processed successfully.\n");
}

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

int main(int argc, char *argv[]) {
    int games_per_pairing = DEFAULT_GAMES_PER_PAIRING;
    double time_limit = DEFAULT_TIME_TOURNAMENT;
    int node_limit = 0;
    
    int opt;
    while ((opt = getopt(argc, argv, "g:t:n:h")) != -1) {
        switch (opt) {
            case 'g': games_per_pairing = atoi(optarg); break;
            case 't': time_limit = atof(optarg); break;
            case 'n': node_limit = atoi(optarg); break;
            case 'h':
            default:
                printf("Usage: %s [-g games] [-t time_limit] [-n node_limit]\n", argv[0]);
                printf("  -g: Games per pairing (default: %d)\n", DEFAULT_GAMES_PER_PAIRING);
                printf("  -t: Time limit per move in seconds (default: %.2fs)\n", DEFAULT_TIME_TOURNAMENT);
                printf("  -n: Node limit per move (0 = use time limit, default: %d)\n", 0);
                return 0;
        }
    }
    
    // If using node limit, set a large time fallback
    if (node_limit > 0) {
        time_limit = 600.0; // 10 minutes fallback (node limit should hit first)
    }

    zobrist_init();
    init_move_tables();
    srand(time(NULL));
    
    printf("=== MCTS ROUND ROBIN TOURNAMENT ===\n");
    if (node_limit > 0) {
        printf("Type: Intelligence Mode (Node-limited: %d per move)\n", node_limit);
    } else {
        printf("Type: Performance Mode (Time-limited: %.2fs per move)\n", time_limit);
    }
    printf("Games per pairing: %d\n", games_per_pairing);
    
#ifdef _OPENMP
    printf("Parallel Execution Enabled: %d Threads\n", omp_get_max_threads());
#endif

    // --- CNN FOR PUCT ---
    CNNWeights cnn_weights;
    cnn_init(&cnn_weights);
    
    const char *weight_paths[] = {
        //"bin/cnn_weights.bin",
        //"cnn_weights.bin",
        //"models/cnn_weights.bin",
        //"bin/cnn_weights_final.bin"
        "models/cnn_weights_final.bin"
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
        printf("⚠️  CNN using random weights (BRAINLESS - weights not found in standard paths)\n");
    }

    // --- CONFIGURATIONS ---
    MCTSConfig cfg_pure        = mcts_get_preset(MCTS_PRESET_PURE_VANILLA);
    MCTSConfig cfg_vanilla     = mcts_get_preset(MCTS_PRESET_VANILLA);
    MCTSConfig cfg_grandmaster = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
    MCTSConfig cfg_cnn         = mcts_get_preset(MCTS_PRESET_ALPHA_ZERO);
    cfg_cnn.cnn_weights = &cnn_weights;
    cfg_grandmaster.cnn_weights = &cnn_weights; // Grandmaster also uses CNN

    MCTSConfig cfg_tt_only     = mcts_get_preset(MCTS_PRESET_TT_ONLY);
    MCTSConfig cfg_solver_only = mcts_get_preset(MCTS_PRESET_SOLVER_ONLY);
    MCTSConfig cfg_tuned_only  = mcts_get_preset(MCTS_PRESET_TUNED_ONLY);
    MCTSConfig cfg_fpu_only    = mcts_get_preset(MCTS_PRESET_FPU_ONLY);
    MCTSConfig cfg_decay_only  = mcts_get_preset(MCTS_PRESET_DECAY_ONLY);
    MCTSConfig cfg_lookahead   = mcts_get_preset(MCTS_PRESET_LOOKAHEAD_ONLY);
    MCTSConfig cfg_reuse       = mcts_get_preset(MCTS_PRESET_TREE_REUSE_ONLY);
    
    MCTSConfig cfg_weights     = mcts_get_preset(MCTS_PRESET_WEIGHTS_ONLY);
    MCTSConfig cfg_smartroll   = mcts_get_preset(MCTS_PRESET_SMART_ROLLOUTS);
    MCTSConfig cfg_progbias    = mcts_get_preset(MCTS_PRESET_PROG_BIAS_ONLY);
    MCTSConfig cfg_random      = (MCTSConfig){ .max_nodes = -1 }; // Special Random Flag

    // Apply node limit if provided
    MCTSConfig *all_configs[] = {
        &cfg_pure, &cfg_vanilla, &cfg_grandmaster, &cfg_cnn, 
        &cfg_tt_only, &cfg_solver_only, &cfg_tuned_only, &cfg_fpu_only, 
        &cfg_decay_only, &cfg_lookahead, &cfg_reuse, 
        &cfg_weights, &cfg_smartroll, &cfg_progbias
    };
    const int NUM_PLAYERS_EXPECTED = 14;
    for (int i = 0; i < NUM_PLAYERS_EXPECTED; i++) all_configs[i]->max_nodes = node_limit;

    // --- PLAYER ROSTER ---
    TournamentPlayer players[] = {
        { "PureVanilla",  cfg_pure,         1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "Vanilla",      cfg_vanilla,      1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "Grandmaster",  cfg_grandmaster,  1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { weights_loaded ? "CNN-PUCT (TRAINED)" : "CNN-PUCT (BRAINLESS)", cfg_cnn, 1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "TT-Only",      cfg_tt_only,      1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "Solver-Only",  cfg_solver_only,  1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "Tuned-Only",   cfg_tuned_only,   1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "FPU-Only",     cfg_fpu_only,     1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "Decay-Only",   cfg_decay_only,   1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "LookAhead",    cfg_lookahead,    1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "TreeReuse",    cfg_reuse,        1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "WeightsOnly",  cfg_weights,      1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "SmartRoll",    cfg_smartroll,    1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 },
        { "ProgBias",     cfg_progbias,     1200.0, 0, 0, 0, 0.0, 0, 0, 0.0, 0 }
    };
    int num_players = NUM_PLAYERS_EXPECTED;
    
    double start_time = (double)clock();
    
    // Matrix to store results
    MatchResult match_results[MAX_PLAYERS][MAX_PLAYERS];
    memset(match_results, 0, sizeof(match_results));
    
    // --- TOURNAMENT LOOP ---
    for (int i = 0; i < num_players; i++) {
        for (int j = i + 1; j < num_players; j++) {
            TournamentPlayer *pA = &players[i];
            TournamentPlayer *pB = &players[j];
            
            printf("\n┌──────────────────────────────────────────────────────────┐\n");
            printf("│ Match: %-18s vs %-18s │\n", pA->name, pB->name);
            printf("└──────────────────────────────────────────────────────────┘\n");
            
            int wins_A = 0;
            int wins_B = 0;
            int draws = 0;
            
            long long total_iters_A = 0, total_depth_A = 0, total_moves_A = 0, tree_nodes_A = 0, memory_A = 0;
            long long total_iters_B = 0, total_depth_B = 0, total_moves_B = 0, tree_nodes_B = 0, memory_B = 0;
            double total_time_A = 0.0, total_time_B = 0.0;
            
            #pragma omp parallel for reduction(+:wins_A, wins_B, draws, total_iters_A, total_depth_A, total_moves_A, total_iters_B, total_depth_B, total_moves_B, total_time_A, total_time_B, tree_nodes_A, tree_nodes_B, memory_A, memory_B) schedule(dynamic)
            for (int game = 0; game < games_per_pairing; game++) {
                GameState state;
                init_game(&state);
                
                // Stats per game (accumulated locally)
                MCTSStats stats_A = {0};
                MCTSStats stats_B = {0};
                
                int move_count = 0;
                
                // Arena per thread
                Arena arena_A, arena_B;
                arena_init(&arena_A, ARENA_SIZE_TOURNAMENT);
                arena_init(&arena_B, ARENA_SIZE_TOURNAMENT);
                
                // MCTS Trees (No persistence)
                Node *root_node_A = NULL;
                Node *root_node_B = NULL;
                
                // Alternate colors: Even games A=White, Odd games A=Black
                int A_is_white = (game % 2 == 0);
                
                // History buffer for CNN
                GameState history[2];
                int history_valid = 0;
                
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
                        
                        // --- HISTORY INJECTION (A) ---
                        if (pA->config.cnn_weights && history_valid >= 1) {
                             Node *h1 = (Node*)arena_alloc(&arena_A, sizeof(Node));
                             memset(h1, 0, sizeof(Node));
                             pthread_mutex_init(&h1->lock, NULL);
                             h1->state = history[0]; 
                             root_node_A->parent = h1;
                             if (history_valid >= 2) {
                                 Node *h2 = (Node*)arena_alloc(&arena_A, sizeof(Node));
                                 memset(h2, 0, sizeof(Node));
                                 pthread_mutex_init(&h2->lock, NULL);
                                 h2->state = history[1];
                                 h1->parent = h2;
                             }
                        }
                        // -----------------------------
                        
                        best_move = mcts_search(root_node_A, &arena_A, time_limit, pA->config, &stats_A, &root_node_A);
                        
                        // Update opponent's tree (B) - Disabled but kept for structure
                        if (root_node_B && pB->config.use_tree_reuse) {
                            Node *next = find_child_by_move(root_node_B, &best_move);
                            root_node_B = next; 
                        }
                    } else {
                        // Player B's turn
                        if (!root_node_B) root_node_B = mcts_create_root(state, &arena_B, pB->config);
                        
                        // --- HISTORY INJECTION (B) ---
                        if (pB->config.cnn_weights && history_valid >= 1) {
                             Node *h1 = (Node*)arena_alloc(&arena_B, sizeof(Node));
                             memset(h1, 0, sizeof(Node));
                             pthread_mutex_init(&h1->lock, NULL);
                             h1->state = history[0]; 
                             root_node_B->parent = h1;
                             if (history_valid >= 2) {
                                 Node *h2 = (Node*)arena_alloc(&arena_B, sizeof(Node));
                                 memset(h2, 0, sizeof(Node));
                                 pthread_mutex_init(&h2->lock, NULL);
                                 h2->state = history[1];
                                 h1->parent = h2;
                             }
                        }
                        // -----------------------------
                        
                        best_move = mcts_search(root_node_B, &arena_B, time_limit, pB->config, &stats_B, &root_node_B);
                        
                        // Update opponent's tree (A)
                        if (root_node_A && pA->config.use_tree_reuse) {
                            Node *next = find_child_by_move(root_node_A, &best_move);
                            root_node_A = next;
                        }
                    }
                    
                    if (game == 0) {
                        double val = 0.0;
                        long long nodes = 0;
                        if ((player == WHITE && A_is_white) || (player == BLACK && !A_is_white)) {
                            val = mcts_get_avg_root_ucb(root_node_A, pA->config);
                            nodes = stats_A.current_move_iterations;
                            printf("\r[G0] Turn %-3d | %-18s | Nodes: %-7s | Avg UCB: %5.2f  ", 
                                   move_count, pA->name, format_thousands(nodes), val);
                        } else {
                            val = mcts_get_avg_root_ucb(root_node_B, pB->config);
                            nodes = stats_B.current_move_iterations;
                            printf("\r[G0] Turn %-3d | %-18s | Nodes: %-7s | Avg UCB: %5.2f  ", 
                                   move_count, pB->name, format_thousands(nodes), val);
                        }
                        fflush(stdout);
                    }
                    
                    // Update History
                    history[1] = history[0];
                    history[0] = state;
                    if (history_valid < 2) history_valid++;
                    
                    apply_move(&state, &best_move);
                    move_count++;
                }
                
                total_iters_A += stats_A.total_iterations;
                total_depth_A += stats_A.total_depth;
                total_moves_A += stats_A.total_moves;
                tree_nodes_A += stats_A.total_nodes;
                memory_A += stats_A.total_memory;
                
                total_iters_B += stats_B.total_iterations;
                total_depth_B += stats_B.total_depth;
                total_moves_B += stats_B.total_moves;
                tree_nodes_B += stats_B.total_nodes;
                memory_B += stats_B.total_memory;
                
                total_time_A += stats_A.total_time;
                total_time_B += stats_B.total_time;
                
                arena_free(&arena_A);
                arena_free(&arena_B);
            }
            
            // --- PRINT MATCH STATISTICS ---
            printf("\n");
            // --- PRINT MATCH STATISTICS ---
            printf("\n");
            printf("   > %-18s Stats: %s iter/move, %s nodes, %.1f KB mem, Depth %.1f\n", 
                   pA->name, 
                   (total_moves_A > 0) ? format_thousands(total_iters_A / total_moves_A) : "0",
                   (total_moves_A > 0) ? format_thousands(tree_nodes_A / total_moves_A) : "0",
                   (total_moves_A > 0) ? (double)memory_A / total_moves_A / 1024.0 : 0,
                   (total_moves_A > 0) ? (double)total_depth_A / total_moves_A : 0);
            printf("   > %-18s Stats: %s iter/move, %s nodes, %.1f KB mem, Depth %.1f\n", 
                   pB->name, 
                   (total_moves_B > 0) ? format_thousands(total_iters_B / total_moves_B) : "0", 
                   (total_moves_B > 0) ? format_thousands(tree_nodes_B / total_moves_B) : "0",
                   (total_moves_B > 0) ? (double)memory_B / total_moves_B / 1024.0 : 0,
                   (total_moves_B > 0) ? (double)total_depth_B / total_moves_B : 0);
            
            // --- UPDATE STATS ---
            pA->wins += wins_A;
            pA->losses += wins_B;
            pA->draws += draws;
            pA->points += wins_A + (0.5 * draws);
            pA->total_iters += total_iters_A;
            pA->tree_nodes += tree_nodes_A;
            pA->total_time += total_time_A;
            pA->total_moves += total_moves_A;
            
            pB->wins += wins_B;
            pB->losses += wins_A;
            pB->draws += draws;
            pB->points += wins_B + (0.5 * draws);
            pB->total_iters += total_iters_B;
            pB->tree_nodes += tree_nodes_B;
            pB->total_time += total_time_B;
            pB->total_moves += total_moves_B;
            
            // Store for Batch Calculation
            match_results[i][j].wins = wins_A;
            match_results[i][j].losses = wins_B;
            match_results[i][j].draws = draws;
            
            match_results[j][i].wins = wins_B;
            match_results[j][i].losses = wins_A;
            match_results[j][i].draws = draws;
            
            // Removed Loop ELO Update - Wait for end
            printf("Result: %s %d - %d %s (Draws: %d)\n", pA->name, wins_A, wins_B, pB->name, draws);
        }
    }
    
    // --- BATCH ELO CALCULATION ---
    printf("\nCalculating Batch ELO...\n");
    calculate_elo_batch(players, num_players, match_results);
    
    double total_time = ((double)clock() - start_time) / CLOCKS_PER_SEC;
    
    // --- PRINT LEADERBOARD ---
    qsort(players, num_players, sizeof(TournamentPlayer), compare_players);
    
    printf("\n\n");
    printf("┌──────┬────────────────────────┬────────┬──────┬──────┬──────┬──────┬────────────┬────────────┬──────────┐\n");
    printf("│ Rank │ Name                   │ Points │ Wins │ Loss │ Draw │ ELO  │ iter/Move  │ Nodes/Move │ Win Rate │\n");
    printf("├──────┼────────────────────────┼────────┼──────┼──────┼──────┼──────┼────────────┼────────────┼──────────┤\n");
    
    for (int i = 0; i < num_players; i++) {
        TournamentPlayer *p = &players[i];
        int total_games = p->wins + p->losses + p->draws;
        double win_rate = (total_games > 0) ? 100.0 * p->wins / total_games : 0.0;
        
        long long avg_iters = (p->total_moves > 0) ? p->total_iters / p->total_moves : 0;
        long long avg_nodes = (p->total_moves > 0) ? p->tree_nodes / p->total_moves : 0;

        printf("│ %-4d │ %-22s │ %-6.1f │ %-4d │ %-4d │ %-4d │ %-4.0f │ %-10s │ %-10s │ %5.1f%%   │\n", 
               i+1, p->name, p->points, p->wins, p->losses, p->draws, p->elo, 
               format_thousands(avg_iters), format_thousands(avg_nodes), win_rate);
    }
    printf("└──────┴────────────────────────┴────────┴──────┴──────┴──────┴──────┴────────────┴────────────┴──────────┘\n");
    printf("Total Execution Time: %.2f seconds\n", total_time);
    
    // Cleanup
    cnn_free(&cnn_weights);
    
    return 0;
}
