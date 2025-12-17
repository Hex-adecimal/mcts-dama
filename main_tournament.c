#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

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

/**
 * Updates ELO rating after a game.
 * @param current_elo Current ELO rating.
 * @param expected Expected score (from expected_score function).
 * @param actual Actual score (1.0 = win, 0.5 = draw, 0.0 = loss).
 * @return New ELO rating.
 */
double update_elo(double current_elo, double expected, double actual) {
    return current_elo + K_FACTOR * (actual - expected);
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
                root = mcts_create_root(state, arena);
            }
        } else {
            // First move or tree reuse disabled
            root = mcts_create_root(state, arena);
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
//  MAIN
// ================================================================================================

int main() {
    srand(time(NULL));
    
    // Initialize Arena
    Arena mcts_arena;
    arena_init(&mcts_arena, ARENA_SIZE);
    
    printf("=== MCTS TOURNAMENT: Config A vs Config B ===\n\n");
    
    // Define two configurations to compare
    MCTSConfig config_A = {
        .ucb1_c = UCB1_C,
        .rollout_epsilon = 0.3,
        .draw_score = DRAW_SCORE,
        .expansion_threshold = EXPANSION_THRESHOLD,
        .use_lookahead = 0,  // Baseline: no lookahead
        .verbose = 0,        // Quiet mode for tournament
        .use_tree_reuse = 1  // Enable tree reuse
    };
    
    MCTSConfig config_B = {
        .ucb1_c = UCB1_C,
        .rollout_epsilon = DEFAULT_ROLLOUT_EPSILON,
        .draw_score = DRAW_SCORE,
        .expansion_threshold = EXPANSION_THRESHOLD,
        .use_lookahead = 1,  // Improved: with lookahead
        .verbose = 0,        // Quiet mode for tournament
        .use_tree_reuse = 1  // Enable tree reuse
    };
    
    printf("Config A: epsilon=%.2f, lookahead=%d\n", config_A.rollout_epsilon, config_A.use_lookahead);
    printf("Config B: epsilon=%.2f, lookahead=%d\n\n", config_B.rollout_epsilon, config_B.use_lookahead);
    
    // Tournament settings
    int num_games = 50;  // Total games (25 per color)
    int wins_A = 0, wins_B = 0, draws = 0;
    
    double elo_A = INITIAL_ELO;
    double elo_B = INITIAL_ELO;
    
    // Initialize statistics
    MCTSStats stats_A = {0, 0, 0, 0.0, 0};
    MCTSStats stats_B = {0, 0, 0, 0.0, 0};
    
    printf("Playing %d games...\n\n", num_games);
    
    // Play tournament
    for (int game = 0; game < num_games; game++) {
        int winner;
        
        // Alternate colors
        if (game % 2 == 0) {
            // A plays White, B plays Black
            printf("Game %d: A(White) vs B(Black) ... ", game + 1);
            fflush(stdout);
            winner = play_match(&config_A, &config_B, &mcts_arena, &stats_A, &stats_B);
            
            if (winner == WHITE) {
                printf("A wins");
                wins_A++;
                
                double exp_A = expected_score(elo_A, elo_B);
                double exp_B = expected_score(elo_B, elo_A);
                elo_A = update_elo(elo_A, exp_A, 1.0);
                elo_B = update_elo(elo_B, exp_B, 0.0);
            } else if (winner == BLACK) {
                printf("B wins");
                wins_B++;
                
                double exp_A = expected_score(elo_A, elo_B);
                double exp_B = expected_score(elo_B, elo_A);
                elo_A = update_elo(elo_A, exp_A, 0.0);
                elo_B = update_elo(elo_B, exp_B, 1.0);
            } else {
                printf("Draw");
                draws++;
                
                double exp_A = expected_score(elo_A, elo_B);
                double exp_B = expected_score(elo_B, elo_A);
                elo_A = update_elo(elo_A, exp_A, 0.5);
                elo_B = update_elo(elo_B, exp_B, 0.5);
            }
        } else {
            // B plays White, A plays Black
            printf("Game %d: B(White) vs A(Black) ... ", game + 1);
            fflush(stdout);
            winner = play_match(&config_B, &config_A, &mcts_arena, &stats_B, &stats_A);
            
            if (winner == WHITE) {
                printf("B wins ");
                wins_B++;
                
                double exp_A = expected_score(elo_A, elo_B);
                double exp_B = expected_score(elo_B, elo_A);
                elo_A = update_elo(elo_A, exp_A, 0.0);
                elo_B = update_elo(elo_B, exp_B, 1.0);
            } else if (winner == BLACK) {
                printf("A wins ");
                wins_A++;
                
                double exp_A = expected_score(elo_A, elo_B);
                double exp_B = expected_score(elo_B, elo_A);
                elo_A = update_elo(elo_A, exp_A, 1.0);
                elo_B = update_elo(elo_B, exp_B, 0.0);
            } else {
                printf("Draw ");
                draws++;
                
                double exp_A = expected_score(elo_A, elo_B);
                double exp_B = expected_score(elo_B, elo_A);
                elo_A = update_elo(elo_A, exp_A, 0.5);
                elo_B = update_elo(elo_B, exp_B, 0.5);
            }
        }
        
        // Print move count and averages for this game
        int moves_this_game = (game % 2 == 0) ? 
            (stats_A.total_moves + stats_B.total_moves - 
             (game > 0 ? (stats_A.total_moves + stats_B.total_moves) / (game + 1) * game : 0)) :
            (stats_B.total_moves + stats_A.total_moves - 
             (game > 0 ? (stats_B.total_moves + stats_A.total_moves) / (game + 1) * game : 0));
        printf("\n");
    }
    
    // Results
    printf("\n=== TOURNAMENT RESULTS ===\n");
    printf("Config A: %d wins, %d losses, %d draws\n", wins_A, wins_B, draws);
    printf("Config B: %d wins, %d losses, %d draws\n", wins_B, wins_A, draws);
    printf("\n");
    printf("Win Rate A: %.1f%%\n", (wins_A / (double)num_games) * 100.0);
    printf("Win Rate B: %.1f%%\n", (wins_B / (double)num_games) * 100.0);
    printf("\n");
    
    // Print detailed statistics
    printf("=== PERFORMANCE STATISTICS ===\n");
    printf("Config A:\n");
    printf("  Average simulations/move: %ld\n", stats_A.total_moves > 0 ? stats_A.total_iterations / stats_A.total_moves : 0);
    printf("  Average tree depth: %ld\n", stats_A.total_moves > 0 ? stats_A.total_depth / stats_A.total_moves : 0);
    printf("  Average iterations/sec: %.0f\n", stats_A.total_time > 0 ? stats_A.total_iterations / stats_A.total_time : 0);
    printf("  Average memory/move: %.1f KB\n", stats_A.total_moves > 0 ? (stats_A.total_memory / stats_A.total_moves) / 1024.0 : 0);
    printf("  Total moves: %d\n", stats_A.total_moves);
    printf("  Total time: %.1f seconds\n", stats_A.total_time);
    printf("\n");
    printf("Config B:\n");
    printf("  Average simulations/move: %ld\n", stats_B.total_moves > 0 ? stats_B.total_iterations / stats_B.total_moves : 0);
    printf("  Average tree depth: %ld\n", stats_B.total_moves > 0 ? stats_B.total_depth / stats_B.total_moves : 0);
    printf("  Average iterations/sec: %.0f\n", stats_B.total_time > 0 ? stats_B.total_iterations / stats_B.total_time : 0);
    printf("  Average memory/move: %.1f KB\n", stats_B.total_moves > 0 ? (stats_B.total_memory / stats_B.total_moves) / 1024.0 : 0);
    printf("  Total moves: %d\n", stats_B.total_moves);
    printf("  Total time: %.1f seconds\n", stats_B.total_time);
    printf("\n");
    
    printf("ELO Rating A: %.0f (Δ%+.0f)\n", elo_A, elo_A - INITIAL_ELO);
    printf("ELO Rating B: %.0f (Δ%+.0f)\n", elo_B, elo_B - INITIAL_ELO);
    printf("ELO Difference: %.0f points\n", fabs(elo_B - elo_A));
    
    arena_free(&mcts_arena);
    return 0;
}
