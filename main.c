#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "game.h"
#include "mcts.h"
#include "params.h"

// --- HELPERS DI DEBUG ---

int main() {
    zobrist_init(); // Initialize Zobrist Hashing Keys
    
    // 1. Setup Random e Arena
    srand(time(NULL));
    
    Arena mcts_arena;
    arena_init(&mcts_arena, ARENA_SIZE);
    
    GameState state;
    init_game(&state);
    
    printf("--- MATCH AI vs AI ---\n");
    printf("Bianco: %.2fs | Nero: %.2fs\n", TIME_WHITE, TIME_BLACK);
    print_board(&state);
    
    int turn_count = 1;
    
    // 2. Loop Infinito
    while(1) {
        printf("\n=== Turno %d: %s ===\n", turn_count, 
               (state.current_player == WHITE) ? "BIANCO" : "NERO");

        // A. Controlla Game Over
        MoveList list;
        generate_moves(&state, &list);
        if (list.count == 0) {
            printf("GAME OVER! Vince %s (l'avversario non ha mosse)\n", 
                   (state.current_player == WHITE) ? "NERO" : "BIANCO");
            break;
        }

        // B. Controlla Patta
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            printf("GAME OVER! Pareggio (40 mosse senza catture).\n");
            break;
        }

        // C. MCTS: Pensa alla mossa
        // IMPORTANTE: Resetta l'arena prima di ogni pensiero!
        arena_reset(&mcts_arena); 
        
        Node *root = mcts_create_root(state, &mcts_arena);
        
        MCTSConfig config = {
            .ucb1_c = UCB1_C,
            .rollout_epsilon = DEFAULT_ROLLOUT_EPSILON, // Definito in params.h o 0.2
            .draw_score = DRAW_SCORE,
            .expansion_threshold = EXPANSION_THRESHOLD,
            .use_lookahead = DEFAULT_USE_LOOKAHEAD,
            .verbose = 1,  // Enable output for normal gameplay
            .use_tree_reuse = 0, // Disable in normal play (single-tree)
            .use_ucb1_tuned = 0, // Default: Standard UCB1
            .use_tt = 0,         // Default: No Transposition Table
            .use_solver = 0,     // Default: No Solver
            .use_progressive_bias = 0, // Default: No Bias
            .bias_constant = 0.0
        };

        double time_limit = (state.current_player == WHITE) ? TIME_WHITE : TIME_BLACK;
        Move chosen_move = mcts_search(root, &mcts_arena, time_limit, config, NULL, NULL);
        

        // D. Stampa info mossa
        printf("Mossa scelta: ");
        print_move_description(chosen_move);

        // E. Applica mossa
        apply_move(&state, &chosen_move);
        print_board(&state);
        
        turn_count++;
        
        // Opzionale: pausa per vedere cosa succede
        // system("sleep 1"); 
    }

    arena_free(&mcts_arena);
    return 0;
}