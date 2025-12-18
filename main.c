#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <ctype.h>

#include "game.h"
#include "mcts.h"
#include "params.h"

// Parse "A3" -> 0-63 index
// Returns -1 on error
int parse_square(const char *str) {
    if (!str || strlen(str) < 2) return -1;
    char col_char = toupper(str[0]);
    char row_char = str[1];
    
    if (col_char < 'A' || col_char > 'H') return -1;
    if (row_char < '1' || row_char > '8') return -1;
    
    int col = col_char - 'A';
    int row = row_char - '1';
    
    return row * 8 + col;
}

// Convert 0-63 index -> "A3"
void format_square(int sq, char *buf) {
    int row = sq / 8;
    int col = sq % 8;
    sprintf(buf, "%c%d", 'A' + col, row + 1);
}

// Rename to avoid conflict
void display_moves(const MoveList *list, Node *root) {
    char buf_from[4], buf_to[4];
    printf("Mosse valide (MCTS Stats):\n");
    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        int dest_idx = (m.length == 0) ? 1 : m.length;
        format_square(m.path[0], buf_from);
        format_square(m.path[dest_idx], buf_to); 
        
        printf("%d: %s -> %s", i + 1, buf_from, buf_to);
        if (m.length > 0) printf(" (Cattura: %d)", m.length);
        
        // MCTS HINT
        if (root) {
            Node *child = find_child_by_move(root, &m);
            if (child && child->visits > 0) {
                // Score is accumulated for the player who made the move (Current Human).
                // Usually score is +1 for Win (for that player).
                double win_rate = (child->score / child->visits) * 100.0;
                // Clamp visual range [0, 100]
                if (win_rate < 0) win_rate = 0; 
                if (win_rate > 100) win_rate = 100;
                
                printf("  [Win: %.1f%%, Visits: %d]", win_rate, child->visits);
                
                // Labels are from Human's perspective:
                // SOLVED_WIN at child = AI (player to move at child) can win = BAD for Human
                // SOLVED_LOSS at child = AI will lose = GOOD for Human
                if (child->status == SOLVED_WIN) printf(" (LOSS)");
                if (child->status == SOLVED_LOSS) printf(" (WIN)");
                if (child->status == SOLVED_DRAW) printf(" (DRAW)");
            }
        }
        
        printf("\n");
    }
}

int main() {
    zobrist_init(); // Initialize Zobrist Hashing Keys
    srand(time(NULL));
    
    Arena mcts_arena;
    arena_init(&mcts_arena, ARENA_SIZE);
    
    GameState state;
    init_game(&state);
    
    printf("=== DAMA ITALIANA: HUMAN VS GRANDMASTER ===\n");
    printf("Grandmaster MCTS Config: SPSA Weights + UCB1-Tuned + TT + Solver + Tree Reuse\n");
    
    int human_color = -1;
    while (human_color != WHITE && human_color != BLACK) {
        printf("Scegli il tuo colore (W = Bianco/Inizia, B = Nero): ");
        char choice;
        scanf(" %c", &choice);
        choice = toupper(choice);
        if (choice == 'W') human_color = WHITE;
        else if (choice == 'B') human_color = BLACK;
    }
    
    printf("\nPartita Iniziata! Tu sei il %s.\n", (human_color == WHITE) ? "BIANCO" : "NERO");
    print_board(&state);
    
    int turn_count = 1;

    // --- PERSISTENT MCTS ROOT ---
    Node *root = NULL;

    // Pre-define Config
    MCTSConfig config = {
        .ucb1_c = UCB1_C,
        .rollout_epsilon = DEFAULT_ROLLOUT_EPSILON, 
        .draw_score = DRAW_SCORE,
        .expansion_threshold = EXPANSION_THRESHOLD,
        .use_lookahead = DEFAULT_USE_LOOKAHEAD,
        .verbose = 1,  
        .use_tree_reuse = 1, // ENABLED (Reuse Subtree)
        .use_ucb1_tuned = 1, // ENABLED (Grandmaster)
        .use_tt = 1,         // ENABLED (Grandmaster)
        .use_solver = 1,     // ENABLED (Grandmaster)
        .use_progressive_bias = 1, // ENABLED (Grandmaster)
        .bias_constant = 3.0,
        .weights = { 
            .w_capture = 10.19, 
            .w_promotion = 4.96, 
            .w_advance = 0.00,  
            .w_center = 3.12, 
            .w_edge = 1.93, 
            .w_base = 3.07      
        }
    };
    
    while(1) {
        printf("\n=== Turno %d: %s ===\n", turn_count, 
               (state.current_player == WHITE) ? "BIANCO" : "NERO");

        // Safety Memory Check
        if (mcts_arena.offset > mcts_arena.size * 0.95) {
             printf("[WARNING] Arena almost full. Resetting tree.\n");
             arena_reset(&mcts_arena);
             root = NULL;
        }

        // Initialize Root if needed
        if (!root) {
             root = mcts_create_root(state, &mcts_arena, config);
        }

        MoveList list;
        generate_moves(&state, &list);
        
        // --- GAME OVER CHECKS ---
        if (list.count == 0) {
            printf("GAME OVER! Vince %s (nessuna mossa disponibile)\n", 
                   (state.current_player == WHITE) ? "NERO" : "BIANCO");
            break;
        }
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            printf("GAME OVER! Pareggio (40 mosse senza catture).\n");
            break;
        }

        Move chosen_move;

        // --- TURN LOGIC ---
        // --- TURN LOGIC ---
        if ((int)state.current_player == human_color) {
            // HUMAN TURN
            
            // Run MCTS for a brief moment to generate hints
            printf("Calcolo suggerimenti (1.0s)...\n");
            // Important: Pass NULL for out_new_root because we are NOT committing to a move yet.
            // We just want to expand the tree from the current root.
            mcts_search(root, &mcts_arena, 1.0, config, NULL, NULL);
            
            display_moves(&list, root);
            
            int chosen_idx = -1;
            while (chosen_idx < 0 || chosen_idx >= list.count) {
                printf("Inserisci il numero della mossa da giocare (1-%d): ", list.count);
                if (scanf("%d", &chosen_idx) == 1) {
                    chosen_idx--; // 1-based to 0-based
                } else {
                    while(getchar() != '\n'); // Clear buffer
                }
            }
            
            chosen_move = list.moves[chosen_idx];
            printf("Hai giocato: ");
            // Custom print to avoid missing symbol
            char buf_from[4], buf_to[4];
            int dest_idx = (chosen_move.length == 0) ? 1 : chosen_move.length;
            format_square(chosen_move.path[0], buf_from);
            format_square(chosen_move.path[dest_idx], buf_to);
            printf("%s -> %s\n", buf_from, buf_to);
             
            apply_move(&state, &chosen_move);

            // Advance Tree (Reuse)
            if (root) {
                Node *next = find_child_by_move(root, &chosen_move);
                if (next) {
                    // CRITICAL: Verify the state matches!
                    if (states_equal(&next->state, &state)) {
                        root = next;
                        // printf("[DEBUG] Tree Reused! New Root Visits: %d\n", root->visits);
                    } else {
                        printf("[WARNING] Tree Desync (Hash Collision or Logic Error)! Resetting Tree.\n");
                        root = NULL;
                    }
                } else {
                    // printf("[DEBUG] Move not in tree. New Root needed next turn.\n");
                    root = NULL; // Opponent played unexpected move (or tree not expanded enough)
                    // If we lose the tree, we might consider a reset if we want to reclaim memory, 
                    // but for simplicity we just start a new root in the same arena.
                }
            }
            
        } else {
            // AI TURN (GRANDMASTER)
             printf("Il Grandmaster sta pensando...\n");
            
            double time_limit = (state.current_player == WHITE) ? TIME_WHITE : TIME_BLACK;
            time_limit += 2.0;
            
            // Search
            chosen_move = mcts_search(root, &mcts_arena, time_limit, config, NULL, &root);
            // mcts_search with &root will update root to the chosen child automatically if implemented!
            // Wait, let's check mcts_search signature in mcts.c line 668:
            // Move mcts_search(..., Node **out_new_root) { ... *out_new_root = best_child; ... }
            // So passing &root updates 'root' to be the best child. Perfect.
            
            printf("Grandmaster gioca: ");
            // Custom print
            char buf_from[4], buf_to[4];
            int dest_idx = (chosen_move.length == 0) ? 1 : chosen_move.length;
            format_square(chosen_move.path[0], buf_from);
            format_square(chosen_move.path[dest_idx], buf_to);
            printf("%s -> %s\n", buf_from, buf_to);

            apply_move(&state, &chosen_move);
        }

        print_board(&state);
        turn_count++;
    }

    arena_free(&mcts_arena);
    return 0;
}