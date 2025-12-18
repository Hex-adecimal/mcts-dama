#include "debug.h"
#include "mcts.h"
#include <stdio.h>
#include <stdlib.h> // for qsort

/**
 * Calculates average UCB value of all children of the root.
 * Useful for debugging FPU tuning.
 */
double mcts_get_avg_root_ucb(Node *root, MCTSConfig config) {
    if (!root || root->num_children == 0) return 0.0;
    
    double total_ucb = 0.0;
    int count = 0;
    
    for (int i = 0; i < root->num_children; i++) {
        double val = calculate_ucb1_score(root->children[i], config);
        // exclude infinite values (unvisited) to avoid skewing average
        if (val < 1e8) { 
            total_ucb += val;
            count++;
        }
    }
    
    return (count > 0) ? (total_ucb / count) : 0.0;
}

/**
 * Prints the algebraic coordinates (e.g., "A1") of a square index.
 */
void print_coords(int square_idx) {
    int rank = (square_idx / 8) + 1;
    char file = (square_idx % 8) + 'A';
    printf("%c%d", file, rank);
}

/**
 * Prints the current board state to the console.
 */
void print_board(const GameState *state) {
    printf("\n   A B C D E F G H\n");
    printf("  +---------------+\n");

    for (int rank = 7; rank >= 0; rank--) {
        printf("%d |", rank + 1);
        
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            char c = '.';

            if (check_bit(state->white_pieces, sq)) c = 'w';
            else if (check_bit(state->black_pieces, sq)) c = 'b';
            else if (check_bit(state->white_ladies, sq)) c = 'W';
            else if (check_bit(state->black_ladies, sq)) c = 'B';
            
            printf("%c|", c);
        }
        printf(" %d\n", rank + 1);
    }
    printf("  +---------------+\n");
    printf("   A B C D E F G H\n\n");
    
    printf("Turn: %s\n", (state->current_player == WHITE) ? "WHITE" : "BLACK");
    printf("White pieces bitboard: %llu\n", state->white_pieces);
}

/**
 * Prints the list of generated moves to the console.
 */
void print_move_list(MoveList *list) {
    printf("------------------------------------------------\n");
    printf("Found %d possible moves:\n", list->count);
    
    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        printf("%d) ", i + 1);
        
        if (m.length == 0) {
            print_coords(m.path[0]);
            printf(" -> ");
            print_coords(m.path[1]);
        } else {
            print_coords(m.path[0]);
            for (int j = 0; j < m.length; j++) {
                printf(" x ");
                print_coords(m.captured_squares[j]);
                printf(" -> ");
                print_coords(m.path[j+1]);
            }
            printf(" (Len: %d, Ladies: %d, First Lady: %d, Is Lady: %d)", 
                   m.length, m.captured_ladies_count, m.first_captured_is_lady, m.is_lady_move);
        }
        printf("\n");
    }
    printf("------------------------------------------------\n");
}

/**
 * Prints a human-readable description of a move.
 */
void print_move_description(Move m) {
    print_coords(m.path[0]);
    
    if (m.length == 0) {
        printf("-");
        print_coords(m.path[1]);
    } else {
        for (int i=0; i<m.length; i++) {
            printf("x");
            print_coords(m.path[i+1]); // Capture destination
        }
    }
}

// Compare function for qsort
int compare_nodes_visits_desc(const void *a, const void *b) {
    Node *nodeA = *(Node **)a;
    Node *nodeB = *(Node **)b;
    return nodeB->visits - nodeA->visits; // Descending order
}

/**
 * Prints debug info for root children, SORTED by visit count.
 */
void print_mcts_stats_sorted(Node *root) {
    if (!root || root->num_children == 0) {
        printf("No children statistics available.\n");
        return;
    }

    // Create a temporary array of pointers to sort without modifying the tree order
    Node **sorted_children = malloc(root->num_children * sizeof(Node*));
    if (!sorted_children) return;

    for (int i=0; i<root->num_children; i++) {
        sorted_children[i] = root->children[i];
    }

    qsort(sorted_children, root->num_children, sizeof(Node*), compare_nodes_visits_desc);

    printf("--- Root Children Stats (Sorted by Visits) ---\n");
    for (int i = 0; i < root->num_children; i++) {
        Node *child = sorted_children[i];
        printf("Move: ");
        print_move_description(child->move_from_parent);
        
        double win_rate = (child->visits > 0) ? (child->score / child->visits) : 0.0;
        printf(" | Visits: %d | Score: %.1f | WinRate: %.1f%% | Status: %d\n", 
               child->visits, child->score, win_rate * 100.0, child->status);
    }
    printf("----------------------------------------------\n");

    free(sorted_children);
}
