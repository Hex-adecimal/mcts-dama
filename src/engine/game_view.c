/**
 * @file game_view.c
 * @brief Debug visualization and printing functions for GameState and Moves.
 */

#include "dama/engine/game_view.h"
#include <stdio.h>

// --- Debug Utilities Implementation ---

void print_board(const GameState *s) {
    printf("\n   A B C D E F G H\n  +---------------+\n");
    for (int r = 7; r >= 0; r--) {
        printf("%d |", r + 1);
        for (int f = 0; f < 8; f++) {
            const int sq = SQUARE(r, f);
            char c = '.';
            if (TEST_BIT(s->piece[WHITE][PAWN], sq))      c = 'w';
            else if (TEST_BIT(s->piece[WHITE][LADY], sq)) c = 'W';
            else if (TEST_BIT(s->piece[BLACK][PAWN], sq)) c = 'b';
            else if (TEST_BIT(s->piece[BLACK][LADY], sq)) c = 'B';
            printf("%c|", c);
        }
        printf(" %d\n", r + 1);
    }
    printf("  +---------------+\n   A B C D E F G H\n\n");
    printf("Turn: %s | Hash: %llx | Draw: %d\n", 
           (s->current_player == WHITE) ? "WHITE" : "BLACK", 
           (unsigned long long)s->hash, s->moves_without_captures);
}

void print_coords(int sq) {
    printf("%c%d", COL(sq) + 'A', ROW(sq) + 1);
}

void print_move_list(const MoveList *list) {
    printf("Moves (%d):\n", list->count);
    for (int i = 0; i < list->count; i++) {
        printf("%d: ", i + 1);
        print_move_description(list->moves[i]);
        printf("\n");
    }
}

void print_move_description(const Move m) {
    print_coords(m.path[0]);
    if (m.length == 0) {
        printf("-");
        print_coords(m.path[1]);
    } else {
        for (int i = 0; i < m.length; i++) {
            printf("x");
            print_coords(m.path[i + 1]);
        }
    }
}
