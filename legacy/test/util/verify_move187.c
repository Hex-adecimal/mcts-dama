/**
 * verify_move187.c - Test mandatory capture in position 187
 */

#include "game.h"
#include "movegen.h"
#include <stdio.h>
#include <stdlib.h>

// Simple inline board print
static void show_board(const GameState *s) {
    printf("\n   A B C D E F G H\n");
    printf("  +---------------+\n");
    for (int row = 7; row >= 0; row--) {
        printf("%d |", row + 1);
        for (int col = 0; col < 8; col++) {
            int sq = row * 8 + col;
            char c = '.';
            if (s->white_pieces & (1ULL << sq)) c = 'w';
            if (s->white_ladies & (1ULL << sq)) c = 'W';
            if (s->black_pieces & (1ULL << sq)) c = 'b';
            if (s->black_ladies & (1ULL << sq)) c = 'B';
            printf("%c|", c);
        }
        printf(" %d\n", row + 1);
    }
    printf("  +---------------+\n");
    printf("   A B C D E F G H\n");
    printf("\nTurn: %s\n", s->current_player == WHITE ? "WHITE" : "BLACK");
}

int main(void) {
    zobrist_init();
    init_move_tables();
    
    printf("=== Verify Move 187 Position ===\n\n");
    
    // Recreate the board from Move 187:
    // 8 |B|.|.|.|.|.|w|.| 8   <- B=Black dama at A8 (sq 56), w=white pawn at G8 (sq 62)
    // 7 |.|.|.|.|.|.|.|.| 7
    // 6 |w|.|.|.|.|.|.|.| 6   <- w=white pawn at A6 (sq 40)
    // 5 |.|.|.|w|.|.|.|.| 5   <- w=white pawn at D5 (sq 35)
    // 4 |.|.|.|.|B|.|B|.| 4   <- B at E4 (sq 28), B at G4 (sq 30)
    // 3 |.|.|.|.|.|.|.|B| 3   <- B at H3 (sq 23)
    // 2 |b|.|.|.|B|.|.|.| 2   <- b=black pawn at A2 (sq 8), B at E2 (sq 12)
    // 1 |.|.|.|B|.|.|.|.| 1   <- B at D1 (sq 3)
    
    GameState state = {0};
    state.current_player = WHITE;
    
    // White pieces (3 total)
    state.white_pieces = (1ULL << 62) | (1ULL << 40) | (1ULL << 35);  // G8, A6, D5
    state.white_ladies = 0;  // No white ladies
    
    // Black pieces (7 total) - 1 pawn, 6 damas
    state.black_pieces = (1ULL << 8);  // A2 (pawn)
    state.black_ladies = (1ULL << 56) | (1ULL << 28) | (1ULL << 30) | (1ULL << 23) | (1ULL << 12) | (1ULL << 3);
    // A8, E4, G4, H3, E2, D1
    
    printf("Reconstructed Position (WHITE to move):\n");
    show_board(&state);
    
    // Generate moves
    MoveList ml;
    generate_moves(&state, &ml);
    
    printf("\nGenerated Moves for WHITE (%d total):\n", ml.count);
    
    for (int i = 0; i < ml.count; i++) {
        Move *m = &ml.moves[i];
        if (m->length > 0) {
            printf("  CAPTURE: %d", m->path[0]);
            for (int k = 1; k <= m->length; k++) {
                printf("->%d", m->path[k]);
            }
            printf(" (captures %d pieces)\n", m->length);
        } else {
            printf("  Simple: %d->%d\n", m->path[0], m->path[1]);
        }
    }
    
    // Check if there are any captures
    MoveList captures_only;
    captures_only.count = 0;
    generate_captures(&state, &captures_only);
    printf("\nCapture moves found: %d\n", captures_only.count);
    
    if (captures_only.count == 0 && ml.count > 0 && ml.moves[0].length == 0) {
        printf("\n*** POTENTIAL BUG: Only simple moves generated, no captures! ***\n");
        printf("    But: White pawns cannot capture black damas (Italian rule)\n");
        printf("    Checking if there are any pawns white can capture...\n");
    }
    
    return 0;
}
