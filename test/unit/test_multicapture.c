#include "game.h"
#include "mcts.h"
#include <stdio.h>
#include <assert.h>

// Helper to set specific bits
void set_piece(Bitboard *b, int rank, int file) {
    int sq = rank * 8 + file;
    *b |= (1ULL << sq);
}

void test_multicapture_bug() {
    printf("Testing Multi-Capture Bug...\n");
    GameState s;
    init_game(&s);
    
    // Clear board
    s.white_pieces = 0; s.white_ladies = 0;
    s.black_pieces = 0; s.black_ladies = 0;
    
    s.current_player = BLACK;

    // --- RECONSTRUCTING BOARD from Turn 10 ---
    // Black (Mover)
    // Rank 7 (Row 8): A8, C8, E8, G8 (Indices: 56, 58, 60, 62)
    set_piece(&s.black_pieces, 7, 0); // A8
    set_piece(&s.black_pieces, 7, 2); // C8
    set_piece(&s.black_pieces, 7, 4); // E8
    set_piece(&s.black_pieces, 7, 6); // G8
    
    // Rank 6 (Row 7): B7, D7, H7 (Indices: 49, 51, 55)
    set_piece(&s.black_pieces, 6, 1); // B7 <--- THE MOVER
    set_piece(&s.black_pieces, 6, 3); // D7
    set_piece(&s.black_pieces, 6, 7); // H7
    
    // Rank 5 (Row 6): A6, D6, F6 (Indices: 40, 43, 45)
    set_piece(&s.black_pieces, 5, 0); // A6
    set_piece(&s.black_pieces, 5, 3); // D6
    set_piece(&s.black_pieces, 5, 5); // F6
    
    // White (Victims)
    // Rank 5 (Row 6): C6 (Index 42) (Promoted? No 'w' small)
    // Wait, row 6 in ASCII: "6 |b|.|w|.|b|.|b|.|". 
    // Col A(b), B(.), C(w). So C6 is White Pawn.
    set_piece(&s.white_pieces, 5, 2); // C6 <--- Victim 1

    // Rank 3 (Row 4): C4, G4 (Indices: 26, 30)
    // "4 |.|.|w|.|.|.|w|.|" -> C4(w), G4(w).
    set_piece(&s.white_pieces, 3, 2); // C4 <--- Victim 2
    set_piece(&s.white_pieces, 3, 6); // G4
    
    // Rank 2 (Row 3): H3 (Index 23)
    set_piece(&s.white_pieces, 2, 7); // H3
    
    // Rank 1 (Row 2): A2, C2, G2 (Indices: 8, 10, 14)
    set_piece(&s.white_pieces, 1, 0);
    set_piece(&s.white_pieces, 1, 2);
    set_piece(&s.white_pieces, 1, 6);
    
    // Rank 0 (Row 1): B1, D1, F1, H1 (Indices: 1, 3, 5, 7)
    set_piece(&s.white_pieces, 0, 1);
    set_piece(&s.white_pieces, 0, 3);
    set_piece(&s.white_pieces, 0, 5);
    set_piece(&s.white_pieces, 0, 7);
    
    print_board(&s);
    
    // GENERATE MOVES
    MoveList list;
    generate_moves(&s, &list);
    
    printf("Generated %d moves:\n", list.count);
    int found_multicapture = 0;
    
    for(int i=0; i<list.count; i++) {
        Move m = list.moves[i];
        printf("Move %d: Length %d. ", i, m.length);
        char buf1[4], buf2[4];
        // print path
        for(int k=0; k<=m.length; k++) {
             printf("%d", m.path[k]);
             if(k < m.length) printf("->");
        }
        printf("\n");
        
        // B7 is 49. C6 is 42. D5 is 35. C4 is 26. B3 is 17.
        // Expected Path: 49 -> 35 -> 17. Length 2.
        if (m.path[0] == 49 && m.length == 2 && m.path[2] == 17) {
            found_multicapture = 1;
        }
    }
    
    if (found_multicapture) {
        printf("[PASS] Multi-capture B7->D5->B3 found.\n");
    } else {
        printf("[FAIL] Prioritized Multi-capture NOT found.\n");
    }
}

int main() {
    zobrist_init();
    test_multicapture_bug();
    return 0;
}
