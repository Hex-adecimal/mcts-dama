#include <stdio.h>
#include <string.h>
#include "game.h"
#include "movegen.h"
#include "cnn.h"

// Mock cnn.c internals if needed, or link against it
// We need to verify if Black state is flipped correctly

void print_tensor_board(float *tensor, int channel) {
    printf("Channel %d:\n", channel);
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int idx = channel * 64 + r * 8 + c;
            printf("%.0f ", tensor[idx]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Rotation Verification ===\n");
    
    // Case 1: White at start
    GameState white_start;
    init_game(&white_start);
    // Move White piece from (2,1)[17] to (3,0)[24] ( Down Left? +7 )
    // Index 17 is Row 2, Col 1.
    // Index 24 is Row 3, Col 0.
    // Diff +7.
    
    float tensor[12 * 64];
    float player;
    cnn_encode_state(&white_start, tensor, &player);
    
    // Check White Pawn channel (0)
    // Should see pieces at Row 0, 1, 2. (Visual top)
    // printf("White Start (White Perspective):\n");
    // print_tensor_board(tensor, 0);
    
    // Case 2: Black at start
    GameState black_start;
    init_game(&black_start);
    black_start.current_player = BLACK;
    
    // Setup specific Black piece at 40 (Row 5, Col 0).
    // In canonical view (flipped), 40 becomes 63-40 = 23 (Row 2, Col 7).
    
    // Clear board for clarity
    black_start.white_pieces = 0;
    black_start.black_pieces = (1ULL << 40); 
    
    // Encode
    cnn_encode_state(&black_start, tensor, &player);
    
    printf("\nTEST: Black Piece at 40 (Row 5).\n");
    printf("Expected Canonical: 63-40 = 23 (Row 2, Col 7).\n");
    
    // Black pawns are usually Channel 2?
    // In canonical, "My Pawns" is Channel 0.
    // Since CurrentPlayer is Black, Black Pawns are "My Pawns".
    
    int found_at = -1;
    for (int i=0; i<64; i++) {
        if (tensor[i] > 0.5f) found_at = i;
    }
    
    printf("Found in Channel 0 at: %d\n", found_at);
    
    if (found_at == 23) printf("✓ State Rotation CORRECT\n");
    else printf("X State Rotation FAILED\n");
    
    // TEST MOVE INDEXING
    // Black moves 40 -> 35 (-5? No. 40 is Row 5 Col 0. 35 is Row 4 Col 3. Knight jump)
    // Valid move: 40 (Row 5 Col 0) -> 35?
    // 40: 000000 101000...
    // Valid moves from 40: To 33 (-7)? No 40 is edge?
    // Bit 40. Row 5. Col 0.
    // Moves: Down-Right (+9) -> 49 (Row 6 Col 1).
    // Up-Right (-7) -> 33 (Row 4 Col 1).
    // Up-Left (-9) -> Invalid (Col 0).
    
    // Let's test move 40 -> 33 (-7).
    Move m;
    m.path[0] = 40;
    m.length = 0;
    m.path[1] = 33;
    
    int idx = cnn_move_to_index(&m, BLACK);
    
    // Manually calculate expected:
    // Flipped From: 63 - 40 = 23.
    // Flipped To: 63 - 33 = 30.
    // Diff: 30 - 23 = +7.
    // Dir +7 (Down-Left) -> 1?
    // get_move_direction:
    // +7 -> returns 1.
    // Idx = 23 * 8 + 1 = 184 + 1 = 185.
    
    printf("\nTEST: Black Move 40->33 (-7).\n");
    printf("Expected Index: 23*8 + 1 = 185.\n");
    printf("Calculated Index: %d\n", idx);
    
    if (idx == 185) printf("✓ Move Rotation CORRECT\n");
    else printf("X Move Rotation FAILED\n");
    
    return 0;
}
