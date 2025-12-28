/**
 * debug_flip.c
 * Unit test to verify Canonical Flip logic consistency.
 */

#include "game.h"
#include "cnn.h"
#include "movegen.h"
#include <stdio.h>
#include <string.h>

void print_tensor_active(float *tensor) {
    printf("Active Tensor Inputs:\n");
    for (int c = 0; c < 4; c++) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                int idx = c * 64 + y * 8 + x;
                if (tensor[idx] > 0.0f) {
                    printf("  Channel %d: (%d, %d) [Idx %d] -> Canonical Square %d\n", c, y, x, idx, y*8+x);
                }
            }
        }
    }
}

int main(void) {
    printf("=== DEBUG FLIP LOGIC ===\n");
    
    init_move_tables();
    
    GameState state;
    memset(&state, 0, sizeof(GameState));
    
    // Setup: Black Piece at A6 (Index 40)
    // A6 is Row 5, Col 0.
    state.black_pieces = (1ULL << 40);
    state.current_player = BLACK;
    
    printf("State: Black Piece at A6 (Index 40). To Move: BLACK.\n");
    
    // 1. Check Tensor Encoding
    float tensor[256];
    float player;
    cnn_encode_state(&state, tensor, &player);
    
    // Expected: Black piece (My Pieces -> Ch 0)
    // Flip: 63 - 40 = 23.
    // 23 is Row 2, Col 7 (H3).
    // Tensor Ch 0, (2, 7).
    print_tensor_active(tensor);
    
    // 2. Check Move Mapping
    // Move A6 (40) -> B5 (33). SE Direction.
    Move m = {0};
    m.path[0] = 40;
    m.path[1] = 33; // 40-7 = 33
    m.length = 0;
    
    printf("Move: A6 (40) -> B5 (33). (Real SE)\n");
    
    int idx = cnn_move_to_index(&m, BLACK);
    
    // Expected:
    // Flip From: 63-40 = 23 (H3).
    // Flip To: 63-33 = 30 (G4).
    // Diff: +7. Direction NW (1).
    // Index: 23 * 8 + 1 = 184 + 1 = 185.
    
    printf("Computed Policy Index: %d\n", idx);
    
    int from_part = idx / 8;
    int dir_part = idx % 8;
    
    printf("  -> Mapped From: %d\n", from_part);
    printf("  -> Mapped Dir: %d\n", dir_part);
    
    if (from_part == 23 && dir_part == 1) {
        printf("SUCCESS: Move maps to From 23 (H3) Dir NW.\n");
        printf("Matches Tensor input at 23.\n");
    } else {
        printf("FAILURE: Mismatch!\n");
    }
    
    return 0;
}
