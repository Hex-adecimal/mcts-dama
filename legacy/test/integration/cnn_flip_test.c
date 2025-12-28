/**
 * cnn_flip_test.c - Verify Board Flip Logic During Gameplay
 */

#include "game.h"
#include "movegen.h"
#include "cnn.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== CNN Flip Logic Test ===\n\n");
    
    zobrist_init();
    init_move_tables();
    
    // Load weights
    CNNWeights weights;
    cnn_init(&weights);
    if (cnn_load_weights(&weights, "bin/cnn_weights.bin") != 0) {
        printf("ERROR: Could not load weights\n");
        return 1;
    }
    
    // Initialize game
    GameState state;
    init_game(&state);
    
    printf("=== Initial Position (WHITE to move) ===\n");
    print_board(&state);
    
    // Get CNN output for WHITE
    CNNOutput out_white;
    cnn_forward(&weights, &state, &out_white);
    printf("CNN Value (WHITE perspective): %.4f\n", out_white.value);
    
    // Find top 3 policy moves for WHITE
    MoveList ml;
    generate_moves(&state, &ml);
    printf("Legal moves for WHITE: %d\n", ml.count);
    
    printf("Top 3 Policy Probs (WHITE):\n");
    for (int i = 0; i < ml.count && i < 3; i++) {
        int idx = cnn_move_to_index(&ml.moves[i], WHITE);
        printf("  Move %d->%d: idx=%d, prob=%.4f\n", 
               ml.moves[i].path[0], ml.moves[i].path[1], idx, out_white.policy[idx]);
    }
    
    // Make a move (first legal)
    apply_move(&state, &ml.moves[0]);
    
    printf("\n=== After White Move (BLACK to move) ===\n");
    print_board(&state);
    
    // Get CNN output for BLACK
    CNNOutput out_black;
    cnn_forward(&weights, &state, &out_black);
    printf("CNN Value (BLACK perspective): %.4f\n", out_black.value);
    
    // Find top 3 policy moves for BLACK
    generate_moves(&state, &ml);
    printf("Legal moves for BLACK: %d\n", ml.count);
    
    printf("Top 3 Policy Probs (BLACK):\n");
    for (int i = 0; i < ml.count && i < 3; i++) {
        int idx = cnn_move_to_index(&ml.moves[i], BLACK);
        printf("  Move %d->%d: idx=%d, prob=%.4f\n", 
               ml.moves[i].path[0], ml.moves[i].path[1], idx, out_black.policy[idx]);
    }
    
    // Test: Print what the network "sees" for Black
    printf("\n=== Canonical Board Visualization (What CNN sees for BLACK) ===\n");
    printf("Note: If correct, Black's pawns should appear at BOTTOM (rows 0-2)\n");
    printf("      and White's pawns should appear at TOP (rows 5-7)\n");
    
    float tensor[4 * 64];
    float player;
    cnn_encode_state(&state, tensor, &player);
    
    printf("Channel 0 (My Pawns - should be Black's pawns at bottom):\n");
    for (int r = 7; r >= 0; r--) {
        printf("%d: ", r);
        for (int c = 0; c < 8; c++) {
            int idx = 0 * 64 + r * 8 + c;
            printf("%c ", tensor[idx] > 0.5f ? 'P' : '.');
        }
        printf("\n");
    }
    
    printf("\nChannel 2 (Opponent Pawns - should be White's pawns at top):\n");
    for (int r = 7; r >= 0; r--) {
        printf("%d: ", r);
        for (int c = 0; c < 8; c++) {
            int idx = 2 * 64 + r * 8 + c;
            printf("%c ", tensor[idx] > 0.5f ? 'O' : '.');
        }
        printf("\n");
    }
    
    cnn_free(&weights);
    printf("\n=== Test Complete ===\n");
    return 0;
}
