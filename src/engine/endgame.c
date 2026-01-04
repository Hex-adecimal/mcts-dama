/**
 * endgame.c - Endgame Position Generation
 * 
 * Generates random but valid endgame positions for training data.
 * Positions are balanced to avoid trivially won/lost games.
 */

#include "dama/engine/endgame.h"
#include <string.h>

// =============================================================================
// HELPERS
// =============================================================================

// Shuffle array of dark square indices using Fisher-Yates
static void shuffle_squares(int *arr, int n, RNG *rng) {
    for (int i = n - 1; i > 0; i--) {
        int j = rng_u32(rng) % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// Check if square is in promotion zone for given color
static int is_promotion_zone(int sq, Color color) {
    int row = ROW(sq);
    return (color == WHITE && row == 7) || (color == BLACK && row == 0);
}

// Compute hash for the state (reusing logic from game.c)
static uint64_t compute_hash_for_endgame(const GameState *s) {
    extern uint64_t zobrist_keys[NUM_COLORS][NUM_PIECE_TYPES][NUM_SQUARES];
    extern uint64_t zobrist_black_move;
    
    uint64_t h = 0;
    for (Color c = WHITE; c <= BLACK; c++) {
        for (Piece t = PAWN; t <= LADY; t++) {
            Bitboard bb = s->piece[c][t];
            while (bb) {
                int sq = __builtin_ctzll(bb);
                h ^= zobrist_keys[c][t][sq];
                POP_LSB(bb);
            }
        }
    }
    if (s->current_player == BLACK) h ^= zobrist_black_move;
    return h;
}

// Place a piece on the board
static void place_piece(GameState *s, Color c, Piece t, int sq) {
    SET_BIT(s->piece[c][t], sq);
}

// =============================================================================
// ENDGAME SETUP
// =============================================================================

/**
 * Setup endgame with specified piece counts.
 * Returns 1 if successful, 0 if no valid position found.
 */
static int setup_endgame_with_counts(
    GameState *state,
    RNG *rng,
    int white_ladies, int white_pawns,
    int black_ladies, int black_pawns
) {
    // Reset state
    memset(state, 0, sizeof(GameState));
    state->current_player = (rng_u32(rng) % 2 == 0) ? WHITE : BLACK;
    state->moves_without_captures = 0;
    
    // Create shuffled list of available squares
    int available[NUM_DARK_SQUARES];
    for (int i = 0; i < NUM_DARK_SQUARES; i++) {
        available[i] = DARK_SQUARES[i];
    }
    shuffle_squares(available, NUM_DARK_SQUARES, rng);
    
    int idx = 0;
    int total_needed = white_ladies + white_pawns + black_ladies + black_pawns;
    
    if (total_needed > NUM_DARK_SQUARES) return 0;
    
    // Place white ladies (avoid promotion zone for variety)
    for (int i = 0; i < white_ladies && idx < NUM_DARK_SQUARES; ) {
        int sq = available[idx++];
        // Ladies can be anywhere (already promoted)
        place_piece(state, WHITE, LADY, sq);
        i++;
    }
    
    // Place white pawns (NOT in promotion zone, would auto-promote)
    for (int i = 0; i < white_pawns && idx < NUM_DARK_SQUARES; ) {
        int sq = available[idx++];
        if (is_promotion_zone(sq, WHITE)) continue; // Skip, find another
        place_piece(state, WHITE, PAWN, sq);
        i++;
    }
    
    // Place black ladies
    for (int i = 0; i < black_ladies && idx < NUM_DARK_SQUARES; ) {
        int sq = available[idx++];
        place_piece(state, BLACK, LADY, sq);
        i++;
    }
    
    // Place black pawns (NOT in promotion zone)
    for (int i = 0; i < black_pawns && idx < NUM_DARK_SQUARES; ) {
        int sq = available[idx++];
        if (is_promotion_zone(sq, BLACK)) continue;
        place_piece(state, BLACK, PAWN, sq);
        i++;
    }
    
    // Compute hash
    state->hash = compute_hash_for_endgame(state);
    
    // Validate: current player must have legal moves
    return position_has_moves(state);
}

int setup_random_endgame(GameState *state, RNG *rng) {
    // Select endgame type with weighted probabilities
    int type_roll = rng_u32(rng) % 100;
    EndgameType type;
    
    if (type_roll < 15) {
        type = EG_2D_VS_2D;           // 15%: Complex queen endgame
    } else if (type_roll < 35) {
        type = EG_2D_VS_1D1P;         // 20%: Slight imbalance
    } else if (type_roll < 55) {
        type = EG_1D2P_VS_1D2P;       // 20%: Mixed endgame
    } else if (type_roll < 70) {
        type = EG_2D_VS_1D;           // 15%: Material advantage
    } else if (type_roll < 85) {
        type = EG_1D1P_VS_1D1P;       // 15%: Simple
    } else {
        type = EG_RANDOM_SPARSE;      // 15%: Random sparse
    }
    
    int max_attempts = 10;
    
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        int success = 0;
        
        switch (type) {
            case EG_2D_VS_2D:
                success = setup_endgame_with_counts(state, rng, 2, 0, 2, 0);
                break;
            case EG_2D_VS_1D1P:
                // Randomly decide who has advantage
                if (rng_u32(rng) % 2 == 0) {
                    success = setup_endgame_with_counts(state, rng, 2, 0, 1, 1);
                } else {
                    success = setup_endgame_with_counts(state, rng, 1, 1, 2, 0);
                }
                break;
            case EG_1D2P_VS_1D2P:
                success = setup_endgame_with_counts(state, rng, 1, 2, 1, 2);
                break;
            case EG_2D_VS_1D:
                if (rng_u32(rng) % 2 == 0) {
                    success = setup_endgame_with_counts(state, rng, 2, 0, 1, 0);
                } else {
                    success = setup_endgame_with_counts(state, rng, 1, 0, 2, 0);
                }
                break;
            case EG_1D1P_VS_1D1P:
                success = setup_endgame_with_counts(state, rng, 1, 1, 1, 1);
                break;
            case EG_RANDOM_SPARSE:
                // 2-4 pieces per side, random distribution
                {
                    int w_pieces = 2 + (rng_u32(rng) % 3);  // 2-4
                    int b_pieces = 2 + (rng_u32(rng) % 3);  // 2-4
                    int w_ladies = rng_u32(rng) % (w_pieces + 1);
                    int w_pawns = w_pieces - w_ladies;
                    int b_ladies = rng_u32(rng) % (b_pieces + 1);
                    int b_pawns = b_pieces - b_ladies;
                    success = setup_endgame_with_counts(state, rng, 
                        w_ladies, w_pawns, b_ladies, b_pawns);
                }
                break;
            default:
                success = setup_endgame_with_counts(state, rng, 1, 1, 1, 1);
        }
        
        if (success) return 1;
    }
    
    // Fallback: simple endgame that always works
    return setup_endgame_with_counts(state, rng, 1, 0, 1, 0);
}
