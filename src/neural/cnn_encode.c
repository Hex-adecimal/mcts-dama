/**
 * cnn_encode.c - Game State Encoding for CNN Input
 * 
 * Extracted from cnn_core.c for better modularity.
 * Contains: encode_state_channels_canonical, cnn_encode_state, cnn_encode_sample
 */

#include "dama/neural/cnn.h"
#include <string.h>

// =============================================================================
// STATE ENCODING (CANONICAL FORM)
// =============================================================================

/**
 * Encode state in CANONICAL FORM.
 * 
 * Board is always represented from the current player's perspective:
 * - Channel 0: "my" pawns
 * - Channel 1: "my" ladies  
 * - Channel 2: "opponent" pawns
 * - Channel 3: "opponent" ladies
 *
 * If it's Black's turn, the board is flipped vertically so Black "starts from bottom".
 */
void encode_state_channels_canonical(const GameState *state, float *tensor, int channel_offset) {
    int is_white = (state->current_player == WHITE);
    
    for (int sq = 0; sq < 64; sq++) {
        int target_sq = is_white ? sq : flip_square(sq);
        
        if (is_white) {
            // White's perspective: White = "my", Black = "opponent"
            if (check_bit(state->piece[WHITE][PAWN], sq)) tensor[channel_offset * 64 + target_sq] = 1.0f;
            if (check_bit(state->piece[WHITE][LADY], sq)) tensor[(channel_offset + 1) * 64 + target_sq] = 1.0f;
            if (check_bit(state->piece[BLACK][PAWN], sq)) tensor[(channel_offset + 2) * 64 + target_sq] = 1.0f;
            if (check_bit(state->piece[BLACK][LADY], sq)) tensor[(channel_offset + 3) * 64 + target_sq] = 1.0f;
        } else {
            // Black's perspective: Black = "my", White = "opponent" (flipped board)
            if (check_bit(state->piece[BLACK][PAWN], sq)) tensor[channel_offset * 64 + target_sq] = 1.0f;
            if (check_bit(state->piece[BLACK][LADY], sq)) tensor[(channel_offset + 1) * 64 + target_sq] = 1.0f;
            if (check_bit(state->piece[WHITE][PAWN], sq)) tensor[(channel_offset + 2) * 64 + target_sq] = 1.0f;
            if (check_bit(state->piece[WHITE][LADY], sq)) tensor[(channel_offset + 3) * 64 + target_sq] = 1.0f;
        }
    }
}

// Wrapper for compatibility (uses canonical form)
void encode_state_channels(const GameState *state, float *tensor, int channel_offset) {
    encode_state_channels_canonical(state, tensor, channel_offset);
}

/**
 * Encode a single game state for CNN input.
 */
void cnn_encode_state(const GameState *state, float *tensor, float *player) {
    memset(tensor, 0, CNN_INPUT_CHANNELS * 64 * sizeof(float));
    encode_state_channels_canonical(state, tensor, 0);
    *player = 1.0f;  // Always "my turn" in canonical form
}

/**
 * Encode a training sample with history for CNN input.
 */
void cnn_encode_sample(const TrainingSample *sample, float *tensor, float *player) {
    memset(tensor, 0, CNN_INPUT_CHANNELS * 64 * sizeof(float));
    encode_state_channels_canonical(&sample->state, tensor, 0);
    // Note: History states need same canonical transform as main state
    
    // Temporarily set history player to match main state for consistent encoding
    GameState hist0 = sample->history[0];
    GameState hist1 = sample->history[1];
    hist0.current_player = sample->state.current_player;
    hist1.current_player = sample->state.current_player;
    
    encode_state_channels_canonical(&hist0, tensor, 4);
    encode_state_channels_canonical(&hist1, tensor, 8);
    *player = 1.0f;  // Always 1.0 in canonical form (it's always "my turn")
}
