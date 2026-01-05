/**
 * @file game.c
 * @brief Core Game State Management for Italian Checkers.
 */

#include "dama/engine/game.h"
#include "dama/common/debug.h"
#include "dama/engine/zobrist.h"


// --- Game Initialization ---
void init_game(GameState *state) {
    DBG_NOT_NULL(state);
    state->piece[WHITE][PAWN] = INITIAL_WHITE_PAWNS;
    state->piece[BLACK][PAWN] = INITIAL_BLACK_PAWNS;
    state->piece[WHITE][LADY] = 0;
    state->piece[BLACK][LADY] = 0;
    state->current_player = WHITE;
    state->moves_without_captures = 0;
    state->hash = zobrist_compute_hash(state);
}

// --- Move Execution ---
static void perform_movement(GameState *s, const int from, const int to) {
    DBG_VALID_SQ(from);
    DBG_VALID_SQ(to);

    const Color us = s->current_player;
    const Bitboard from_to_mask = BIT(from) | BIT(to);
    
    const Piece type = TEST_BIT(s->piece[us][LADY], from) ? LADY : PAWN;
    s->piece[us][type] ^= from_to_mask;
    
    if (type == PAWN && TEST_BIT(PROM_RANKS[us], to)) {
        CLEAR_BIT(s->piece[us][PAWN], to);
        SET_BIT(s->piece[us][LADY], to);
    }
}

void apply_move(GameState *state, const Move *move) {
    DBG_NOT_NULL(state);
    DBG_NOT_NULL(move);

    const Color us = state->current_player;
    const Color them = us ^ 1;
    
    const int from = move->path[0];
    const int to = (move->length == 0) ? move->path[1] : move->path[move->length];

    state->hash ^= zobrist_keys[us][move->is_lady_move][from];
    perform_movement(state, from, to);
    
    const Piece piece_now = TEST_BIT(state->piece[us][LADY], to) ? LADY : PAWN;
    state->hash ^= zobrist_keys[us][piece_now][to];

    if (move->length > 0) {
        for (int i = 0; i < move->length; i++) {
            const int cap_sq = move->captured_squares[i];
            const Piece cap_type = TEST_BIT(state->piece[them][LADY], cap_sq) ? LADY : PAWN;
            state->hash ^= zobrist_keys[them][cap_type][cap_sq];
            CLEAR_BIT(state->piece[them][cap_type], cap_sq);
        }
        state->moves_without_captures = 0;
    } else {
        if (state->piece[WHITE][LADY] || state->piece[BLACK][LADY])
            state->moves_without_captures++;
        else
            state->moves_without_captures = 0;
    }

    state->current_player = them;
    state->hash ^= zobrist_black_move;
}