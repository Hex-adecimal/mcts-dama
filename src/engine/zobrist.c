/**
 * @file zobrist.c
 * @brief Zobrist Hashing Implementation.
 */

#include "dama/engine/zobrist.h"

uint64_t zobrist_keys[NUM_COLORS][NUM_PIECE_TYPES][NUM_SQUARES];
uint64_t zobrist_black_move;

static uint64_t rand64(void) {
    static uint64_t seed = ZOBRIST_SEED;
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
}

void zobrist_init(void) {
    for (Color c = WHITE; c <= BLACK; c++) {
        for (Piece t = PAWN; t <= LADY; t++) {
            for (int sq = 0; sq < NUM_SQUARES; sq++) {
                zobrist_keys[c][t][sq] = rand64();
            }
        }
    }
    zobrist_black_move = rand64();
}

uint64_t zobrist_compute_hash(const GameState *s) {
    uint64_t h = 0;
    for (Color c = WHITE; c <= BLACK; c++) {
        for (Piece t = PAWN; t <= LADY; t++) {
            Bitboard bb = s->piece[c][t];
            while (bb) {
                const int sq = __builtin_ctzll(bb);
                h ^= zobrist_keys[c][t][sq];
                POP_LSB(bb);
            }
        }
    }
    if (s->current_player == BLACK) h ^= zobrist_black_move;
    return h;
}
