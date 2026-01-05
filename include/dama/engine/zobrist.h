/**
 * @file zobrist.h
 * @brief Zobrist hashing for Italian Checkers game state.
 */

#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "dama/engine/game.h"
#include <stdint.h>

// Globals
extern uint64_t zobrist_keys[NUM_COLORS][NUM_PIECE_TYPES][NUM_SQUARES];
extern uint64_t zobrist_black_move;

/**
 * @brief Initialize Zobrist keys with random values.
 *
 * Must be called once at startup.
 */
void zobrist_init(void);

/**
 * @brief Compute the full Zobrist hash of a game state.
 *
 * @param s Pointer to the GameState.
 * @return The 64-bit Zobrist hash.
 */
uint64_t zobrist_compute_hash(const GameState *s);

#endif /* ZOBRIST_H */
