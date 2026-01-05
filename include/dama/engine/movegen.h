/**
 * @file movegen.h
 * @brief Move Generation for Italian Checkers.
 *
 * Call movegen_init() once at startup, then use movegen_generate().
 */

#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "dama/engine/game.h"

// --- Public API ---

/**
 * @brief Initialize lookup tables. 
 *
 * Must call once before move generation.
 */
void movegen_init(void);

/**
 * @brief Main entry point: generates all legal moves with Italian priority filtering.
 *
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void movegen_generate(const GameState *s, MoveList *list);

/**
 * @brief Generate simple (non-capture) moves only.
 *
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void movegen_generate_simple(const GameState *s, MoveList *list);

/**
 * @brief Generate all capture moves (no Italian priority filtering).
 *
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void movegen_generate_captures(const GameState *s, MoveList *list);

/**
 * @brief Check if a square could be captured by opponent on their next move.
 *
 * @param state Pointer to the current GameState.
 * @param square The square index to check.
 * @return 1 if threatened, 0 otherwise.
 */
int movegen_is_square_threatened(const GameState *state, int square);

#endif /* MOVEGEN_H */
