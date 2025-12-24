#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "game.h"

// =============================================================================
// MOVE GENERATION API
// =============================================================================

/**
 * Initialize pre-computed move lookup tables.
 * Must be called once before any move generation.
 */
void init_move_tables(void);

/**
 * Generates all legal moves for the current player.
 * Enforces Italian Checkers rules: mandatory captures and priority rules.
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void generate_moves(const GameState *s, MoveList *list);

/**
 * Generates all simple moves (non-captures) for the current player.
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void generate_simple_moves(const GameState *s, MoveList *list);

/**
 * Generates all capture moves (including chains) for the current player.
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void generate_captures(const GameState *s, MoveList *list);

/**
 * Checks if a specific square is under potential attack by the opponent.
 * Used for Safety Heuristics.
 * @param state The state AFTER our move.
 * @param square The square index to check.
 * @return 1 if attacked, 0 otherwise.
 */
int is_square_threatened(const GameState *state, int square);

#endif // MOVEGEN_H
