/**
 * =============================================================================
 * movegen.h - Move Generation Interface for Italian Checkers
 * =============================================================================
 * 
 * This header provides the API for legal move generation. The implementation
 * handles all Italian Checkers rules including:
 * 
 *   - Mandatory capture rule (must capture if possible)
 *   - Capture chain generation (multiple sequential jumps)
 *   - Italian priority rules for selecting the "best" capture
 *   - Pawn promotion when reaching the back rank
 * 
 * Usage:
 *   1. Call init_move_tables() once at program startup
 *   2. Use generate_moves() to get all legal moves for a position
 * 
 * Italian Checkers Priority Rules (in order of importance):
 *   1. Longest capture sequence wins
 *   2. Ladies capture before pawns (if equal length)
 *   3. Prefer capturing more Ladies
 *   4. Prefer capturing a Lady first
 * 
 * =============================================================================
 */

#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "game.h"

/* =============================================================================
 * INITIALIZATION
 * ============================================================================= */

/**
 * Initialize pre-computed move lookup tables.
 * 
 * MUST be called once at program startup before any move generation.
 * Pre-computes valid move targets for all squares and piece types.
 * Safe to call multiple times (subsequent calls are no-ops).
 */
void init_move_tables(void);

/* =============================================================================
 * MOVE GENERATION API
 * ============================================================================= */

/**
 * Generate all legal moves for the current player.
 * 
 * This is the main entry point for move generation. Implements the
 * mandatory capture rule: if captures are available, only capture
 * moves are returned (filtered by Italian priority rules).
 * 
 * @param s     Current game state
 * @param list  Move list to populate (will be cleared first)
 */
void generate_moves(const GameState *s, MoveList *list);

/**
 * Generate all simple (non-capture) moves.
 * 
 * Returns diagonal moves for pawns (forward only) and ladies (all directions).
 * Does NOT check if captures are available - use generate_moves() for
 * proper rule enforcement.
 * 
 * @param s     Current game state
 * @param list  Move list to populate (appends to existing)
 */
void generate_simple_moves(const GameState *s, MoveList *list);

/**
 * Generate all capture moves (including multi-jump chains).
 * 
 * Finds all possible capture sequences using recursive depth-first search.
 * Does NOT apply Italian priority filtering - use generate_moves() for
 * properly filtered results.
 * 
 * @param s     Current game state
 * @param list  Move list to populate (appends to existing)
 */
void generate_captures(const GameState *s, MoveList *list);

/* =============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================= */

/**
 * Check if a square is threatened by the opponent.
 * 
 * Useful for AI evaluation heuristics. Determines if a piece on the
 * given square could be captured on the opponent's next turn.
 * 
 * Note: This generates all opponent moves, so it has some overhead.
 * For performance-critical code, consider caching results.
 * 
 * @param state   Current game state
 * @param square  Square index to check (0-63)
 * @return        1 if square is threatened, 0 otherwise
 */
int is_square_threatened(const GameState *state, int square);

#endif /* MOVEGEN_H */
