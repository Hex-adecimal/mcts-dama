/**
 * endgame.h - Endgame Position Generation for Training
 * 
 * Generates random but valid endgame positions to improve CNN coverage
 * on positions with queens (ladies).
 */

#ifndef ENDGAME_H
#define ENDGAME_H

#include "dama/engine/game.h"
#include "dama/common/rng.h"
#include "dama/engine/movegen.h"

// =============================================================================
// DARK SQUARES (Italian Checkers uses only dark squares)
// =============================================================================

// Squares where pieces can be placed: (row + col) % 2 == 1
extern const int DARK_SQUARES[];
#define NUM_DARK_SQUARES 32

// =============================================================================
// ENDGAME TYPES
// =============================================================================

typedef enum {
    EG_2D_VS_2D,       // 2 ladies each (complex)
    EG_2D_VS_1D1P,     // 2 ladies vs 1 lady + 1 pawn
    EG_1D2P_VS_1D2P,   // 1 lady + 2 pawns each (balanced)
    EG_2D_VS_1D,       // Material advantage
    EG_1D1P_VS_1D1P,   // Simple endgame
    EG_RANDOM_SPARSE,  // 2-4 pieces per side, random types
    NUM_ENDGAME_TYPES
} EndgameType;

// =============================================================================
// API
// =============================================================================

/**
 * @brief Setup a random endgame position.
 * 
 * @param state Output: the game state to initialize
 * @param rng Thread-local RNG for reproducibility
 * @return 1 if valid position generated, 0 if failed (caller should retry)
 */
int setup_random_endgame(GameState *state, RNG *rng);

/**
 * @brief Check if a position has at least one legal move.
 * @param state Pointer to the GameState to check.
 * @return 1 if moves exist, 0 otherwise.
 */
static inline int position_has_moves(const GameState *state) {
    MoveList ml;
    movegen_generate(state, &ml);
    return ml.count > 0;
}

#endif // ENDGAME_H
