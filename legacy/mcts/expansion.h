#ifndef EXPANSION_H
#define EXPANSION_H

#include "mcts_types.h"
#include "mcts_internal.h"

// =============================================================================
// EXPANSION API
// =============================================================================

/**
 * Evaluate a move based on heuristics (Promotion, Safety, Center Control).
 * Higher score = better move.
 */
double evaluate_move_heuristic(const GameState *state, const Move *move, MCTSConfig config);

/**
 * Create a new MCTS node.
 */
Node* create_node(Node *parent, Move move, GameState state, Arena *arena, MCTSConfig config);

/**
 * Expand a leaf node by adding one child for an untried move.
 */
Node* expand_node(Node *node, Arena *arena, TranspositionTable *tt, MCTSConfig config);

/**
 * Compare two moves for equality.
 */
int moves_equal(const Move *m1, const Move *m2);

/**
 * Find child node matching the given move.
 */
Node* find_child_by_move(Node *parent, const Move *move);

#endif // EXPANSION_H
