/**
 * mcts_tree.h - MCTS Tree Operations
 * 
 * Consolidated from: selection.h, expansion.h, backprop.h
 * Contains: UCB selection, node creation/expansion, backpropagation
 */

#ifndef MCTS_TREE_H
#define MCTS_TREE_H

#include "mcts_types.h"

// =============================================================================
// SELECTION
// =============================================================================

/**
 * Calculate UCB1 score for a child node.
 */
double calculate_ucb1_score(Node *child, MCTSConfig config);

/**
 * Select the most promising node for expansion/simulation.
 * Traverses down the tree using UCB1/PUCT until finding an unexpanded node.
 */
Node* select_promising_node(Node *root, MCTSConfig config);

// =============================================================================
// EXPANSION
// =============================================================================

/**
 * Evaluate move heuristic for progressive bias.
 */
double evaluate_move_heuristic(const GameState *state, const Move *move, MCTSConfig config);

/**
 * Create a new MCTS node in the arena.
 */
Node* create_node(Node *parent, Move move, GameState state, Arena *arena, MCTSConfig config);

/**
 * Expand a node by adding one child for an untried move.
 */
Node* expand_node(Node *node, Arena *arena, TranspositionTable *tt, MCTSConfig config);

/**
 * Find a child node by move (for tree reuse).
 */
Node* find_child_by_move(Node *parent, const Move *move);

/**
 * Check if two moves are equal.
 */
int moves_equal(const Move *m1, const Move *m2);

// =============================================================================
// BACKPROPAGATION
// =============================================================================

/**
 * Backpropagate simulation result up the tree.
 * Flips perspective at each level and optionally updates solver status.
 */
void backpropagate(Node *node, double result, int use_solver);

/**
 * Update solver status for a node based on children.
 */
void update_solver_status(Node *node);

#endif // MCTS_TREE_H
