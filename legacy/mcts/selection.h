#ifndef SELECTION_H
#define SELECTION_H

#include "mcts_types.h"

// =============================================================================
// SELECTION API
// =============================================================================

/**
 * Calculate UCB1 score for a node.
 */
double calculate_ucb1(Node *child, MCTSConfig config);

/**
 * Calculate UCB1-Tuned score for a node.
 */
double calculate_ucb1_tuned(Node *child, MCTSConfig config);

/**
 * Calculate PUCT score for a node with neural network prior.
 */
double calculate_puct(Node *child, MCTSConfig config, float prior);

/**
 * Calculate the combined UCB1 score with optional progressive bias.
 */
double calculate_ucb1_score(Node *child, MCTSConfig config);

/**
 * Select the most promising node to explore.
 * Descends the tree until a node with untried moves or a terminal node is reached.
 */
Node* select_promising_node(Node *root, MCTSConfig config);

#endif // SELECTION_H
