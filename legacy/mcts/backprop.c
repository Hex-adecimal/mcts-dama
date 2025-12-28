/**
 * backprop.c - MCTS Backpropagation
 * 
 * Contains: update_solver_status, backpropagate
 */

#include "backprop.h"

void update_solver_status(Node *node) {
    if (node->status != SOLVED_NONE) return; // Already solved
    
    // Condition 1: Can we win immediately?
    // If ANY child is SOLVED_LOSS (for the opponent), implies SOLVED_WIN for us.
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i]->status == SOLVED_LOSS) {
            node->status = SOLVED_WIN;
            return; 
        }
    }

    // Condition 2: Are we forced to lose?
    // Must be fully expanded (no untried moves) AND all children are SOLVED_WIN (for the opponent).
    if (node->untried_moves.count == 0) {
        int all_win = 1;
        for (int i = 0; i < node->num_children; i++) {
            if (node->children[i]->status != SOLVED_WIN) {
                all_win = 0;
                break;
            }
        }
        
        if (all_win && node->num_children > 0) {
            node->status = SOLVED_LOSS;
        }
    }
}

void backpropagate(Node *node, double result, int use_solver) {
    Node *child = NULL; // Keep track of the node we came from
    
    while (node != NULL) {
        // 1. Remove Virtual Loss (atomic)
        atomic_fetch_sub(&node->virtual_loss, 1);
        
        // 2. Increment Visits (atomic)
        atomic_fetch_add(&node->visits, 1);
        
        // 3. Update Score (Lock Proteced)
        pthread_mutex_lock(&node->lock);
        
        // Add Result (from this node's player perspective)
        node->score += result;
        node->sum_sq_score += (result * result);

        // SOLVER UPDATE
        if (use_solver) {
            if (child == NULL || child->status != SOLVED_NONE) {
                update_solver_status(node);
            }
        }
        
        pthread_mutex_unlock(&node->lock);

        // CRITICAL: Flip perspective for the parent node!
        result = 1.0 - result;

        child = node;
        node = node->parent;
    }
}
