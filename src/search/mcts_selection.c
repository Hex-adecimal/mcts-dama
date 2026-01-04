/**
 * mcts_selection.c - MCTS Node Selection Algorithms
 * 
 * Extracted from mcts_tree.c for better modularity.
 * Contains: UCB1, UCB1-Tuned, PUCT, select_promising_node
 */

#include "dama/search/mcts_tree.h"
#include "dama/search/mcts_types.h"
#include "dama/common/params.h"
#include <math.h>
#include <stdio.h>

// =============================================================================
// UCB1 SELECTION
// =============================================================================

static double calculate_ucb1(Node *child, MCTSConfig config) {
    if (child->visits == 0) {
        return config.use_fpu ? config.fpu_value : 1e9;
    }
    double win_rate = child->score / (double)child->visits;
    double exploration = UCB1_C * sqrt(log((double)child->parent->visits) / (double)child->visits);
    return win_rate + exploration;
}

// =============================================================================
// UCB1-TUNED SELECTION
// =============================================================================

static double calculate_ucb1_tuned(Node *child, MCTSConfig config) {
    if (child->visits == 0) {
        return config.use_fpu ? config.fpu_value : 1e9;
    }
    double N = (double)child->parent->visits;
    double n = (double)child->visits;
    double mean = child->score / n;
    double avg_sq_score = child->sum_sq_score / n;
    double variance = avg_sq_score - (mean * mean);
    double v_upper = variance + sqrt(2.0 * log(N) / n);
    double min_v = (v_upper < 0.25) ? v_upper : 0.25;
    double exploration = sqrt((log(N) / n) * min_v);
    return mean + exploration;
}

// =============================================================================
// PUCT SELECTION (AlphaZero-style)
// =============================================================================

/**
 * Calculate PUCT score (AlphaZero-style exploration).
 * 
 * PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
 * Uses virtual loss for thread-safe parallel MCTS.
 */
static double calculate_puct(Node *child, MCTSConfig config, float prior) {
    int v_loss = atomic_load_explicit(&child->virtual_loss, memory_order_relaxed);
    int visits = atomic_load_explicit(&child->visits, memory_order_relaxed);
    
    double effective_score = child->score - (double)v_loss;
    double effective_visits = (double)(visits + v_loss);
    
    double q_value;
    if (effective_visits < 1.0) {
        effective_visits = 1e-9;
        // Avoid division by epsilon if score is negative or zero,
        // just use the score itself as Q-value for unvisited nodes.
        // This prevents -1.0 / 1e-9 = -1e9 explosions.
        q_value = effective_score; 
    } else {
        q_value = effective_score / effective_visits;
    }

    double parent_visits = (double)atomic_load_explicit(&child->parent->visits, memory_order_relaxed);
    double u_value = config.puct_c * prior * sqrt(parent_visits) / (1.0 + effective_visits);
    
    if (config.verbose) {
        printf("PUCT: Q=%.3f U=%.3f EffScore=%.3f Score=%.3f EffVis=%.9f VLoss=%d\n", 
               q_value, u_value, effective_score, child->score, effective_visits, v_loss);
    }
    
    return q_value + u_value;
}

// =============================================================================
// UNIFIED UCB SCORE CALCULATION
// =============================================================================

double calculate_ucb1_score(Node *child, MCTSConfig config) {
    double base_score;
    
    if (config.use_puct) {
        base_score = calculate_puct(child, config, child->prior);
    } else if (config.use_ucb1_tuned) {
        base_score = calculate_ucb1_tuned(child, config);
    } else {
        base_score = calculate_ucb1(child, config);
    }
    
    if (config.use_progressive_bias) {
        int visits = atomic_load_explicit(&child->visits, memory_order_relaxed);
        double bias = config.bias_constant * (child->heuristic_score / (double)(visits + 1));
        return base_score + bias;
    }
    
    return base_score;
}

// =============================================================================
// NODE SELECTION (Tree Policy)
// =============================================================================

/**
 * Select the most promising leaf node for expansion.
 * 
 * Traverses from root to leaf using UCB/PUCT scores.
 * Applies virtual loss for thread-safe parallel MCTS.
 * Handles solved nodes (proven wins/losses) when solver is enabled.
 */
Node* select_promising_node(Node *root, MCTSConfig config) {
    Node *current = root;
    while (!current->is_terminal && current->untried_moves.count == 0) {
        if (current->num_children == 0) break;

        // Solver: take winning move immediately
        int found_winning_child = 0;
        if (config.use_solver && current->status == SOLVED_WIN) {
            for (int i = 0; i < current->num_children; i++) {
                if (current->children[i]->status == SOLVED_LOSS) {
                    current = current->children[i];
                    atomic_fetch_add(&current->virtual_loss, 1);
                    found_winning_child = 1;
                    break;
                }
            }
        }
        if (found_winning_child) continue;
        
        double best_score = -1e9;
        Node *best_node = NULL;
        
        for (int i = 0; i < current->num_children; i++) {
            Node *child = current->children[i];
            if (!child) continue;
            
            double ucb_value;
            if (config.use_solver) {
                if (child->status == SOLVED_WIN) {
                    ucb_value = -100000.0;
                } else if (child->status == SOLVED_LOSS) {
                    ucb_value = 100000.0 + child->score;
                } else {
                    ucb_value = calculate_ucb1_score(child, config);
                }
            } else {
                ucb_value = calculate_ucb1_score(child, config);
            }
            if (config.verbose) {
                printf("Sel Child %d: Visits=%d Prior=%.3f Heur=%.3f BiasConst=%.3f UCB=%.3f\n", 
                       i, atomic_load(&child->visits), child->prior, child->heuristic_score, config.bias_constant, ucb_value);
            }
            
            if (ucb_value > best_score) {
                best_score = ucb_value;
                best_node = child;
            }
        }
        
        if (best_node) {
            current = best_node;
            atomic_fetch_add(&current->virtual_loss, 1);
        } else {
            break;
        }
    }
    return current;
}
