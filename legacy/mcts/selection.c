/**
 * selection.c - MCTS Node Selection
 * 
 * Contains: UCB1, UCB1-Tuned, PUCT, select_promising_node
 */

#include "selection.h"
#include "../nn/cnn.h"
#include "../nn/cnn.h"
#include "../params.h"
#include <math.h>

double calculate_ucb1(Node *child, MCTSConfig config) {
    if (child->visits == 0) {
        return config.use_fpu ? config.fpu_value : 1e9;
    }

    double win_rate = child->score / (double)child->visits;
    double exploration = UCB1_C * sqrt(log((double)child->parent->visits) / (double)child->visits);

    return win_rate + exploration;
}

double calculate_ucb1_tuned(Node *child, MCTSConfig config) {
    if (child->visits == 0) {
        return config.use_fpu ? config.fpu_value : 1e9;
    }

    double N = (double)child->parent->visits;
    double n = (double)child->visits;
    
    double mean = child->score / n;
    
    // Calculate variance of rewards
    // Var(X) = E[X^2] - (E[X])^2
    double avg_sq_score = child->sum_sq_score / n;
    double variance = avg_sq_score - (mean * mean);
    
    // Variance Upper Bound
    double v_upper = variance + sqrt(2.0 * log(N) / n);

    // Tuned exploration term
    double min_v = (v_upper < 0.25) ? v_upper : 0.25;
    double exploration = sqrt((log(N) / n) * min_v);

    return mean + exploration;
}

double calculate_puct(Node *child, MCTSConfig config, float prior) {
    // Atomic loads for thread safety
    int v_loss = atomic_load_explicit(&child->virtual_loss, memory_order_relaxed);
    int visits = atomic_load_explicit(&child->visits, memory_order_relaxed);
    
    // Virtual Loss Logic: Penalize score by 1.0 for each in-flight traversal
    // Q = (W - VL) / (N + VL)
    double effective_score = child->score - (double)v_loss;
    double effective_visits = (double)(visits + v_loss);
    
    if (effective_visits < 1.0) effective_visits = 1e-9; // Avoid div by zero
    
    double q_value = effective_score / effective_visits;
    
    // Parent visits (approximate is fine for exploration term)
    double parent_visits = (double)atomic_load_explicit(&child->parent->visits, memory_order_relaxed);
    
    double u_value = config.puct_c * prior * 
                     sqrt(parent_visits) / 
                     (1.0 + effective_visits);
                     
    return q_value + u_value;
}

double calculate_ucb1_score(Node *child, MCTSConfig config) {
    double base_score;
    
    // PUCT selection (using cached prior from expansion/creation)
    if (config.use_puct) {
        base_score = calculate_puct(child, config, child->prior);
    } else if (config.use_ucb1_tuned) {
        base_score = calculate_ucb1_tuned(child, config);
    } else {
        base_score = calculate_ucb1(child, config);
    }
    
    if (config.use_progressive_bias) {
        // Atomic visits for bias stability
        int visits = atomic_load_explicit(&child->visits, memory_order_relaxed);
        double bias = config.bias_constant * (child->heuristic_score / (double)(visits + 1));
        return base_score + bias;
    }
    
    return base_score;
}

Node* select_promising_node(Node *root, MCTSConfig config) {
    Node *current = root;
    while (!current->is_terminal && current->untried_moves.count == 0) {
        if (current->num_children == 0) break;

        // SOLVER LOGIC: Taking the win immediately
        if (config.use_solver && current->status == SOLVED_WIN) {
            for (int i=0; i < current->num_children; i++) {
                if (current->children[i]->status == SOLVED_LOSS) {
                    current = current->children[i];
                    // Virtual Loss for Solver path too
                    atomic_fetch_add(&current->virtual_loss, 1);
                    goto next_node;
                }
            }
        }
        
        double best_score = -1e9;
        Node *best_node = NULL;
        
        for (int i = 0; i < current->num_children; i++) {
            Node *child = current->children[i];
            if (!child) continue; // Race condition safety
            double ucb_value;
            
            // SOLVER PRUNING:
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
            
            if (ucb_value > best_score) {
                best_score = ucb_value;
                best_node = child;
            }
        }
        
        if (best_node) {
            current = best_node;
            // Apply Virtual Loss to steer other threads away
            atomic_fetch_add(&current->virtual_loss, 1);
        } else {
            break; // Should not happen
        }
        
        next_node:;
    }
    return current;
}
