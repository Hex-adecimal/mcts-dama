/**
 * mcts_tree.c - MCTS Tree Operations
 * 
 * Consolidated from: selection.c, expansion.c, backprop.c
 * Contains: UCB selection, node creation/expansion, backpropagation
 */

#include "mcts_tree.h"
#include "mcts_types.h"
#include "../core/movegen.h"
#include "../nn/cnn.h"
#include "../params.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// =============================================================================
// SELECTION
// =============================================================================

static double calculate_ucb1(Node *child, MCTSConfig config) {
    if (child->visits == 0) {
        return config.use_fpu ? config.fpu_value : 1e9;
    }
    double win_rate = child->score / (double)child->visits;
    double exploration = UCB1_C * sqrt(log((double)child->parent->visits) / (double)child->visits);
    return win_rate + exploration;
}

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
    
    if (config.verbose) { // Only print if verbose
        printf("PUCT: Q=%.3f U=%.3f EffScore=%.3f Score=%.3f EffVis=%.9f VLoss=%d\n", 
               q_value, u_value, effective_score, child->score, effective_visits, v_loss);
    }
    
    return q_value + u_value;
}

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

Node* select_promising_node(Node *root, MCTSConfig config) {
    Node *current = root;
    while (!current->is_terminal && current->untried_moves.count == 0) {
        if (current->num_children == 0) break;

        // Solver: take winning move immediately
        if (config.use_solver && current->status == SOLVED_WIN) {
            for (int i = 0; i < current->num_children; i++) {
                if (current->children[i]->status == SOLVED_LOSS) {
                    current = current->children[i];
                    atomic_fetch_add(&current->virtual_loss, 1);
                    goto next_node;
                }
            }
        }
        
        double best_score = -1e9;
        Node *best_node = NULL;
        
        
        for (int i = 0; i < current->num_children; i++) {
             Node *child = current->children[i];
             // We don't know if it used PUCT or UCB inside the function call logic, 
             // but we can just print what we have.
             // Actually, let's just print loop index.
             // printf("Sel Check Child %d/%d (%p)\n", i, current->num_children, child);
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
        
        next_node:;
    }
    return current;
}

// =============================================================================
// EXPANSION
// =============================================================================

double evaluate_move_heuristic(const GameState *state, const Move *move, MCTSConfig config) {
    double score = 0.0;
    int us = state->current_player;
    
    int target_idx = (move->length == 0) ? 1 : move->length;
    int from = move->path[0];
    int to = move->path[target_idx];
    int row = to / 8;
    int col = to % 8;
    int from_row = from / 8;

    // Capture bonus
    if (move->length > 0) score += config.weights.w_capture * move->length;

    // Promotion bonus
    if (!move->is_lady_move) {
        if (row == 0 || row == 7) score += config.weights.w_promotion;
        int dist = (us == WHITE) ? (7 - row) : row;
        score += (7 - dist) * config.weights.w_advance;
    }

    // Edge safety (pieces only)
    if (!move->is_lady_move && (col == 0 || col == 7)) score += config.weights.w_edge;

    // Center control
    if ((row == 3 || row == 4) && (col >= 2 && col <= 5)) score += config.weights.w_center;

    // Base protection penalty
    if (!move->is_lady_move) {
        if ((us == WHITE && from_row == 0) || (us == BLACK && from_row == 7)) {
            score -= config.weights.w_base;
        }
    }
    
    // Queen activity
    if (move->is_lady_move) score += config.weights.w_lady_activity;
    
    if (score < -100.0) {
        printf("Heur Debug: Score=%.2f MoveLen=%d Us=%d Rank=%d\n", score, move->length, us, row);
    }
    
    return score;
}

Node* create_node(Node *parent, Move move, GameState state, Arena *arena, MCTSConfig config) {
    Node *node = (Node*)arena_alloc(arena, sizeof(Node));
    if (!node) return NULL;
    
    memset(node, 0, sizeof(Node));
    node->state = state;
    node->move_from_parent = move;
    node->parent = parent;
    node->player_who_just_moved = (state.current_player == WHITE) ? BLACK : WHITE;

    node->children = (Node**)arena_alloc(arena, MAX_MOVES * sizeof(Node*));
    node->num_children = 0;
    atomic_init(&node->visits, 0);
    atomic_init(&node->virtual_loss, 0);
    
    if (pthread_mutex_init(&node->lock, NULL) != 0) {
        printf("ERROR: Mutex init failed\n");
    }

    node->score = 0.0;
    node->sum_sq_score = 0.0;
    node->cached_policy = NULL;
    
    node->heuristic_score = evaluate_move_heuristic(&node->state, &node->move_from_parent, config);
    
    generate_moves(&node->state, &node->untried_moves);
    node->is_terminal = (node->untried_moves.count == 0 && node->num_children == 0) ? 1 : 0;
    
    // Solver init
    node->status = SOLVED_NONE;
    
    if (config.verbose) printf("Creation End %p Heur=%.2f\n", node, node->heuristic_score);
    if (node->is_terminal) {
        node->status = SOLVED_LOSS;
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            node->status = SOLVED_DRAW;
        }
    }
    
    // Heuristic & PUCT init
    if (parent) {
        node->heuristic_score = evaluate_move_heuristic(&parent->state, &move, config);
        
        if (config.weights.w_threat > 0.0) {
            int dest = (move.length == 0) ? move.path[1] : move.path[move.length];
            if (is_square_threatened(&node->state, dest)) {
                node->heuristic_score -= config.weights.w_threat;
            }
        }
        
        // PUCT prior from cached policy
        if (config.use_puct) {
            if (!parent->cached_policy && config.cnn_weights) {
                parent->cached_policy = (float*)arena_alloc(arena, CNN_POLICY_SIZE * sizeof(float));
                
                CNNOutput out;
                const GameState *hist1 = parent->parent ? &parent->parent->state : NULL;
                const GameState *hist2 = (parent->parent && parent->parent->parent) ? &parent->parent->parent->state : NULL;
                cnn_forward_with_history((CNNWeights*)config.cnn_weights, &parent->state, hist1, hist2, &out);
                
                MoveList legal_moves;
                generate_moves(&parent->state, &legal_moves);
                
                double sum = 0.0;
                for (int i = 0; i < legal_moves.count; i++) {
                    int idx = cnn_move_to_index(&legal_moves.moves[i], parent->state.current_player);
                    if (idx >= 0 && idx < CNN_POLICY_SIZE) sum += out.policy[idx];
                }
                
                for (int i = 0; i < CNN_POLICY_SIZE; i++) parent->cached_policy[i] = 0.0f;
                if (sum < 1e-9) sum = 1.0;
                double scale = 1.0 / sum;
                
                for (int i = 0; i < legal_moves.count; i++) {
                    int idx = cnn_move_to_index(&legal_moves.moves[i], parent->state.current_player);
                    if (idx >= 0 && idx < CNN_POLICY_SIZE) parent->cached_policy[idx] = out.policy[idx] * scale;
                }
            }
            
            if (parent->cached_policy) {
                int idx = cnn_move_to_index(&move, parent->state.current_player);
                node->prior = (idx >= 0 && idx < CNN_POLICY_SIZE) ? parent->cached_policy[idx] : 0.0f;
            } else {
                node->prior = 1.0f;
            }
        } else {
            node->prior = 0.0f;
        }

        // Loop detection
        Node *ancestor = parent;
        while (ancestor) {
            if (ancestor->state.hash == node->state.hash) {
                node->is_terminal = 1;
                node->status = SOLVED_DRAW;
                node->heuristic_score -= 50000.0;
                node->score = -1.0;
                break;
            }
            ancestor = ancestor->parent;
        }
    } else {
        node->heuristic_score = 0.0;
        node->prior = 0.0f;
    }

    return node;
}

int moves_equal(const Move *m1, const Move *m2) {
    if (m1->length != m2->length) return 0;
    int limit = (m1->length == 0) ? 1 : m1->length;
    for (int i = 0; i <= limit; i++) {
        if (m1->path[i] != m2->path[i]) return 0;
    }
    return 1;
}

Node* find_child_by_move(Node *parent, const Move *move) {
    if (!parent || !move) return NULL;
    for (int i = 0; i < parent->num_children; i++) {
        if (moves_equal(&parent->children[i]->move_from_parent, move)) {
            return parent->children[i];
        }
    }
    return NULL;
}

Node* expand_node(Node *node, Arena *arena, TranspositionTable *tt, MCTSConfig config) {
    if (node->untried_moves.count == 0) return node;

    int idx = node->untried_moves.count - 1;
    Move move_to_try = node->untried_moves.moves[idx];
    node->untried_moves.count--;

    GameState next_state = node->state;
    apply_move(&next_state, &move_to_try);

    Node *child = create_node(node, move_to_try, next_state, arena, config);

    if (tt) {
        Node *match = tt_lookup(tt, &next_state);
        if (match) {
            child->visits = match->visits;
            child->score = match->score;
            child->sum_sq_score = match->sum_sq_score;
            child->status = match->status;
        }
        tt_insert(tt, child);
    }

    if (node->num_children >= MAX_MOVES) {
        printf("ERROR: Exceeded MAX_MOVES in expand_node\n");
        return node;
    }

    node->children[node->num_children] = child;
    atomic_thread_fence(memory_order_release);
    node->num_children++;

    return child;
}

// =============================================================================
// BACKPROPAGATION
// =============================================================================

void update_solver_status(Node *node) {
    if (node->status != SOLVED_NONE) return;
    
    // Can we win? (any child is loss for opponent)
    for (int i = 0; i < node->num_children; i++) {
        if (node->children[i]->status == SOLVED_LOSS) {
            node->status = SOLVED_WIN;
            return;
        }
    }

    // Are we forced to lose? (fully expanded, all children are wins for opponent)
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
    Node *child = NULL;
    
    while (node != NULL) {
        atomic_fetch_sub(&node->virtual_loss, 1);
        atomic_fetch_add(&node->visits, 1);
        
        pthread_mutex_lock(&node->lock);
        node->score += result;
        node->sum_sq_score += (result * result);
        
        if (use_solver && (child == NULL || child->status != SOLVED_NONE)) {
            update_solver_status(node);
        }
        pthread_mutex_unlock(&node->lock);

        result = 1.0 - result;
        child = node;
        node = node->parent;
    }
}
