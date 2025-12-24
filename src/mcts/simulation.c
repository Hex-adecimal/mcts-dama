/**
 * simulation.c - MCTS Rollout Simulation
 * 
 * Contains: pick_smart_move, simulate_rollout
 */

#include "simulation.h"
#include "expansion.h"
#include "../core/movegen.h"
#include "../params.h"
#include <stdlib.h>
#include <math.h>

Move pick_smart_move(const MoveList *list, const GameState *state, int use_lookahead, MCTSConfig config) {
    if (list->count == 1) return list->moves[0];

    int best_score = -100000;
    int best_idx = 0; 

    // Evaluate moves
    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        int score = 0;
        
        // Prefer captures (already handled by rule, but good for robust fallback)
        if (m.length > 0) score += 1000 * m.length;
        
        // Use shared heuristic function
        score += (int)evaluate_move_heuristic(state, &m, config);

        // Danger Check (1-ply lookahead)
        // Only in endgame positions (< 12 total pieces) where tactics matter most
        if (use_lookahead && m.length == 0) {
            int total_pieces = __builtin_popcountll(state->white_pieces | state->black_pieces | 
                                                     state->white_ladies | state->black_ladies);
            
            if (total_pieces < 12) {
                GameState future_state = *state;
                apply_move(&future_state, &m);
                
                MoveList enemy_moves;
                generate_moves(&future_state, &enemy_moves);
                
                // If opponent can capture after this move, it's dangerous
                if (enemy_moves.count > 0 && enemy_moves.moves[0].length > 0) {
                    score -= WEIGHT_DANGER;
                }
            }
        }

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return list->moves[best_idx];
}

#include "../nn/cnn.h"

double simulate_rollout(Node *node, MCTSConfig config) {
    // CNN Value Evaluation (preferred if available)
    if (config.cnn_weights != NULL) {
        CNNOutput out;
        
        // Recover history from node hierarchy
        const GameState *hist1 = (node->parent) ? &node->parent->state : NULL;
        const GameState *hist2 = (node->parent && node->parent->parent) ? &node->parent->parent->state : NULL;
        
        cnn_forward_with_history((CNNWeights*)config.cnn_weights, &node->state, hist1, hist2, &out);
        
        // Output value is [-1, 1], map to [0, 1] for MCTS
        return (out.value + 1.0f) / 2.0f;
    }

    GameState temp_state = node->state;
    
    // Check if the node itself is terminal
    if (node->is_terminal) {
        int winner = (temp_state.current_player == WHITE) ? BLACK : WHITE;
        if (winner == node->player_who_just_moved) return WIN_SCORE;
        else return LOSS_SCORE;
    }

    int depth = 0;
    MoveList temp_moves;
    
    // Limit depth to avoid infinite loops
    while (depth < MAX_ROLLOUT_DEPTH) {
        generate_moves(&temp_state, &temp_moves);

        if (temp_moves.count == 0) {
            int winner = (temp_state.current_player == WHITE) ? BLACK : WHITE;
            if (winner == node->player_who_just_moved) {
                double score = WIN_SCORE;
                if (config.use_decaying_reward) {
                    score *= pow(config.decay_factor, depth);
                }
                return score;
            } else {
                return LOSS_SCORE;
            }
        }
        
        if (temp_state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            return DRAW_SCORE; // Draw
        }

        Move chosen_move;

        // Epsilon-Greedy Strategy
        double r = (double)rand() / RAND_MAX;

        if (r < config.rollout_epsilon) {
            int random_idx = rand() % temp_moves.count;
            chosen_move = temp_moves.moves[random_idx];
        } else {
            chosen_move = pick_smart_move(&temp_moves, &temp_state, config.use_lookahead, config);
        }

        apply_move(&temp_state, &chosen_move);
        depth++;
    }
    return DRAW_SCORE; // Draw if depth limit reached
}
