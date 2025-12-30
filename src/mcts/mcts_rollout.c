/**
 * mcts_rollout.c - MCTS Rollout/Simulation
 * 
 * Renamed from: simulation.c
 * Contains: simulate_rollout, pick_smart_move
 */

#include "mcts_types.h"
#include "mcts_tree.h"
#include "../core/movegen.h"
#include "../nn/cnn.h"
#include "../params.h"
#include <stdlib.h>
#include <math.h>

/**
 * Pick a move using heuristics (greedy or epsilon-greedy).
 */
static Move pick_smart_move(const MoveList *list, const GameState *state, int use_lookahead, MCTSConfig config) {
    if (list->count == 1) return list->moves[0];

    int best_score = -100000;
    int best_idx = 0;

    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        int score = 0;
        
        if (m.length > 0) score += 1000 * m.length;
        score += (int)evaluate_move_heuristic(state, &m, config);

        // Danger check (1-ply lookahead in endgame)
        if (use_lookahead && m.length == 0) {
            int total_pieces = __builtin_popcountll(get_pieces(state, WHITE) | get_pieces(state, BLACK));
            
            if (total_pieces < 12) {
                GameState future_state = *state;
                apply_move(&future_state, &m);
                
                MoveList enemy_moves;
                generate_moves(&future_state, &enemy_moves);
                
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

/**
 * Simulate random/heuristic rollout from node.
 * Returns value in [0, 1] for backprop.
 */
double simulate_rollout(Node *node, MCTSConfig config) {
    // CNN Value Evaluation (preferred)
    if (config.cnn_weights != NULL) {
        CNNOutput out;
        const GameState *hist1 = node->parent ? &node->parent->state : NULL;
        const GameState *hist2 = (node->parent && node->parent->parent) ? &node->parent->parent->state : NULL;
        cnn_forward_with_history((CNNWeights*)config.cnn_weights, &node->state, hist1, hist2, &out);
        return (out.value + 1.0f) / 2.0f;
    }

    GameState temp_state = node->state;
    int original_player = node->player_who_just_moved;
    
    if (node->is_terminal) {
        int winner = (temp_state.current_player == WHITE) ? BLACK : WHITE;
        return (winner == original_player) ? WIN_SCORE : LOSS_SCORE;
    }

    int depth = 0;
    MoveList temp_moves;
    
    // Use shorter depth for fast rollout
    int max_depth = config.use_fast_rollout ? 
                    (config.fast_rollout_depth > 0 ? config.fast_rollout_depth : 50) : 
                    MAX_ROLLOUT_DEPTH;
    
    while (depth < max_depth) {
        generate_moves(&temp_state, &temp_moves);

        if (temp_moves.count == 0) {
            int winner = (temp_state.current_player == WHITE) ? BLACK : WHITE;
            if (winner == original_player) {
                double score = WIN_SCORE;
                if (config.use_decaying_reward) score *= pow(config.decay_factor, depth);
                return score;
            }
            return LOSS_SCORE;
        }
        
        if (temp_state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            return DRAW_SCORE;
        }

        Move chosen_move;
        double r = (double)rand() / RAND_MAX;

        if (r < config.rollout_epsilon) {
            chosen_move = temp_moves.moves[rand() % temp_moves.count];
        } else {
            chosen_move = pick_smart_move(&temp_moves, &temp_state, config.use_lookahead, config);
        }

        apply_move(&temp_state, &chosen_move);
        depth++;
        
        // Fast rollout: early termination on material advantage
        if (config.use_fast_rollout && depth % 5 == 0) {  // Check every 5 moves
            int my_pieces = __builtin_popcountll(get_pieces(&temp_state, original_player));
            int opp_pieces = __builtin_popcountll(get_pieces(&temp_state, 1 - original_player));
            int diff = my_pieces - opp_pieces;
            
            if (diff >= 3) return 0.85;   // Strong material advantage
            if (diff <= -3) return 0.15;  // Strong material disadvantage
        }
    }
    
    // At max depth: return material-based evaluation instead of draw
    if (config.use_fast_rollout) {
        int my_pieces = __builtin_popcountll(get_pieces(&temp_state, original_player));
        int opp_pieces = __builtin_popcountll(get_pieces(&temp_state, 1 - original_player));
        double material_score = 0.5 + 0.05 * (my_pieces - opp_pieces);  // Range [0.1, 0.9]
        if (material_score < 0.1) material_score = 0.1;
        if (material_score > 0.9) material_score = 0.9;
        return material_score;
    }
    
    return DRAW_SCORE;
}
