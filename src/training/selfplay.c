/**
 * selfplay.c - Self-play data generation logic
 */

#include "dama/training/selfplay.h"
#include "dama/common/rng.h"
#include "dama/common/logging.h"
#include "dama/engine/movegen.h"
#include "dama/neural/cnn.h"
#include "dama/training/dataset.h"
#include "dama/engine/endgame.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// HELPER TYPES & CONSTANTS
// =============================================================================

typedef struct {
    float policy[CNN_POLICY_SIZE];
    GameState state;
    GameState history[2];
    int player;
} GameStep;

typedef enum {
    END_CHECKMATE,
    END_RESIGNATION,
    END_MERCY,
    END_40_MOVE,
    END_MAX_MOVES
} EndReason;

#define DEFAULT_DIRICHLET_ALPHA 0.3f
#define DEFAULT_DIRICHLET_EPSILON 0.25f
#define DEFAULT_TEMP_THRESHOLD 30

// =============================================================================
// HELPERS
// =============================================================================

static void add_dirichlet_noise(Node *root, RNG *rng, float eps, float alpha) {
    if (!root || root->num_children == 0) return;
    int n = root->num_children;
    
    // Check after malloc to prevent crash on allocation failure
    float *noise = malloc(n * sizeof(float));
    if (!noise) {
        log_warn("[selfplay] Failed to allocate Dirichlet noise array");
        return;
    }
    
    float sum = 0;
    for (int i = 0; i < n; i++) {
        noise[i] = rng_gamma(rng, alpha);
        sum += noise[i];
    }
    
    // Division-by-zero protection when all noise values are zero
    if (sum > 1e-9f) {
        for (int i = 0; i < n; i++) {
            root->children[i]->prior = (1.0f - eps) * root->children[i]->prior + eps * noise[i] / sum;
        }
    }
    free(noise);
}

static int game_over(const GameState *s) {
    MoveList m;
    movegen_generate(s, &m);
    return m.count == 0;
}



// =============================================================================
// GAME PLAY LOOP
// =============================================================================

/**
 * Play a single self-play game.
 * 
 * Generates training data by running MCTS simulations against itself.
 * Features:
 * - Temperature annealing (exploration -> exploitation)
 * - Dirichlet noise for root exploration
 * - Resignation check using NN value
 * - Tree reuse for efficiency
 */
static int play_game(
    const CNNWeights *weights, 
    MCTSConfig cfg_white, 
    MCTSConfig cfg_black,
    GameStep *history_buffer, 
    int *out_steps, 
    EndReason *out_reason, 
    float initial_temp, 
    RNG *rng,
    float endgame_prob
) {
    GameState state;
    
    // Endgame Initialization (Configurable Probability)
    if (rng_f32(rng) < endgame_prob) {
        setup_random_endgame(&state, rng);
    } else {
        init_game(&state);
        
        // 2 random opening moves (only for normal games)
        for (int i = 0; i < 2; i++) {
            MoveList list;
            movegen_generate(&state, &list);
            if (list.count > 0) {
                int idx = rng_u32(rng) % list.count;
                apply_move(&state, &list.moves[idx]);
            }
        }
    }
    
    Arena arena;
    arena_init(&arena, ARENA_SIZE_SELFPLAY); // Use central constant
    Node *root = NULL; // Persistent root for tree reuse

    int moves = 0;
    int max_moves = 200;
    float policy[CNN_POLICY_SIZE];
    
    while (!game_over(&state) && moves < max_moves) {
        float temp = (moves < DEFAULT_TEMP_THRESHOLD) ? initial_temp : 0.1f;
        
        MCTSConfig cfg = (state.current_player == WHITE) ? cfg_white : cfg_black;
        
        // Tree Reuse: Only create root if we don't have one (start of game or lost track)
        if (!DEFAULT_TREE_REUSE || !root) {
             if (!DEFAULT_TREE_REUSE) arena_reset(&arena);
             root = mcts_create_root(state, &arena, cfg);
        } else {
             // Verify root state matches current state
             if (root->state.hash != state.hash) {
                 root = mcts_create_root(state, &arena, cfg);
             }
        }
        
        // Add Dirichlet noise for exploration (first 30 moves)
        if (moves < 30) {
            add_dirichlet_noise(root, rng, DEFAULT_DIRICHLET_EPSILON, DEFAULT_DIRICHLET_ALPHA);
        }
        
        mcts_search(root, &arena, 0.0, cfg, NULL, NULL); // Time 0.0 -> use max_nodes or implicit
        mcts_get_policy(root, policy, temp, &state);
        
        // Select move based on temperature
        Move chosen = {0};
        double r_val = rng_f32(rng);
        int best_child = 0;
        double max_visit = -1;
        
        if (temp < 0.1f) {
            // Low temp: select most visited child (greedy)
            for (int i=0; i<root->num_children; i++) {
                if (root->children[i]->visits > max_visit) {
                    max_visit = root->children[i]->visits;
                    best_child = i;
                }
            }
            chosen = root->children[best_child]->move_from_parent;
        } else {
            // High temp: sample proportionally to visit counts
             double sum = 0;
             double exp_t = 1.0/temp;
             for(int i=0; i<root->num_children; i++) sum += pow(root->children[i]->visits, exp_t);
             
             double threshold = r_val * sum;
             double current = 0;
             for(int i=0; i<root->num_children; i++) {
                 current += pow(root->children[i]->visits, exp_t);
                 if (current >= threshold) {
                     chosen = root->children[i]->move_from_parent;
                     break;
                 }
             }
             if (chosen.path[0] == 0) chosen = root->children[0]->move_from_parent;
        }
        
        // Record
        history_buffer[moves].state = state;
        history_buffer[moves].player = state.current_player;
        memcpy(history_buffer[moves].policy, policy, sizeof(policy));
        if (moves >= 1) history_buffer[moves].history[0] = history_buffer[moves-1].state;
        if (moves >= 2) history_buffer[moves].history[1] = history_buffer[moves-2].state;
        
        apply_move(&state, &chosen);
        
        // Tree Reuse: Advance root to the chosen child
        if (DEFAULT_TREE_REUSE && root) {
            Node *next_root = find_child_by_move(root, &chosen);
            if (next_root) {
                // Found! Reuse this subtree
                root = next_root;
                root->parent = NULL; // Detach from parent to be safe (though logically fine)
            } else {
                root = NULL; // Will recreate next iter
            }
        } else {
             root = NULL;
             // Reset handled at start of next loop via (!DEFAULT_TREE_REUSE) check
        }
        
        // Arena accumulates for whole game (~40k nodes).
        moves++;
        
        // Checks
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            arena_free(&arena);
            *out_steps = moves;
            *out_reason = END_40_MOVE;
            return 0; // Draw
        }
        
        // Resignation check (using NN directly after RESIGN_CHECK_THRESHOLD moves)
        if (moves > RESIGN_CHECK_THRESHOLD && weights) {
            CNNOutput out;
            GameState *h1 = (moves>=1) ? &history_buffer[moves-1].state : NULL;
            GameState *h2 = (moves>=2) ? &history_buffer[moves-2].state : NULL;
            cnn_forward_with_history(weights, &state, h1, h2, &out);
             
            if (out.value < RESIGN_THRESHOLD) {
                 arena_free(&arena);
                 *out_steps = moves;
                 *out_reason = END_RESIGNATION;
                 return (state.current_player == WHITE) ? -1 : 1;
             }
        }
    }
    
    arena_free(&arena);
    *out_steps = moves;
    *out_reason = (moves >= max_moves) ? END_MAX_MOVES : END_CHECKMATE;
    
    if (*out_reason == END_MAX_MOVES) return 0;
    return (state.current_player == WHITE) ? -1 : 1;
}

// =============================================================================
// RUN
// =============================================================================

void selfplay_run(const SelfplayConfig *sp_cfg, const MCTSConfig *mcts_cfg) {
    if (sp_cfg->on_start) sp_cfg->on_start(sp_cfg->games);
    
    // File setup
    if (sp_cfg->overwrite_data) {
        FILE *f = fopen(sp_cfg->output_file, "wb");
        if (f) fclose(f); // Create/Truncate
    }
    
    int completed_games = 0;
    int total_wins = 0, total_losses = 0, total_draws = 0;
    
    int num_threads = (sp_cfg->parallel_threads > 0) ? sp_cfg->parallel_threads : 1;
    
    // CNN weights from mcts_cfg
    const CNNWeights *weights = (const CNNWeights*)mcts_cfg->cnn_weights;
    
    #pragma omp parallel num_threads(num_threads)
    {
        // Thread-local RNG
        RNG rng;
        unsigned int seed = time(NULL) ^ (omp_get_thread_num() * 12345);
        rng_seed(&rng, seed);
        
        // Check for memory allocation failure
        GameStep *history = malloc(sp_cfg->max_moves * sizeof(GameStep));
        TrainingSample *batch = malloc(sp_cfg->max_moves * sizeof(TrainingSample));
        int alloc_failed = (!history || !batch);
        
        if (alloc_failed) {
            log_error("[selfplay] Failed to allocate game buffers (thread %d)", omp_get_thread_num());
            free(history);
            free(batch);
            history = NULL;
            batch = NULL;
        }
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < sp_cfg->games; i++) {
            // Skip if allocation failed for this thread
            if (alloc_failed) continue;
            int steps = 0;
            EndReason reason;
            
            // Mixed Opponent: Probabilistically replace one side with Grandmaster
            MCTSConfig w_cfg = *mcts_cfg;
            MCTSConfig b_cfg = *mcts_cfg;
            
            if (rng_f32(&rng) < MIX_OPPONENT_PROB) {
                if (rng_u32(&rng) % 2 == 0) {
                    w_cfg = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
                    w_cfg.max_nodes = mcts_cfg->max_nodes; // Match complexity
                } else {
                    b_cfg = mcts_get_preset(MCTS_PRESET_GRANDMASTER);
                    b_cfg.max_nodes = mcts_cfg->max_nodes;
                }
            }
            
            int res = play_game(weights, w_cfg, b_cfg, history, &steps, &reason, sp_cfg->temp, &rng, sp_cfg->endgame_prob);
            
            // Stats update (atomic)
            #pragma omp atomic
            completed_games++;
            
            if (res > 0) { 
                #pragma omp atomic 
                total_wins++; 
            } else if (res < 0) { 
                #pragma omp atomic
                total_losses++; 
            } else { 
                #pragma omp atomic
                total_draws++; 
            }
            
            // Save samples
            int batch_cnt = 0;
            for (int k = 0; k < steps; k++) {
                TrainingSample *s = &batch[batch_cnt++];
                s->state = history[k].state;
                s->history[0] = history[k].history[0];
                s->history[1] = history[k].history[1];
                memcpy(s->target_policy, history[k].policy, sizeof(s->target_policy));
                
                int sign = (history[k].player == WHITE) ? 1 : -1;
                // Reward shaping
                float val = (res == 0) ? 0.0f : (float)(res * sign);
                if (reason == END_RESIGNATION && res != 0) val *= 1.0f; // Resign is strong win
                else if (reason == END_MERCY) val *= 0.8f; // Mercy is softer
                else if (reason == END_CHECKMATE) val *= 1.0f;
                // Draw is 0
                
                s->target_value = val;
            }
            
            // Write to file (Critical section)
            #pragma omp critical(file_io)
            {
                dataset_save_append(sp_cfg->output_file, batch, batch_cnt);
                if (sp_cfg->on_game_complete) {
                    sp_cfg->on_game_complete(i, sp_cfg->games, res, steps, reason);
                }
                if (sp_cfg->on_progress) {
                    sp_cfg->on_progress(completed_games, sp_cfg->games, total_wins, total_losses, total_draws);
                }
            }
        }
        
        free(history);
        free(batch);
    }
}
