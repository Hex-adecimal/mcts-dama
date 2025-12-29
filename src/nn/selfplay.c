/**
 * selfplay.c - Self-play data generation logic
 */

#include "selfplay.h"
#include "rng.h"
#include "movegen.h"
#include "cnn.h"
#include "dataset.h"
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
#define MAX_MOVES_WITHOUT_CAPTURES 40

// =============================================================================
// HELPERS
// =============================================================================

static void add_dirichlet_noise(Node *root, RNG *rng, float eps, float alpha) {
    if (!root || root->num_children == 0) return;
    int n = root->num_children;
    float *noise = malloc(n * sizeof(float));
    float sum = 0;
    for (int i = 0; i < n; i++) {
        noise[i] = rng_gamma(rng, alpha);
        sum += noise[i];
    }
    for (int i = 0; i < n; i++) {
        root->children[i]->prior = (1.0f - eps) * root->children[i]->prior + eps * noise[i] / sum;
    }
    free(noise);
}

static int game_over(const GameState *s) {
    MoveList m;
    generate_moves(s, &m);
    return m.count == 0;
}



// =============================================================================
// GAME PLAY LOOP
// =============================================================================

static int play_game(
    int game_id, 
    const CNNWeights *weights, 
    MCTSConfig cfg_white, 
    MCTSConfig cfg_black,
    GameStep *history_buffer, 
    int *out_steps, 
    EndReason *out_reason, 
    float initial_temp, 
    RNG *rng,
    const MercyConfig *mercy
) {
    GameState state;
    init_game(&state);
    
    Arena arena;
    arena_init(&arena, 50 * 1024 * 1024); // 50MB per game
    
    // 2 random opening moves
    for (int i = 0; i < 2; i++) {
        MoveList list;
        generate_moves(&state, &list);
        if (list.count > 0) {
            int idx = rng_u32(rng) % list.count;
            apply_move(&state, &list.moves[idx]);
        }
    }
    
    int moves = 0;
    int max_moves = 200; // Hardcoded or config?
    float policy[CNN_POLICY_SIZE];
    
    while (!game_over(&state) && moves < max_moves) {
        float temp = (moves < DEFAULT_TEMP_THRESHOLD) ? initial_temp : 0.1f;
        
        MCTSConfig cfg = (state.current_player == WHITE) ? cfg_white : cfg_black;
        Node *root = mcts_create_root(state, &arena, cfg);
        
        // Noise
        if (moves < 30) {
            add_dirichlet_noise(root, rng, DEFAULT_DIRICHLET_EPSILON, DEFAULT_DIRICHLET_ALPHA);
        }
        
        mcts_search(root, &arena, 0.0, cfg, NULL, NULL); // Time 0.0 -> use max_nodes or implicit
        mcts_get_policy(root, policy, temp, &state);
        
        // Select move
        Move chosen = {0};
        // Reuse selection logic...
        // For brevity, simple probabilistic selection:
        double r_val = rng_f32(rng);
        double cumsum = 0;
        int best_child = 0;
        double max_visit = -1;
        
        // Recover children visits from policy? No, reuse children directly if possible.
        // But mcts_get_policy returns policy array.
        // Let's use root children.
        
        if (temp < 0.1f) {
            // Argmax visits
            for (int i=0; i<root->num_children; i++) {
                if (root->children[i]->visits > max_visit) {
                    max_visit = root->children[i]->visits;
                    best_child = i;
                }
            }
            chosen = root->children[best_child]->move_from_parent;
        } else {
            // Weighted random
             // Note: mcts_get_policy already applied temp to policy array
             // But mapping back to move is tricky without index.
             // Better to select from children based on visits^1/temp
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
             if (chosen.path[0] == 0) chosen = root->children[0]->move_from_parent; // Fallback
        }
        
        // Record
        history_buffer[moves].state = state;
        history_buffer[moves].player = state.current_player;
        memcpy(history_buffer[moves].policy, policy, sizeof(policy));
        if (moves >= 1) history_buffer[moves].history[0] = history_buffer[moves-1].state;
        if (moves >= 2) history_buffer[moves].history[1] = history_buffer[moves-2].state;
        
        apply_move(&state, &chosen);
        arena_reset(&arena);
        moves++;
        
        // Checks
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            arena_free(&arena);
            *out_steps = moves;
            *out_reason = END_40_MOVE;
            return 0; // Draw
        }
        
        // Resignation check (using NN directly)
        if (moves > 40 && weights) {
            CNNOutput out;
            // cnn_forward_with_history(weights, &state, ..., &out); 
            // Needs prototype or include cnn_inference. 
            // Assuming cnn.h provides it.
             GameState *h1 = (moves>=1) ? &history_buffer[moves-1].state : NULL;
             GameState *h2 = (moves>=2) ? &history_buffer[moves-2].state : NULL;
             cnn_forward_with_history(weights, &state, h1, h2, &out);
             
             if (out.value < -0.90f) { // Resign threshold
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
    unsigned long long start_time = time(NULL); // ms precision needed? use clock()
    
    int num_threads = (sp_cfg->parallel_threads > 0) ? sp_cfg->parallel_threads : 1;
    
    // CNN weights from mcts_cfg
    const CNNWeights *weights = (const CNNWeights*)mcts_cfg->cnn_weights;
    
    #pragma omp parallel num_threads(num_threads)
    {
        // Thread-local RNG
        RNG rng;
        unsigned int seed = time(NULL) ^ (omp_get_thread_num() * 12345);
        rng_seed(&rng, seed);
        
        GameStep *history = malloc(sp_cfg->max_moves * sizeof(GameStep));
        TrainingSample *batch = malloc(sp_cfg->max_moves * sizeof(TrainingSample));
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < sp_cfg->games; i++) {
            int steps = 0;
            EndReason reason;
            
            // Opponent config: copy of mcts_cfg
            MCTSConfig w_cfg = *mcts_cfg;
            MCTSConfig b_cfg = *mcts_cfg;
            
            int res = play_game(i, weights, w_cfg, b_cfg, history, &steps, &reason, sp_cfg->temp, &rng, &sp_cfg->mercy);
            
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
                if (sp_cfg->on_progress && (completed_games % 10 == 0 || completed_games == sp_cfg->games)) {
                    sp_cfg->on_progress(completed_games, sp_cfg->games, total_wins, total_losses, total_draws);
                }
            }
        }
        
        free(history);
        free(batch);
    }
}
