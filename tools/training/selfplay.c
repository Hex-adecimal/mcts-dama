/**
 * selfplay.c - AlphaZero Self-Play Loop
 * 
 * Generates training data by playing games against itself (CNN vs CNN).
 * Uses MCTS for policy improvement.
 */

#include "game.h"
#include "mcts.h"
#include "cnn.h"
#include "dataset.h"
#include "params.h"
#include "movegen.h"
#include "mcts_presets.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

#define NUM_GAMES_TO_GENERATE   50000
#define TEMP_THRESHOLD_MOVES    60      // Exploration up to move 60 (was 30)


#define DIRICHLET_ALPHA         0.3     
#define DIRICHLET_EPSILON       0.25    
#define MAX_MOVES_GAME          200     

// =============================================================================
// THREAD-SAFE RNG
// =============================================================================

typedef struct {
    uint32_t state;
} RngState;

static void rng_init(RngState *rng, uint32_t seed) {
    rng->state = seed;
    if (rng->state == 0) rng->state = 1;
}

static uint32_t rng_next(RngState *rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static float rng_float(RngState *rng) {
    return (float)rng_next(rng) / (float)UINT32_MAX;
}

static float rng_gamma(RngState *rng, float alpha) {
    if (alpha < 1.0f) {
        return rng_gamma(rng, 1.0f + alpha) * powf(rng_float(rng), 1.0f / alpha);
    }
    float d = alpha - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    float v, x, u;
    while (1) {
        do {
            float u1 = rng_float(rng);
            float u2 = rng_float(rng);
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
            x = z;
            v = 1.0f + c * x;
        } while (v <= 0.0f);
        v = v * v * v;
        u = rng_float(rng);
        if (u < 1.0f - 0.0331f * x * x * x * x) return d * v;
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v;
    }
}

// =============================================================================
// LOGIC
// =============================================================================

typedef struct {
    float policy[CNN_POLICY_SIZE];
    GameState state;
    GameState history[CNN_HISTORY_T - 1];
    int player;
} GameStep;

void add_dirichlet_noise(Node *root, RngState *rng, float epsilon, float alpha) {
    if (!root || root->num_children == 0) return;
    int count = root->num_children;
    float *noise = malloc(count * sizeof(float));
    float sum_noise = 0.0f;
    for (int i = 0; i < count; i++) {
        noise[i] = rng_gamma(rng, alpha);
        sum_noise += noise[i];
    }
    for (int i = 0; i < count; i++) {
        Node *child = root->children[i];
        float n = noise[i] / sum_noise;
        child->prior = (1.0f - epsilon) * child->prior + epsilon * n;
    }
    free(noise);
}

static int is_game_terminated(const GameState *state) {
    MoveList ml;
    generate_moves(state, &ml);
    return (ml.count == 0);
}

static int check_mercy_rule(const GameState *state, int moves) {
    if (moves < 60) return 0;
    int white_count = __builtin_popcountll(state->white_pieces | state->white_ladies);
    int black_count = __builtin_popcountll(state->black_pieces | state->black_ladies);
    int white_ladies = __builtin_popcountll(state->white_ladies);
    int black_ladies = __builtin_popcountll(state->black_ladies);
    int white_score = white_count + white_ladies;
    int black_score = black_count + black_ladies;
    if (white_score >= black_score + 4 && black_count <= 3) return 1;
    if (black_score >= white_score + 4 && white_count <= 3) return -1;
    return 0;
}

int play_game(int game_id, const CNNWeights *weights, MCTSConfig cfg_vanilla, MCTSConfig cfg_puct, GameStep *history, int *history_len, RngState *rng) {
    GameState state;
    init_game(&state);
    Arena arena;
    arena_init(&arena, 1024 * 1024 * 100); 
    int moves = 0;
    float policy[CNN_POLICY_SIZE];
    int hybrid_mode = (game_id % 2 == 0);
    
    while (!is_game_terminated(&state) && moves < MAX_MOVES_GAME) {
        float temp = (moves < TEMP_THRESHOLD_MOVES) ? 1.0f : 0.05f; 
        MCTSConfig config;
        int is_cnn_turn = 1;
        if (hybrid_mode == 0) {
            config = cfg_puct;
            config.cnn_weights = (void*)weights;
        } else {
            int cnn_is_white = ((game_id / 2) % 2 == 0);
            is_cnn_turn = (state.current_player == WHITE) == cnn_is_white;
            if (is_cnn_turn) {
                config = cfg_puct;
                config.cnn_weights = (void*)weights;
            } else {
                config = cfg_vanilla; // Actually acts as 'cfg_opponent' passed from main
            }
        }
        
        Node *root = mcts_create_root(state, &arena, config);
        if (moves < 30 && is_cnn_turn) {
            mcts_search(root, &arena, 0.0, config, NULL, NULL); 
            add_dirichlet_noise(root, rng, DIRICHLET_EPSILON, DIRICHLET_ALPHA);
        }
        mcts_search(root, &arena, 0.0, config, NULL, NULL);
        mcts_get_policy(root, policy, temp, &state);
        
        Move chosen_move = {0};
        if (temp < 0.1f) {
            float max_p = -1;
            int max_i = -1;
            for(int i=0; i<CNN_POLICY_SIZE; i++) if(policy[i] > max_p) { max_p = policy[i]; max_i = i; }
            for(int i=0; i<root->num_children; i++) {
                Node *child = root->children[i];
                if (cnn_move_to_index(&child->move_from_parent, state.current_player) == max_i) {
                    chosen_move = child->move_from_parent;
                    break;
                }
            }
        } else {
            double sum_visits = 0;
            double exponent = 1.0 / temp;
            for(int i=0; i<root->num_children; i++) sum_visits += pow(root->children[i]->visits, exponent);
            double sample = rng_float(rng) * sum_visits;
            double current = 0;
            Move fallback = {0};
            for(int i=0; i<root->num_children; i++) {
                Node *child = root->children[i];
                if (i==0) fallback = child->move_from_parent;
                current += pow(child->visits, exponent);
                if (current >= sample) { chosen_move = child->move_from_parent; break; }
            }
            if (chosen_move.path[0] == 0 && chosen_move.path[1] == 0) chosen_move = fallback;
        }
        
        history[moves].state = state;
        history[moves].player = state.current_player;
        memcpy(history[moves].policy, policy, sizeof(policy));
        if (moves >= 1) history[moves].history[0] = history[moves - 1].state;
        else memset(&history[moves].history[0], 0, sizeof(GameState));
        if (moves >= 2) history[moves].history[1] = history[moves - 2].state;
        else memset(&history[moves].history[1], 0, sizeof(GameState));
        
        apply_move(&state, &chosen_move);
        arena_reset(&arena);
        moves++;
        
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) break;
        if (check_mercy_rule(&state, moves) != 0) break;
        
        // Check for resignation/early stop using Value Head
        CNNOutput out;
        const GameState *h1 = (moves >= 1) ? &history[moves-1].state : NULL;
        const GameState *h2 = (moves >= 2) ? &history[moves-2].state : NULL;
        cnn_forward_with_history(weights, &state, h1, h2, &out);
        
        if (out.value < -0.85f && moves > 40) break; // Resign earlier (-0.85 was -0.95)
    }
    
    arena_free(&arena);
    *history_len = moves;
    int mercy = check_mercy_rule(&state, moves);
    if (mercy != 0) return mercy;
    if (moves >= MAX_MOVES_GAME || state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) return 0;
    return (state.current_player == WHITE) ? -1 : 1; 
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char **argv) {
    int num_games = NUM_GAMES_TO_GENERATE;
    int tid = 0;
    if (argc > 1) num_games = atoi(argv[1]);
    if (argc > 2) tid = atoi(argv[2]);
    
    printf("=== AlphaZero Self-Play Generator (Thread %d) ===\n", tid);
    
    zobrist_init();
    init_move_tables();
    
    CNNWeights weights;
    cnn_init(&weights);
    cnn_load_weights(&weights, "models/cnn_weights.bin");
    
    MCTSConfig cfg_puct = { .ucb1_c = 1.5, .use_puct = 1, .puct_c = PUCT_C, .cnn_weights = (void*)&weights, .max_nodes = 800 };
    MCTSConfig opponents[5];
    opponents[0] = mcts_get_preset(MCTS_PRESET_VANILLA);
    opponents[1] = mcts_get_preset(MCTS_PRESET_PURE_VANILLA);
    opponents[2] = mcts_get_preset(MCTS_PRESET_SMART_ROLLOUTS); // More heuristic
    opponents[3] = mcts_get_preset(MCTS_PRESET_SOLVER_ONLY);   // Strong endgame
    opponents[4] = mcts_get_preset(MCTS_PRESET_WEIGHTS_ONLY);  // Weak but different
    
    // Set consistent node limit for all opponents to ensure speed
    for(int k=0; k<5; k++) opponents[k].max_nodes = 1600;

    char filename[128];
    sprintf(filename, "data/selfplay/raw_gen_%d.bin", tid);
    remove(filename);
    
    RngState rng;
    rng_init(&rng, time(NULL) + tid * 100);
    
    int batch_cap = 2000;
    TrainingSample *batch = malloc(batch_cap * sizeof(TrainingSample));
    int batch_count = 0, total_samples = 0;
    
    for (int i = 0; i < num_games; i++) {
        GameStep *history = malloc(MAX_MOVES_GAME * sizeof(GameStep));
        int steps = 0;
        
        // Select opponent based on game ID (Deterministic per thread)
        MCTSConfig opponent_config = opponents[(i + tid) % 5];
        
        int result = play_game(i, &weights, opponent_config, cfg_puct, history, &steps, &rng);
        total_samples += steps;
        
        for (int k = 0; k < steps; k++) {
            if (batch_count >= batch_cap) { dataset_save_append(filename, batch, batch_count); batch_count = 0; }
            TrainingSample *s = &batch[batch_count++];
            s->state = history[k].state;
            s->history[0] = history[k].history[0];
            s->history[1] = history[k].history[1];
            memcpy(s->target_policy, history[k].policy, sizeof(s->target_policy));
            int current_player_sign = (history[k].player == WHITE) ? 1 : -1;
            s->target_value = (result == 0) ? 0.0f : (float)(result * current_player_sign);
        }
        free(history);
        if (tid == 0 && i % 10 == 0) { printf("\rGames: %d/%d (Samples: %d)", i, num_games, total_samples); fflush(stdout); }
    }
    
    if (batch_count > 0) dataset_save_append(filename, batch, batch_count);
    free(batch);
    cnn_free(&weights);
    printf("\nDone. Thread %d total samples: %d\n", tid, total_samples);
    return 0;
}
