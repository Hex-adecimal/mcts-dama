/**
 * bootstrap.c - Heuristic Data Generator
 * 
 * Generates initial training data from "Grandmaster" MCTS (Heuristic-only)
 * to kickstart the imitation learning process.
 */

#include "game.h"
#include "movegen.h"
#include "mcts.h"
#include "cnn.h"
#include "dataset.h"
#include "params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define min(a,b) ((a) < (b) ? (a) : (b))

#define OUTPUT_FILE "data/bootstrap/heuristic_samples.bin"

// =============================================================================
// HEURISTIC SELF-PLAY
// =============================================================================

static int play_heuristic_game(TrainingSample *buffer, int buffer_offset, int max_samples) {
    GameState game_states[MAX_GAME_TURNS];
    float game_policies[MAX_GAME_TURNS][CNN_POLICY_SIZE];
    int game_length = 0;
    
    GameState state;
    init_game(&state);
    
    MCTSConfig config = {
        .ucb1_c = UCB1_C,
        .rollout_epsilon = ROLLOUT_EPSILON_RANDOM, 
        .draw_score = DRAW_SCORE,
        .expansion_threshold = EXPANSION_THRESHOLD,
        .verbose = 0,
        .use_lookahead = 0,
        .use_tree_reuse = 0,
        .use_ucb1_tuned = 1,
        .use_tt = 1,
        .use_solver = 1,
        .use_progressive_bias = 0,
        .use_fpu = 1,
        .fpu_value = FPU_VALUE,
        .use_decaying_reward = 0,
        .weights = {
            .w_capture = W_CAPTURE,
            .w_promotion = W_PROMOTION,
            .w_advance = W_ADVANCE,
            .w_center = W_CENTER,
            .w_edge = W_EDGE,
            .w_base = W_BASE,
            .w_threat = W_THREAT,
            .w_lady_activity = W_LADY_ACTIVITY
        },
        .use_puct = 0,
        .cnn_weights = NULL,
        .max_nodes = 1000 
    };
    
    while (game_length < MAX_GAME_TURNS) {
        MoveList moves;
        generate_moves(&state, &moves);
        
        if (moves.count == 0) break;
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) break;
        
        game_states[game_length] = state;
        memset(game_policies[game_length], 0, sizeof(game_policies[game_length]));
        
        Arena arena;
        arena_init(&arena, ARENA_SIZE_TOURNAMENT); 
        Node *root = mcts_create_root(state, &arena, config);
        
        Move best = mcts_search(root, &arena, 10.0, config, NULL, NULL);
        
        if (root->visits > 0 && root->num_children > 0) {
            for (int i = 0; i < root->num_children; i++) {
                int idx = cnn_move_to_index(&root->children[i]->move_from_parent, state.current_player);
                if (idx >= 0 && idx < CNN_POLICY_SIZE) {
                    game_policies[game_length][idx] = (float)root->children[i]->visits / (float)root->visits;
                }
            }
        }
        
        if (game_length < 30 && root->visits > 0) {
            int total_visits = root->visits;
            int r = rand() % total_visits;
            int cumulative = 0;
            for (int i = 0; i < root->num_children; i++) {
                cumulative += root->children[i]->visits;
                if (r < cumulative) {
                    best = root->children[i]->move_from_parent;
                    break;
                }
            }
        }
        
        game_length++;
        apply_move(&state, &best);
        arena_free(&arena);
    }
    
    MoveList final_moves;
    generate_moves(&state, &final_moves);
    
    float result;
    if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
        result = 0.0f;
    } else if (final_moves.count == 0) {
        result = 1.0f;
    } else {
        result = 0.0f;
    }
    
    int samples_stored = 0;
    for (int i = game_length - 1; i >= 0 && samples_stored < max_samples; i--) {
        buffer[buffer_offset + samples_stored].state = game_states[i];
        if (i >= 1) {
            buffer[buffer_offset + samples_stored].history[0] = game_states[i - 1];
        } else {
            memset(&buffer[buffer_offset + samples_stored].history[0], 0, sizeof(GameState));
        }
        if (i >= 2) {
            buffer[buffer_offset + samples_stored].history[1] = game_states[i - 2];
        } else {
            memset(&buffer[buffer_offset + samples_stored].history[1], 0, sizeof(GameState));
        }
        
        memcpy(buffer[buffer_offset + samples_stored].target_policy, game_policies[i], CNN_POLICY_SIZE * sizeof(float));
        buffer[buffer_offset + samples_stored].target_value = result;
        
        result = -result;
        samples_stored++;
    }
    
    return samples_stored;
}

// =============================================================================
// MAIN
// =============================================================================

int main(void) {
    printf("=== Dama Italiana Bootstrap Generator ===\n\n");
    
    srand((unsigned)time(NULL));
    zobrist_init();
    init_move_tables();
    
    printf("✓ Using Grandmaster MCTS for high-quality data generation\n");
    
    #define MAX_TOTAL_SAMPLES 1000000 
    #define TARGET_GAMES 10000

    TrainingSample *buffer = malloc(MAX_TOTAL_SAMPLES * sizeof(TrainingSample));
    if (!buffer) {
        fprintf(stderr, "Failed to allocate sample buffer\n");
        return 1;
    }
    
    printf("Configuration:\n");
    printf("  Target Games: %d\n", TARGET_GAMES);
    printf("  Output: %s\n", OUTPUT_FILE);
    printf("\nGenerating data...\n\n");
    
    int total_samples = 0;
    clock_t start_time = clock();
    
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int g = 0; g < TARGET_GAMES; g++) {
            int current_total;
            #pragma omp atomic read
            current_total = total_samples;
            
            if (current_total >= MAX_TOTAL_SAMPLES) continue;
            
            int my_offset;
            #pragma omp critical(buffer_reserve)
            {
                my_offset = total_samples;
                total_samples += 150; 
            }
            
            int collected = play_heuristic_game(buffer, my_offset, min(150, MAX_TOTAL_SAMPLES - my_offset));
            
            #pragma omp critical(stats_update)
            {
                total_samples = total_samples - 150 + collected;
                if ((g + 1) % 50 == 0) {
                    printf("\r  Progress: %d/%d games (%d samples)", g + 1, TARGET_GAMES, total_samples);
                    fflush(stdout);
                }
            }
        }
    }
    
    printf("\n\nSaving dataset...\n");
    if (dataset_save(OUTPUT_FILE, buffer, total_samples) != 0) {
        fprintf(stderr, "Failed to save data\n");
    }
    
    free(buffer);
    printf("\n=== Bootstrap Complete: %d samples ===\n", total_samples);
    
    return 0;
}
