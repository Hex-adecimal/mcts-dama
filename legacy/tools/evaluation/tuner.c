#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "game.h"
#include "movegen.h"
#include "mcts.h"
#include "params.h"

#define TIME_TUNER              0.05
#define MAX_GAME_TURNS_TUNER    400


// Tuner Configuration
#define SPSA_ITERATIONS 200
#define GAMES_PER_ITERATION 20 // 10 pairs

// Matches main_tournament.c structure
static void play_game_tuner(MCTSConfig cfg_A, MCTSConfig cfg_B, int *winner) {
    Arena arena_A, arena_B;
    arena_init(&arena_A, ARENA_SIZE_TUNER);
    arena_init(&arena_B, ARENA_SIZE_TUNER);

    GameState state;
    init_game(&state);
    
    int turn = 0;
    while(1) {
        if (state.moves_without_captures >= MAX_MOVES_WITHOUT_CAPTURES) {
            *winner = 0; // Draw
            break;
        }

        MoveList list;
        generate_moves(&state, &list);
        if (list.count == 0) {
            *winner = (state.current_player == WHITE) ? -1 : 1; // 1 = White Wins, -1 = Black Wins
            break;
        }

        // Determine who plays
        MCTSConfig *current_cfg = (state.current_player == WHITE) ? &cfg_A : &cfg_B;
        Arena *current_arena    = (state.current_player == WHITE) ? &arena_A : &arena_B;

        arena_reset(current_arena);
        Node *root = mcts_create_root(state, current_arena, *current_cfg);
        
        Move best_move = mcts_search(root, current_arena, TIME_TUNER, *current_cfg, NULL, NULL);
        
        apply_move(&state, &best_move);
        turn++;
        
        if (turn > MAX_GAME_TURNS_TUNER) { *winner = 0; break; }
    }

    arena_free(&arena_A);
    arena_free(&arena_B);
}

int main() {
    zobrist_init();
    init_move_tables();
    srand(time(NULL));

    // Optimize ONLY 8 Heuristic Weights
    #define NUM_PARAMS 8
    // Initial weights from params.h
    double theta[NUM_PARAMS] = { 
        W_CAPTURE, W_PROMOTION, W_ADVANCE, W_CENTER, W_EDGE, W_BASE, W_THREAT, W_LADY_ACTIVITY
    };
    char *names[NUM_PARAMS]  = { 
        "Capture", "Promotion", "Advance", "Center", "Edge", "Base", "Threat", "LadyActivity"
    };

    printf("=== SPSA Tuner (Weights Only) Started ===\n");
    // Print initial
    printf("Init: Cap=%.2f, Prom=%.2f, Adv=%.2f, Cen=%.2f, Edg=%.2f, Bas=%.2f, Thr=%.2f, Lady=%.2f\n",
           theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7]);

    double a = 4.0; // Step size
    double c = 2.0; // Perturbation (High for weights ~1-10)
    double A = 20.0;
    double alpha = 0.602;
    double gamma = 0.101;

    for (int k = 1; k <= SPSA_ITERATIONS; k++) {
        double ak = a / pow(k + A, alpha);
        double ck = c / pow(k, gamma);

        // Bernoulli vector
        double delta[NUM_PARAMS];
        for(int i=0; i<NUM_PARAMS; i++) delta[i] = (rand() % 2 == 0) ? 1.0 : -1.0;

        // Base Config: Grandmaster Structure (Fixed)
        MCTSConfig base = { 
            .ucb1_c = UCB1_C, // 1.30
            .rollout_epsilon = ROLLOUT_EPSILON_RANDOM, 
            .expansion_threshold = EXPANSION_THRESHOLD,
            .use_progressive_bias = 1, 
            .bias_constant = DEFAULT_BIAS_CONSTANT, // 1.2
            .use_ucb1_tuned = 1,
            .use_tt = 1,
            .use_solver = 1,
            .use_fpu = 1,
            .fpu_value = FPU_VALUE, // 100.3
            .use_decaying_reward = 1,
            .decay_factor = DEFAULT_DECAY_FACTOR // 0.999
        };
        
        MCTSConfig cfg_plus = base;
        MCTSConfig cfg_minus = base;

        for(int i=0; i<NUM_PARAMS; i++) {
            double change = ck * delta[i];
            double p_val = theta[i] + change;
            double m_val = theta[i] - change;
            
            if (p_val < 0) p_val = 0;
            if (m_val < 0) m_val = 0;
            
             switch(i) {
                case 0: cfg_plus.weights.w_capture = p_val; cfg_minus.weights.w_capture = m_val; break;
                case 1: cfg_plus.weights.w_promotion = p_val; cfg_minus.weights.w_promotion = m_val; break;
                case 2: cfg_plus.weights.w_advance = p_val; cfg_minus.weights.w_advance = m_val; break;
                case 3: cfg_plus.weights.w_center = p_val; cfg_minus.weights.w_center = m_val; break;
                case 4: cfg_plus.weights.w_edge = p_val; cfg_minus.weights.w_edge = m_val; break;
                case 5: cfg_plus.weights.w_base = p_val; cfg_minus.weights.w_base = m_val; break;
                case 6: cfg_plus.weights.w_threat = p_val; cfg_minus.weights.w_threat = m_val; break;
                case 7: cfg_plus.weights.w_lady_activity = p_val; cfg_minus.weights.w_lady_activity = m_val; break;
            }
        }

        // Play matches
        int score_plus = 0;
        int games = GAMES_PER_ITERATION;
        
        #pragma omp parallel for reduction(+:score_plus)
        for (int g = 0; g < games; g++) {
            int winner = 0;
            if (g % 2 == 0) play_game_tuner(cfg_plus, cfg_minus, &winner); 
            else            play_game_tuner(cfg_minus, cfg_plus, &winner);

            if (g % 2 == 0) {
                if (winner == 1) score_plus++;      
                else if (winner == -1) score_plus--; 
            } else {
                if (winner == 1) score_plus--;      
                else if (winner == -1) score_plus++; 
            }
        }
        
        double y_diff = (double)score_plus / games;

        for(int i=0; i<NUM_PARAMS; i++) {
            double grad = y_diff / (2.0 * ck * delta[i]);
            theta[i] += ak * grad;
            if (theta[i] < 0) theta[i] = 0;
        }

        printf("Iter %d: Score=%.2f. Weights: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", 
               k, y_diff, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7]);
    }
    
    printf("\n=== Optimization Complete ===\n");
    printf("Final Weights:\n");
    for(int i=0; i<NUM_PARAMS; i++) printf("  %s: %.4f\n", names[i], theta[i]);
    
    return 0;
}
