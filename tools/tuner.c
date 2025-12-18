#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "game.h"
#include "mcts.h"
#include "params.h"

// Tuner Configuration
#define SPSA_ITERATIONS 100
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

    // Initial Parameters (8 weights)
    double theta[8] = { W_CAPTURE, W_PROMOTION, W_ADVANCE, W_CENTER, W_EDGE, W_BASE, W_THREAT, W_LADY_ACTIVITY };
    char *names[8]  = { "Capture", "Promotion", "Advance", "Center", "Edge", "Base", "Threat", "LadyActivity" };

    printf("=== SPSA Tuner Started ===\n");
    printf("Initial Weights: Cap=%.2f, Prom=%.2f, Adv=%.2f, Cen=%.2f, Edg=%.2f, Bas=%.2f, Thr=%.2f, Lady=%.2f\n",
           theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7]);

    double a = 2.0; // Step size
    double c = 1.0; // Perturbation
    double A = 10.0;
    double alpha = 0.602;
    double gamma = 0.101;

    for (int k = 1; k <= SPSA_ITERATIONS; k++) {
        double ak = a / pow(k + A, alpha);
        double ck = c / pow(k, gamma);

        // Bernoulli vector (+1/-1)
        double delta[8];
        for(int i=0; i<8; i++) delta[i] = (rand() % 2 == 0) ? 1.0 : -1.0;

        // Theta Plus / Minus
        MCTSConfig cfg_plus, cfg_minus;
        // Base config: Tuned + Bias + Heuristics
        MCTSConfig base = { .ucb1_c = UCB1_C, .rollout_epsilon = ROLLOUT_EPSILON_SMART, .expansion_threshold = EXPANSION_THRESHOLD,
                            .use_progressive_bias = 1, .bias_constant = DEFAULT_BIAS_CONSTANT, .use_ucb1_tuned = 1 };
        
        cfg_plus = base;
        cfg_minus = base;

        for(int i=0; i<8; i++) {
            double p_val = theta[i] + ck * delta[i];
            double m_val = theta[i] - ck * delta[i];
            
            // Safety Clamps (Weights shouldn't be negative generally)
            if (p_val < 0) p_val = 0;
            if (m_val < 0) m_val = 0;
            
            // Assign
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

        // Play matches: Plus vs Minus
        int score_plus = 0;
        int games = GAMES_PER_ITERATION;
        
        #pragma omp parallel for reduction(+:score_plus)
        for (int g = 0; g < games; g++) {
            int winner = 0;
            // Alternate colors
            if (g % 2 == 0) play_game_tuner(cfg_plus, cfg_minus, &winner); // + is White
            else            play_game_tuner(cfg_minus, cfg_plus, &winner); // - is White

            // Simplified return from play_game: 1=White, -1=Black, 0=Draw
            if (g % 2 == 0) {
                if (winner == 1) score_plus++;      // Plus Wins
                else if (winner == -1) score_plus--; // Minus Wins
            } else {
                if (winner == 1) score_plus--;      // Minus Wins
                else if (winner == -1) score_plus++; // Plus Wins
            }
        }
        
        double y_diff = (double)score_plus / games; // Range [-1.0, 1.0]

        for(int i=0; i<8; i++) {
            double grad = y_diff / (2.0 * ck * delta[i]);
            theta[i] += ak * grad;
            if (theta[i] < 0) theta[i] = 0; // Keep non-negative
        }

        printf("Iter %d: Score=%.2f. Weights: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", 
               k, y_diff, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7]);
    }
    
    printf("\n=== Optimization Complete ===\n");
    printf("Final Weights:\n");
    for(int i=0; i<8; i++) printf("  %s: %.4f\n", names[i], theta[i]);
    
    return 0;
}
