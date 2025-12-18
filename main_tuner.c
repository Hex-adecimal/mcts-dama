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
#define CK = 2.0  // Perturbation size (exploration) (c)
#define AK = 0.5  // Learning rate (step size) (a)
// SPSA usually decays a and c. a_k = a / (A + k)^alpha, c_k = c / k^gamma.
// Simplified: Fixed step for now or simple decay.

// Default weights
#define W_CAPTURE   10.0
#define W_PROMOTION 5.0
#define W_ADVANCE   0.5
#define W_CENTER    3.0
#define W_EDGE      2.0
#define W_BASE      2.0
#define W_THREAT    10.0

// Matches main_tournament.c structure
static void play_game_tuner(MCTSConfig cfg_A, MCTSConfig cfg_B, int *winner) {
    Arena arena_A, arena_B;
    arena_init(&arena_A, 256 * 1024 * 1024); // Smaller arena for speed (256MB)
    arena_init(&arena_B, 256 * 1024 * 1024);

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
        
        // Fast move: 0.05s per move for tuning speed
        Move best_move = mcts_search(root, current_arena, 0.05, *current_cfg, NULL, NULL);
        
        apply_move(&state, &best_move);
        turn++;
        
        if (turn > 400) { *winner = 0; break; } // Safety
    }

    arena_free(&arena_A);
    arena_free(&arena_B);
}

int main() {
    zobrist_init();
    srand(time(NULL));

    // Initial Parameters
    double theta[7] = { W_CAPTURE, W_PROMOTION, W_ADVANCE, W_CENTER, W_EDGE, W_BASE, W_THREAT };
    char *names[7]  = { "Capture", "Promotion", "Advance", "Center", "Edge", "Base", "Threat" };

    printf("=== SPSA Tuner Started ===\n");
    printf("Initial Weights: Cap=%.2f, Prom=%.2f, Adv=%.2f, Cen=%.2f, Edg=%.2f, Bas=%.2f, Thr=%.2f\n",
           theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6]);

    double a = 2.0; // Step size
    double c = 1.0; // Perturbation
    double A = 10.0;
    double alpha = 0.602;
    double gamma = 0.101;

    for (int k = 1; k <= SPSA_ITERATIONS; k++) {
        double ak = a / pow(k + A, alpha);
        double ck = c / pow(k, gamma);

        // Bernoulli vector (+1/-1)
        double delta[7];
        for(int i=0; i<7; i++) delta[i] = (rand() % 2 == 0) ? 1.0 : -1.0;

        // Theta Plus / Minus
        MCTSConfig cfg_plus, cfg_minus;
        // Base config: Tuned + Bias + Heuristics
        MCTSConfig base = { .ucb1_c = 1.414, .rollout_epsilon = 0.1, .expansion_threshold = 10,
                            .use_progressive_bias = 1, .bias_constant = 3.0, .use_ucb1_tuned = 1 };
        
        cfg_plus = base;
        cfg_minus = base;

        for(int i=0; i<7; i++) {
            double p_val = theta[i] + ck * delta[i];
            double m_val = theta[i] - ck * delta[i];
            
            // Safety Clamps (Weights shouldn't be negative generally, except Base Maybe)
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

        for(int i=0; i<7; i++) {
            double grad = y_diff / (2.0 * ck * delta[i]);
            theta[i] += ak * grad;
            if (theta[i] < 0) theta[i] = 0; // Keep non-negative
        }

        printf("Iter %d: Score=%.2f. New Weights: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", 
               k, y_diff, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6]);
    }
    
    printf("\n=== Optimization Complete ===\n");
    printf("Final Weights:\n");
    for(int i=0; i<7; i++) printf("  %s: %.4f\n", names[i], theta[i]);
    
    return 0;
}
