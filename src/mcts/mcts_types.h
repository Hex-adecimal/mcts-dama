#ifndef MCTS_TYPES_H
#define MCTS_TYPES_H

#include "../core/game.h"
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <stdatomic.h>


// =============================================================================
// SOLVER STATUS
// =============================================================================

typedef enum {
    SOLVED_NONE = 0,
    SOLVED_WIN = 1,
    SOLVED_LOSS = -1,
    SOLVED_DRAW = 2
} SolverStatus;

// =============================================================================
// MCTS CONFIGURATION
// =============================================================================

typedef struct {
    // Core MCTS Parameters
    double ucb1_c;              // Exploration constant (e.g., 1.414)
    double rollout_epsilon;     // Epsilon for epsilon-greedy rollouts
    double draw_score;          // Score assigned for draws (e.g., 0.5)
    int expansion_threshold;    // Minimum visits before expansion
    
    // Feature Flags
    int verbose;                // Print search statistics
    int use_lookahead;          // 1-ply lookahead in simulation
    int use_tree_reuse;         // Reuse tree between moves
    int use_ucb1_tuned;         // UCB1-Tuned selection
    int use_tt;                 // Transposition Table
    int use_solver;             // MCTS-Solver (proven wins/losses)
    int use_progressive_bias;   // Progressive Bias
    double bias_constant;       // Weight for Progressive Bias
    int use_fpu;                // First Play Urgency
    double fpu_value;
    int use_decaying_reward;    // Decaying Reward
    double decay_factor;

    // Heuristic Weights (for rollout policy)
    struct {
        double w_capture;
        double w_promotion;
        double w_advance;
        double w_center;
        double w_edge;
        double w_base;
        double w_threat;
        double w_lady_activity;
    } weights;

    // PUCT Neural Network
    int use_puct;          // Enable PUCT selection
    double puct_c;         // PUCT exploration constant
    void *nn_weights;      // Pointer to loaded NNWeights (MLP, opaque)
    void *cnn_weights;     // Pointer to loaded CNNWeights (CNN, opaque)
    int max_nodes;         // Hard limit on iterations (0 = unlimited/time-based)
} MCTSConfig;

// =============================================================================
// MCTS NODE
// =============================================================================

typedef struct Node {
    GameState state;
    Move move_from_parent;
    int player_who_just_moved;
    
    struct Node *parent;
    struct Node **children;
    int num_children;
    
    MoveList untried_moves;
    int is_terminal;
    
    // Concurrency & Stats
    _Atomic int visits;         // Atomic visit count
    _Atomic int virtual_loss;   // Virtual loss for tree parallelization
    
    double score;
    double sum_sq_score;
    double heuristic_score;
    float prior;
    float *cached_policy; 
    
    pthread_mutex_t lock;       // Protects child expansion/modification
    
    int8_t status;  // SolverStatus
} Node;

// =============================================================================
// MCTS STATISTICS
// =============================================================================

typedef struct {
    int total_moves;
    long total_iterations;
    long current_move_iterations;
    long total_depth;
    double total_time;
    size_t total_memory;
} MCTSStats;

#endif // MCTS_TYPES_H
