#ifndef MCTS_H
#define MCTS_H

#include "game.h"
#include <stdlib.h>

// =============================================================================
// MEMORY MANAGEMENT (Arena Allocator)
// =============================================================================
// Arena, arena_init, arena_alloc, arena_reset, arena_free

/**
 * Simple Arena Allocator.
 * Allocates MCTS nodes in a contiguous memory block, avoiding fragmentation
 * and the overhead of thousands of malloc/free calls.
 */
typedef struct {
    unsigned char *buffer;
    size_t size;
    size_t offset;
} Arena;

void arena_init(Arena *a, size_t total_size);
void* arena_alloc(Arena *a, size_t bytes);
void arena_reset(Arena *a);
void arena_free(Arena *a);

// =============================================================================
// TYPES & STRUCTURES
// =============================================================================
// SolverStatus, MCTSConfig, TranspositionTable, MCTSStats, Node

typedef enum {
    SOLVED_NONE = 0,
    SOLVED_WIN = 1,
    SOLVED_LOSS = -1,
    SOLVED_DRAW = 2
} SolverStatus;

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

    // First Play Urgency (Plugin)
    int use_fpu;
    double fpu_value;
    
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
} MCTSConfig;

//Transposition Table: maps Zobrist Hash -> Node pointer.
typedef struct {
    struct Node **buckets;
    size_t size;
    size_t mask;
    size_t count;
    size_t collisions;
} TranspositionTable;

//Statistics tracker for MCTS performance analysis.
typedef struct {
    int total_moves;
    long total_iterations;
    long total_depth;
    double total_time;
    size_t total_memory;
} MCTSStats;

//MCTS tree node.
typedef struct Node {
    GameState state;
    Move move_from_parent;
    int player_who_just_moved;
    
    struct Node *parent;
    struct Node **children;
    int num_children;
    
    MoveList untried_moves;
    int is_terminal;
    
    int visits;
    double score;
    double sum_sq_score;
    double heuristic_score;
    
    int8_t status;  // SolverStatus
} Node;

// =============================================================================
// MCTS API
// =============================================================================
// mcts_create_root, mcts_search


//Creates the root node for the MCTS search.
Node* mcts_create_root(GameState state, Arena *arena, MCTSConfig config);

/**
 * Executes the MCTS search algorithm.
 * @param root Pointer to the root node.
 * @param arena Pointer to the arena allocator.
 * @param time_limit_seconds Maximum search time.
 * @param config Search configuration.
 * @param stats Optional stats to update.
 * @param out_new_root Optional: returns chosen child for tree reuse.
 * @return The best move found.
 */
Move mcts_search(Node *root, Arena *arena, double time_limit_seconds, MCTSConfig config,
                 MCTSStats *stats, Node **out_new_root);

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================
// find_child_by_move, get_tree_depth, states_equal

Node* find_child_by_move(Node *parent, const Move *move);
int get_tree_depth(Node *node);
int states_equal(const GameState *s1, const GameState *s2);

// Debug / Stats Helper
// double mcts_get_avg_root_ucb(Node *root, MCTSConfig config); // Moved to debug.h

// Core functions exposed for debug/tools
double calculate_ucb1_score(Node *child, MCTSConfig config);

#endif // MCTS_H