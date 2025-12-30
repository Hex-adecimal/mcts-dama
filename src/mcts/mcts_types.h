/**
 * mcts_types.h - MCTS Core Types and Infrastructure
 * 
 * Consolidated from: mcts_types.h, mcts_internal.h, mcts_presets.h
 * Contains: Node, Config, Arena, TT, Presets
 */

#ifndef MCTS_TYPES_H
#define MCTS_TYPES_H

#include "../core/game.h"
#include "../params.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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
    double ucb1_c;
    double rollout_epsilon;
    double draw_score;
    int expansion_threshold;
    
    int verbose;
    int use_lookahead;
    int use_tree_reuse;
    int use_ucb1_tuned;
    int use_tt;
    int use_solver;
    int use_progressive_bias;
    double bias_constant;
    int use_fpu;
    double fpu_value;
    int use_decaying_reward;
    double decay_factor;
    
    // Fast rollout: early termination on material advantage, shorter depth
    int use_fast_rollout;
    int fast_rollout_depth;  // Max depth when fast rollout enabled (default: 50)

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

    int use_puct;
    double puct_c;
    void *nn_weights;
    void *cnn_weights;
    int max_nodes;
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
    
    _Atomic int visits;
    _Atomic int virtual_loss;
    
    double score;
    double sum_sq_score;
    double heuristic_score;
    float prior;
    float *cached_policy;
    
    pthread_mutex_t lock;
    int8_t status;  // SolverStatus
} Node;

// =============================================================================
// MCTS STATISTICS
// =============================================================================

typedef struct {
    int total_moves;
    long total_iterations;
    long total_nodes;
    long current_move_iterations;
    long total_depth;
    double total_time;
    size_t total_memory;
    // Debug stats for tree analysis
    long total_expansions;      // How many nodes were expanded
    long total_policy_cached;   // How many times CNN policy was computed
} MCTSStats;

// =============================================================================
// ARENA ALLOCATOR
// =============================================================================

typedef struct {
    unsigned char *buffer;
    size_t size;
    size_t offset;
    pthread_mutex_t lock;
} Arena;

static inline void arena_init(Arena *a, size_t total_size) {
    a->buffer = malloc(total_size);
    if (!a->buffer) {
        fprintf(stderr, "FATAL: Malloc failed for Arena of size %zu\n", total_size);
        exit(1);
    }
    a->size = total_size;
    a->offset = 0;
    pthread_mutex_init(&a->lock, NULL);
}

static inline void* arena_alloc(Arena *a, size_t bytes) {
    pthread_mutex_lock(&a->lock);
    uintptr_t current_ptr = (uintptr_t)(a->buffer + a->offset);
    uintptr_t padding = (8 - (current_ptr % 8)) % 8;
    
    if (a->offset + padding + bytes > a->size) {
        pthread_mutex_unlock(&a->lock);
        fprintf(stderr, "FATAL: Arena Out of Memory! (Size: %zu)\n", a->size);
        exit(1);
    }
    
    void *ptr = a->buffer + a->offset + padding;
    a->offset += padding + bytes;
    pthread_mutex_unlock(&a->lock);
    return ptr;
}

static inline void arena_reset(Arena *a) {
    pthread_mutex_lock(&a->lock);
    a->offset = 0;
    pthread_mutex_unlock(&a->lock);
}

static inline void arena_free(Arena *a) {
    pthread_mutex_destroy(&a->lock);
    free(a->buffer);
}

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

typedef struct {
    Node **buckets;
    size_t size;
    size_t mask;
    size_t count;
    size_t collisions;
    pthread_mutex_t lock;
} TranspositionTable;

static inline int states_equal(const GameState *s1, const GameState *s2) {
    return s1->piece[WHITE][PAWN] == s2->piece[WHITE][PAWN] &&
           s1->piece[WHITE][LADY] == s2->piece[WHITE][LADY] &&
           s1->piece[BLACK][PAWN] == s2->piece[BLACK][PAWN] &&
           s1->piece[BLACK][LADY] == s2->piece[BLACK][LADY] &&
           s1->current_player == s2->current_player &&
           s1->moves_without_captures == s2->moves_without_captures;
}

static inline TranspositionTable* tt_create(size_t size) {
    TranspositionTable *tt = malloc(sizeof(TranspositionTable));
    tt->size = size;
    tt->mask = size - 1;
    tt->count = 0;
    tt->collisions = 0;
    tt->buckets = calloc(size, sizeof(Node*));
    pthread_mutex_init(&tt->lock, NULL);
    return tt;
}

static inline void tt_free(TranspositionTable *tt) {
    if (tt) {
        pthread_mutex_destroy(&tt->lock);
        free(tt->buckets);
        free(tt);
    }
}

static inline Node* tt_lookup(TranspositionTable *tt, const GameState *state) {
    if (!tt) return NULL;
    size_t idx = state->hash & tt->mask;
    
    pthread_mutex_lock(&tt->lock);
    Node *node = tt->buckets[idx];
    if (node && node->state.hash == state->hash && states_equal(&node->state, state)) {
        pthread_mutex_unlock(&tt->lock);
        return node;
    }
    pthread_mutex_unlock(&tt->lock);
    return NULL;
}

static inline void tt_insert(TranspositionTable *tt, Node *node) {
    if (!tt) return;
    size_t idx = node->state.hash & tt->mask;
    
    pthread_mutex_lock(&tt->lock);
    if (tt->buckets[idx] != NULL) tt->collisions++;
    tt->buckets[idx] = node;
    tt->count++;
    pthread_mutex_unlock(&tt->lock);
}

// =============================================================================
// PRESETS
// =============================================================================

typedef enum {
    MCTS_PRESET_PURE_VANILLA,
    MCTS_PRESET_VANILLA,
    MCTS_PRESET_GRANDMASTER,
    MCTS_PRESET_ALPHA_ZERO,
    MCTS_PRESET_TT_ONLY,
    MCTS_PRESET_SOLVER_ONLY,
    MCTS_PRESET_TUNED_ONLY,
    MCTS_PRESET_FPU_ONLY,
    MCTS_PRESET_DECAY_ONLY,
    MCTS_PRESET_LOOKAHEAD_ONLY,
    MCTS_PRESET_TREE_REUSE_ONLY,
    MCTS_PRESET_WEIGHTS_ONLY,
    MCTS_PRESET_SMART_ROLLOUTS,
    MCTS_PRESET_PROG_BIAS_ONLY
} MCTSPreset;

static inline void apply_weights(MCTSConfig *c) {
    c->weights.w_capture = W_CAPTURE;
    c->weights.w_promotion = W_PROMOTION;
    c->weights.w_advance = W_ADVANCE;
    c->weights.w_center = W_CENTER;
    c->weights.w_edge = W_EDGE;
    c->weights.w_base = W_BASE;
    c->weights.w_threat = W_THREAT;
    c->weights.w_lady_activity = W_LADY_ACTIVITY;
}

static inline MCTSConfig mcts_get_preset(MCTSPreset preset) {
    MCTSConfig cfg;
    memset(&cfg, 0, sizeof(MCTSConfig));

    cfg.ucb1_c = UCB1_C;
    cfg.draw_score = DRAW_SCORE;
    cfg.expansion_threshold = EXPANSION_THRESHOLD;
    cfg.rollout_epsilon = ROLLOUT_EPSILON_RANDOM;
    cfg.use_lookahead = 0;
    cfg.use_tree_reuse = 0;

    switch (preset) {
        case MCTS_PRESET_PURE_VANILLA:
            break;
        case MCTS_PRESET_VANILLA:
            cfg.use_lookahead = DEFAULT_USE_LOOKAHEAD;
            cfg.use_tree_reuse = DEFAULT_TREE_REUSE;
            break;
        case MCTS_PRESET_GRANDMASTER:
            cfg.use_puct = 1;
            cfg.puct_c = PUCT_C;
            cfg.rollout_epsilon = ROLLOUT_EPSILON_NN;
            cfg.use_solver = DEFAULT_USE_SOLVER;
            cfg.use_progressive_bias = 1;
            cfg.bias_constant = DEFAULT_BIAS_CONSTANT;
            apply_weights(&cfg);
            break;
        case MCTS_PRESET_ALPHA_ZERO:
            cfg.use_puct = 1;
            cfg.puct_c = PUCT_C;
            cfg.rollout_epsilon = ROLLOUT_EPSILON_NN;
            cfg.use_solver = DEFAULT_USE_SOLVER;
            break;
        case MCTS_PRESET_TT_ONLY:
            cfg.use_tt = DEFAULT_USE_TT;
            break;
        case MCTS_PRESET_SOLVER_ONLY:
            cfg.use_solver = DEFAULT_USE_SOLVER;
            break;
        case MCTS_PRESET_TUNED_ONLY:
            cfg.use_ucb1_tuned = DEFAULT_USE_UCB1_TUNED;
            break;
        case MCTS_PRESET_FPU_ONLY:
            cfg.use_fpu = DEFAULT_USE_FPU;
            cfg.fpu_value = FPU_VALUE;
            break;
        case MCTS_PRESET_DECAY_ONLY:
            cfg.use_decaying_reward = DEFAULT_USE_DECAY;
            cfg.decay_factor = DEFAULT_DECAY_FACTOR;
            break;
        case MCTS_PRESET_LOOKAHEAD_ONLY:
            cfg.use_lookahead = 1;
            break;
        case MCTS_PRESET_TREE_REUSE_ONLY:
            cfg.use_tree_reuse = 1;
            break;
        case MCTS_PRESET_WEIGHTS_ONLY:
            apply_weights(&cfg);
            cfg.rollout_epsilon = DEFAULT_ROLLOUT_EPSILON;
            break;
        case MCTS_PRESET_SMART_ROLLOUTS:
            apply_weights(&cfg);
            cfg.rollout_epsilon = ROLLOUT_EPSILON_HEURISTIC;
            break;
        case MCTS_PRESET_PROG_BIAS_ONLY:
            apply_weights(&cfg);
            cfg.use_progressive_bias = 1;
            cfg.bias_constant = DEFAULT_BIAS_CONSTANT;
            break;
    }

    return cfg;
}

#endif // MCTS_TYPES_H
