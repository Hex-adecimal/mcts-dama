#ifndef MCTS_H
#define MCTS_H

#include "game.h"
#include <stdlib.h> // size_t

// ================================================================================================
//  MEMORY MANAGEMENT (ARENA)
// ================================================================================================

/**
 * Simple Arena Allocator.
 * Used to allocate MCTS nodes efficiently in a contiguous memory block.
 * This avoids memory fragmentation and the overhead of thousands of malloc/free calls.
 */
typedef struct {
    unsigned char *buffer;  // Pointer to the start of the memory block
    size_t size;            // Total size of the block in bytes
    size_t offset;          // Current allocation offset
} Arena;

/**
 * Initializes the arena with a specific size.
 * @param a Pointer to the Arena.
 * @param total_size Total bytes to allocate for the buffer.
 */
void arena_init(Arena *a, size_t total_size);

/**
 * Allocates memory from the arena.
 * @param a Pointer to the Arena.
 * @param bytes Number of bytes to allocate.
 * @return Pointer to the allocated memory.
 */
void* arena_alloc(Arena *a, size_t bytes);

/**
 * Resets the arena offset to 0.
 * Does NOT free the buffer. Effectively "frees" all objects allocated in the arena instantly.
 * specific usage: Call this before starting a new MCTS search.
 */
void arena_reset(Arena *a);

/**
 * Frees the underlying buffer of the arena.
 * Should be called only when the program terminates.
 */
void arena_free(Arena *a);


// ================================================================================================
//  MCTS CONFIGURATION & STRUCTURES
// ================================================================================================

/**
 * Configuration parameters for the MCTS algorithm.
 * Allows tuning the search behavior without recompiling.
 */
// Forward declaration
typedef struct Node Node;

typedef enum {
    SOLVED_NONE = 0,
    SOLVED_WIN = 1,   // Current player wins from this node
    SOLVED_LOSS = -1, // Current player loses from this node
    SOLVED_DRAW = 2   // Proven draw
} SolverStatus;

typedef struct {
    double ucb1_c;           // Exploration constant (e.g., 1.414). Higher = more exploration.
    double rollout_epsilon;  // Probability of choosing a random move during rollout (Epsilon-Greedy).
    double draw_score;       // Score assigned for a draw (e.g., 0.5).
    
    int expansion_threshold; // Minimum visits before a node is expanded.

    int verbose;             // Print search statistics (1=enabled, 0=quiet mode for tournaments).

    int use_lookahead;       // Enable 1-ply lookahead in simulation (0=disabled, 1=enabled).


    int use_tree_reuse;      // Enable tree reuse between moves (0=disabled, 1=enabled).
    
    int use_ucb1_tuned;      // Enable UCB1-Tuned selection strategy (0=disabled, 1=enabled).
    
    int use_tt;              // Enable Transposition Table (0=disabled, 1=enabled).
    
    int use_solver;          // Enable MCTS-Solver (Win/Loss propagation) (0=disabled, 1=enabled).
    
    int use_progressive_bias;// Enable Progressive Bias (0=disabled, 1=enabled).
    double bias_constant;    // Constant weight for Progressive Bias (e.g. 10.0).
} MCTSConfig;

/**
 * Transposition Table to detect identical states reached via different paths.
 * Maps Zobrist Hash -> Node Pointer.
 * Uses a simple array of buckets with linear probing or just simple replacement.
 */
typedef struct {
    Node **buckets;
    size_t size;           // Number of buckets (power of 2)
    size_t mask;           // size - 1
    size_t count;          // Number of entries
    size_t collisions;     // Statistic
} TranspositionTable;

/**
 * Statistics tracker for MCTS performance analysis.
 */
typedef struct {
    int total_moves;         // Total number of moves made
    long total_iterations;   // Total simulations across all moves
    long total_depth;        // Total tree depth across all moves
    double total_time;       // Total time spent in MCTS search (seconds)
    size_t total_memory;     // Total memory used by MCTS trees (bytes)
} MCTSStats;

/**
 * Represents a node in the Monte Carlo Tree Search.
 */
struct Node {
    GameState state;            // The game state at this node
    Move move_from_parent;      // The move that led to this state
    int player_who_just_moved;  // The player who made the move (WHITE/BLACK)
    
    struct Node *parent;        // Pointer to parent node (NULL for root)
    struct Node **children;     // Dynamic array of pointers to children nodes
    int num_children;           // Number of children
    
    MoveList untried_moves;     // Moves that have not yet been expanded
    int is_terminal;            // 1 if the game is over at this node, 0 otherwise
    
    int visits;                 // Number of times this node has been visited
    double score;               // Accumulated score (Win=1, Draw=0.5, Loss=0)
    double sum_sq_score;        // Sum of squared scores (for Variance calculation in UCB1-Tuned)
    double heuristic_score;     // Static evaluation score for Progressive Bias
    
    int8_t status;              // SolverStatus: NONE, WIN, LOSS, DRAW
};


// ================================================================================================
//  MCTS API
// ================================================================================================

/**
 * Creates the root node for the MCTS search.
 * @param state The current game state.
 * @param arena Pointer to the arena allocator.
 * @return Pointer to the newly created root node.
 */
Node* mcts_create_root(GameState state, Arena *arena);

/**
 * Executes the MCTS search algorithm.
 * @param root Pointer to the root node.
 * @param arena Pointer to the arena allocator (used for expanding new nodes).
 * @param time_limit_seconds Maximum time allowed for the search.
 * @param config Configuration parameters for the search.
 * @param stats Optional pointer to MCTSStats to update with search statistics.
 * @param out_new_root Optional pointer to store the chosen child (for tree reuse).
 * @return The best move found.
 */
Move mcts_search(Node *root, Arena *arena, double time_limit_seconds, MCTSConfig config,
                 MCTSStats *stats, Node **out_new_root);

/**
 * Helper function to print a human-readable description of a move.
 * @param m The move to print.
 */
void print_move_description(Move m);

/**
 * Finds child node matching the given move (for tree reuse).
 * @param parent Parent node to search in.
 * @param move Move to find.
 * @return Child node if found, NULL otherwise.
 */
Node* find_child_by_move(Node *parent, const Move *move);

/**
 * Calculates the maximum depth of the MCTS tree.
 * @param node Pointer to the root node.
 * @return Maximum depth of the tree.
 */
int get_tree_depth(Node *node);

#endif // MCTS_H