#ifndef GAME_H
#define GAME_H

#include <stdint.h> // Required for uint64_t

/*
- Stack Usage: La funzione ricorsiva find_captures copia array (memcpy di path e captured) ad ogni livello di profondità.
    Poiché la profondità massima è bassa (12), non è un problema di stack overflow, ma copiare memoria è meno efficiente che passare puntatori o usare uno stack globale per la mossa corrente. 
    Tuttavia, per la complessità della Dama, è accettabile.

Branching: La logica è piena di if per gestire le differenze Bianco/Nero e Pedina/Dama. 
In motori super-ottimizzati si cerca di evitare il branching per non svuotare la pipeline della CPU, usando template C++ o logiche branchless, ma questo codice C è molto leggibile e sufficientemente veloce.
*/

// --- BASIC DEFINITIONS ---

// Represents the colors in the game
typedef enum {
    WHITE = 0, 
    BLACK = 1
} Color;

// Represents the squares on the board
typedef enum {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8
} Square;

// Definition for Moves (supports chains)
typedef struct {
    uint8_t path[12];           // Sequence of squares visited (max 12 jumps)
    uint8_t captured_squares[12]; // Sequence of captured pieces
    uint8_t length;             // Number of jumps (0 for simple move)
    
    // Priority Metrics
    uint8_t captured_ladies_count;
    uint8_t is_lady_move;       // 1 if the moving piece is a Lady
    uint8_t first_captured_is_lady; // 1 if the first captured piece is a Lady
} Move;

// Static list to avoid malloc
#define MAX_MOVES 64 
typedef struct {
    Move moves[MAX_MOVES];
    int count;
} MoveList;

// An unsigned 64-bit integer where each bit represents a square on the board
typedef uint64_t Bitboard;

// Represents the complete state of the game - Uses separate bitboards for each piece type to optimize bitwise operations.
typedef struct {
    Bitboard white_pieces;
    Bitboard white_ladies;
    Bitboard black_pieces;
    Bitboard black_ladies;

    Color current_player; // The side to move next

    int moves_without_captures; // Counter for the 40-move draw rule
    
    uint64_t hash; // Zobrist Hash of the state
} GameState;

void zobrist_init(); // Initializes random keys

// --- GLOBAL CONSTANTS ---

#define MAX_MOVES_WITHOUT_CAPTURES 40
#define NO_CAPTURE_IDX -1

// Promotion Ranks
// Index 0 (WHITE): Rank 8 (top) -> 0xFF...00
// Index 1 (BLACK): Rank 1 (bottom) -> 0x00...FF
static const uint64_t PROM_RANKS[2] = {
    0xFF00000000000000ULL, 
    0x00000000000000FFULL
};

// Board Edge Masks
static const uint64_t NOT_FILE_A = 0xfefefefefefefefeULL;
static const uint64_t NOT_FILE_H = 0x7f7f7f7f7f7f7f7fULL;
static const uint64_t NOT_FILE_GH = 0x3f3f3f3f3f3f3f3fULL;
static const uint64_t NOT_FILE_AB = 0xfcfcfcfcfcfcfcfcULL;

// --- INLINE FUNCTIONS (Helpers) ---

static inline void set_bit(Bitboard *bb, int sq) { *bb |= (1ULL << sq); }
static inline void clear_bit(Bitboard *bb, int sq) { *bb &= ~(1ULL << sq); }
static inline int check_bit(Bitboard bb, int sq) { return (bb & (1ULL << sq)) ? 1 : 0; }

// Combines white pieces and ladies into a single bitboard
static inline Bitboard get_all_whites(const GameState *state) {
    return state->white_ladies | state->white_pieces;
}

// Combines black pieces and ladies into a single bitboard
static inline Bitboard get_all_black(const GameState *state) {
    return state->black_ladies | state->black_pieces;
}

// Returns a bitboard of all occupied squares (white and black)
static inline Bitboard get_all_occupied(const GameState *state) {
    return get_all_whites(state) | get_all_black(state);
}

// Essential for generating valid move destinations
static inline Bitboard get_empty_squares(const GameState *state) {
    return ~(get_all_occupied(state));
}

// Universal shift (handles negative offsets)
static inline Bitboard shift_bitboard(Bitboard b, int offset) {
    if (offset > 0) return b << offset;
    return b >> (-offset);
}

// Helper to add moves (Updated for new struct)
static inline void add_move(MoveList *list, int from, int to, int cap) {
    if (list->count < MAX_MOVES) {
        Move *m = &list->moves[list->count++];
        m->path[0] = from;
        m->path[1] = to;
        m->length = 0; // Default to simple
        if (cap != NO_CAPTURE_IDX) {
             // This helper is for simple moves or single captures if used by legacy code
             // But for chains we use manual assignment.
             // Let's keep it compatible for simple usage if needed.
             m->captured_squares[0] = cap;
             m->length = 1; // It's a capture
        }
        m->captured_ladies_count = 0;
        m->is_lady_move = 0; // Unknown here
        m->first_captured_is_lady = 0;
    }
}

// --- FUNCTION PROTOTYPES ---

/**
 * Initializes the game state.
 * Sets up the board with pieces in their starting positions and sets the turn to White.
 * @param state Pointer to the GameState structure to initialize.
 */
/**
 * Initializes the game state to the standard starting position.
 * @param state Pointer to the GameState to initialize.
 */
void init_game(GameState *state);

/**
 * Prints the current board state to the console.
 * Displays the board with coordinates, piece positions, and current turn info.
 * @param state Pointer to the GameState to display.
 */
void print_board(const GameState *state);

/**
 * Prints the list of generated moves to the console.
 * Useful for debugging and verifying move generation logic.
 * @param list Pointer to the MoveList containing moves to print.
 */
void print_move_list(MoveList *list);

/**
 * Prints the algebraic coordinates (e.g., "A1") of a square index.
 * @param square_idx The 0-63 index of the square.
 */
void print_coords(int square_idx);

// Core Engine

/**
 * Applies a generic move (simple or chain capture) to the game state.
 * Unified logic for both simple moves and captures.
 * @param state Pointer to the current GameState.
 * @param move Pointer to the Move structure to apply.
 */
void apply_move(GameState *state, const Move *move);

// Move Generation

/**
 * Generates all legal moves for the current player.
 * Enforces Italian Checkers rules: mandatory captures and priority rules.
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void generate_moves(const GameState *s, MoveList *list);

#endif // GAME_H