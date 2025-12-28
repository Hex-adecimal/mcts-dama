/**
 * =============================================================================
 * game.h - Core Game Data Structures for Italian Checkers
 * =============================================================================
 * 
 * This header defines all fundamental types and structures for the game:
 *   - Color, Square, Bitboard type definitions
 *   - Move and MoveList structures
 *   - GameState structure
 *   - Board constants and masks
 *   - Inline helper functions
 *   - Zobrist hashing declarations
 *   - Game API function prototypes
 * 
 * =============================================================================
 */

#ifndef GAME_H
#define GAME_H

#include <stdint.h>

/* =============================================================================
 * TYPE DEFINITIONS
 * ============================================================================= */

/** 
 * Player color enumeration.
 * WHITE always moves first and advances upward (increasing row).
 */
typedef enum { WHITE = 0, BLACK = 1 } Color;

/**
 * Square enumeration for readable board positions.
 * 
 * Layout (A1=0, H8=63):
 *   A8 B8 C8 D8 E8 F8 G8 H8   (56-63)
 *   A7 B7 C7 D7 E7 F7 G7 H7   (48-55)
 *   ...
 *   A1 B1 C1 D1 E1 F1 G1 H1   (0-7)
 */
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

/**
 * Bitboard type for efficient board representation.
 * Each bit corresponds to a square (bit 0 = A1, bit 63 = H8).
 */
typedef uint64_t Bitboard;

/* =============================================================================
 * MOVE STRUCTURE
 * ============================================================================= */

/**
 * Represents a single move (simple or capture chain).
 * 
 * Simple Move (length = 0):
 *   - path[0]: source square
 *   - path[1]: destination square
 * 
 * Capture Chain (length > 0):
 *   - path[0..length]: sequence of squares visited
 *   - captured_squares[0..length-1]: squares of captured pieces
 *   - length: number of captures made
 * 
 * Priority metrics are used for Italian Checkers filtering rules.
 */
typedef struct {
    uint8_t path[12];              /**< Squares visited (max 12 for longest chain) */
    uint8_t captured_squares[12];  /**< Captured piece locations */
    uint8_t length;                /**< Number of captures (0 = simple move) */
    
    /* Italian Checkers priority metrics */
    uint8_t captured_ladies_count;   /**< How many Ladies were captured */
    uint8_t is_lady_move;            /**< 1 if moving piece is a Lady */
    uint8_t first_captured_is_lady;  /**< 1 if first capture was a Lady */
} Move;

/* =============================================================================
 * MOVE LIST
 * ============================================================================= */

/** Maximum number of legal moves in any position */
#define MAX_MOVES 64

/**
 * Container for generated legal moves.
 */
typedef struct {
    Move moves[MAX_MOVES];  /**< Array of legal moves */
    int count;              /**< Number of moves in array */
} MoveList;

/* =============================================================================
 * GAME STATE
 * ============================================================================= */

/**
 * Complete representation of a game position.
 * 
 * Uses 4 bitboards for efficient piece tracking:
 *   - white_pieces: White pawns (not promoted)
 *   - white_ladies: White promoted pieces
 *   - black_pieces: Black pawns (not promoted)
 *   - black_ladies: Black promoted pieces
 * 
 * The hash field stores the Zobrist hash for transposition tables.
 */
typedef struct {
    Bitboard white_pieces;        /**< White pawn positions */
    Bitboard white_ladies;        /**< White Lady positions */
    Bitboard black_pieces;        /**< Black pawn positions */
    Bitboard black_ladies;        /**< Black Lady positions */
    Color current_player;         /**< Whose turn it is */
    int moves_without_captures;   /**< Counter for 40-move draw rule */
    uint64_t hash;                /**< Zobrist hash of position */
} GameState;

/* =============================================================================
 * GAME CONSTANTS
 * ============================================================================= */

/** 
 * 40-move draw rule threshold.
 * If this many moves pass without a capture (and Ladies exist), game is drawn.
 */
#define MAX_MOVES_WITHOUT_CAPTURES 40

/**
 * Promotion rank masks.
 * A pawn reaching these ranks becomes a Lady.
 *   - WHITE promotes on rank 8 (row 7)
 *   - BLACK promotes on rank 1 (row 0)
 */
static const uint64_t PROM_RANKS[2] = {
    0xFF00000000000000ULL,  /* WHITE: rank 8 */
    0x00000000000000FFULL   /* BLACK: rank 1 */
};

/* =============================================================================
 * BOARD EDGE MASKS
 * =============================================================================
 * 
 * Used to prevent piece movement from wrapping around board edges.
 * NOT_FILE_X means "all squares except file X".
 */

static const uint64_t NOT_FILE_A  = 0xfefefefefefefefeULL;  /**< Exclude file A */
static const uint64_t NOT_FILE_H  = 0x7f7f7f7f7f7f7f7fULL;  /**< Exclude file H */
static const uint64_t NOT_FILE_GH = 0x3f3f3f3f3f3f3f3fULL;  /**< Exclude files G,H */
static const uint64_t NOT_FILE_AB = 0xfcfcfcfcfcfcfcfcULL;  /**< Exclude files A,B */

/* =============================================================================
 * INLINE HELPER FUNCTIONS
 * ============================================================================= */

/**
 * Set a bit in a bitboard.
 * @param bb  Pointer to bitboard
 * @param sq  Square index (0-63)
 */
static inline void set_bit(Bitboard *bb, int sq) { 
    *bb |= (1ULL << sq); 
}

/**
 * Check if a bit is set in a bitboard.
 * @param bb  Bitboard to check
 * @param sq  Square index (0-63)
 * @return    1 if bit is set, 0 otherwise
 */
static inline int check_bit(Bitboard bb, int sq) { 
    return (bb & (1ULL << sq)) ? 1 : 0; 
}

/** Get bitboard of all White pieces (pawns + ladies) */
static inline Bitboard get_all_whites(const GameState *s) {
    return s->white_ladies | s->white_pieces;
}

/** Get bitboard of all Black pieces (pawns + ladies) */
static inline Bitboard get_all_black(const GameState *s) {
    return s->black_ladies | s->black_pieces;
}

/** Get bitboard of all occupied squares */
static inline Bitboard get_all_occupied(const GameState *s) {
    return get_all_whites(s) | get_all_black(s);
}

/** Get bitboard of all empty squares */
static inline Bitboard get_empty_squares(const GameState *s) {
    return ~get_all_occupied(s);
}

/**
 * Shift a bitboard by a signed offset.
 * @param b       Bitboard to shift
 * @param offset  Positive = left shift, Negative = right shift
 * @return        Shifted bitboard
 */
static inline Bitboard shift_bitboard(Bitboard b, int offset) {
    return (offset > 0) ? (b << offset) : (b >> (-offset));
}

/* =============================================================================
 * ZOBRIST HASHING
 * ============================================================================= */

/**
 * Zobrist hash keys: [color][piece_type][square]
 *   - color: WHITE(0), BLACK(1)
 *   - piece_type: PAWN(0), LADY(1)
 *   - square: 0-63
 */
extern uint64_t zobrist_keys[2][2][64];

/** Zobrist key XORed when it's Black's turn */
extern uint64_t zobrist_black_move;

/** Initialize Zobrist hash tables. Call once at startup. */
void zobrist_init(void);

/* =============================================================================
 * GAME API
 * ============================================================================= */

/**
 * Initialize a game to the standard starting position.
 * @param state  GameState to initialize
 */
void init_game(GameState *state);

/**
 * Apply a move to the game state.
 * @param state  Game state to modify
 * @param move   Move to apply (from move generation)
 */
void apply_move(GameState *state, const Move *move);

/* =============================================================================
 * DEBUG & PRINTING
 * ============================================================================= */

/** Print the board state in ASCII format */
void print_board(const GameState *state);

/** Print a square index as algebraic notation (e.g., "A1") */
void print_coords(int square_idx);

/** Print all moves in a move list with details */
void print_move_list(MoveList *list);

/** Print a single move in compact notation */
void print_move_description(Move m);

#endif /* GAME_H */
