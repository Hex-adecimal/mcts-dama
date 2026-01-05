/**
 * @file game.h
 * @brief Core Game Types and State for Italian Checkers.
 */

#ifndef GAME_H
#define GAME_H

#include "dama/common/params.h"
#include <stdint.h>

// --- Type Definitions ---
typedef uint64_t Bitboard;
typedef enum { WHITE = 0, BLACK = 1 } Color;
typedef enum { PAWN = 0, LADY = 1 } Piece;
typedef enum { DIR_NE = 0, DIR_NW = 1, DIR_SE = 2, DIR_SW = 3 } Direction;

// --- Named Constants ---
#define NUM_COLORS      2
#define NUM_PIECE_TYPES 2
#define NUM_SQUARES     64
#define NUM_DIRECTIONS  4
#define NUM_PAWN_DIRS   2
#define MAX_MOVES       64
#define MAX_CHAIN_LENGTH 12

// Pawn capture direction ranges (for find_captures loop)
// WHITE pawns capture NE (0) and NW (1)
// BLACK pawns capture SE (2) and SW (3)
#define WHITE_DIR_START DIR_NE   // 0
#define WHITE_DIR_END   DIR_SE   // 2 (exclusive)
#define BLACK_DIR_START DIR_SE   // 2
#define BLACK_DIR_END   NUM_DIRECTIONS  // 4 (exclusive)

// Direction offsets on 8x8 board (sq index changes)
#define OFFSET_NE  (+9)   // Up-right
#define OFFSET_NW  (+7)   // Up-left
#define OFFSET_SE  (-7)   // Down-right
#define OFFSET_SW  (-9)   // Down-left

// Jump offsets (2 squares in direction)
#define JUMP_NE    (+18)
#define JUMP_NW    (+14)
#define JUMP_SE    (-14)
#define JUMP_SW    (-18)

#define INITIAL_WHITE_PAWNS 0x0000000000AA55AAULL
#define INITIAL_BLACK_PAWNS 0x55AA550000000000ULL
#define ZOBRIST_SEED        0x987654321ULL

// --- Helper Macros ---
#define BIT(sq)           (1ULL << (sq))
#define SET_BIT(bb, sq)   ((bb) |= BIT(sq))
#define CLEAR_BIT(bb, sq) ((bb) &= ~BIT(sq))
#define TEST_BIT(bb, sq)  ((bb) & BIT(sq))
#define POP_LSB(bb)       ((bb) &= (bb) - 1)
#define ROW(sq)           ((sq) / 8)
#define COL(sq)           ((sq) % 8)
#define SQUARE(row, col)  ((row) * 8 + (col))

// --- Move Structures ---
typedef struct {
    uint8_t path[MAX_CHAIN_LENGTH];
    uint8_t captured_squares[MAX_CHAIN_LENGTH];
    uint8_t length;
    uint8_t captured_ladies_count;
    uint8_t is_lady_move;
    uint8_t first_captured_is_lady;
} Move;

typedef struct {
    Move moves[MAX_MOVES];
    uint8_t count;
} MoveList;

// --- Game State ---
typedef struct {
    Bitboard piece[NUM_COLORS][NUM_PIECE_TYPES];
    Color current_player;
    uint8_t moves_without_captures;
    uint64_t hash;
} GameState;

// --- Constant Masks ---
static const Bitboard PROM_RANKS[NUM_COLORS] = {
    0xFF00000000000000ULL,  // WHITE
    0x00000000000000FFULL   // BLACK
};

static const Bitboard NOT_FILE_A  = 0xfefefefefefefefeULL;
static const Bitboard NOT_FILE_H  = 0x7f7f7f7f7f7f7f7fULL;
static const Bitboard NOT_FILE_GH = 0x3f3f3f3f3f3f3f3fULL;
static const Bitboard NOT_FILE_AB = 0xfcfcfcfcfcfcfcfcULL;

// --- Inline Helpers ---
static inline Bitboard get_pieces(const GameState *s, const Color color) {
    return s->piece[color][PAWN] | s->piece[color][LADY];
}

static inline Bitboard get_all_occupied(const GameState *s) {
    return get_pieces(s, WHITE) | get_pieces(s, BLACK);
}

static inline Bitboard get_empty_squares(const GameState *s) {
    return ~get_all_occupied(s);
}

static inline int check_bit(const Bitboard bb, const int sq) {
    return TEST_BIT(bb, sq) ? 1 : 0;
}

// --- Public API ---

/**
 * @brief Initialize the game state to the starting position.
 *
 * @param state Pointer to the GameState structure to initialize.
 */
void init_game(GameState *state);

/**
 * @brief Apply a move to the game state.
 *
 * @param state Pointer to the current GameState.
 * @param move Pointer to the Move to apply.
 */
void apply_move(GameState *state, const Move *move);

#endif /* GAME_H */
