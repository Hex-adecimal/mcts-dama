#ifndef GAME_H
#define GAME_H

#include <stdint.h>

// =============================================================================
// TYPES & STRUCTURES
// =============================================================================
// Color, Square, Bitboard, Move, MoveList, GameState

typedef enum { WHITE = 0, BLACK = 1 } Color;

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

typedef uint64_t Bitboard;

typedef struct {
    uint8_t path[12];             // Sequence of squares visited (max 12 jumps)
    uint8_t captured_squares[12]; // Sequence of captured pieces
    uint8_t length;               // Number of jumps (0 for simple move)
    
    // Priority Metrics (Italian Checkers rules)
    uint8_t captured_ladies_count;
    uint8_t is_lady_move;
    uint8_t first_captured_is_lady;
} Move;

#define MAX_MOVES 64
typedef struct {
    Move moves[MAX_MOVES];
    int count;
} MoveList;

typedef struct {
    Bitboard white_pieces;
    Bitboard white_ladies;
    Bitboard black_pieces;
    Bitboard black_ladies;
    Color current_player;
    int moves_without_captures;
    uint64_t hash;
} GameState;

// =============================================================================
// CONSTANTS
// =============================================================================
// MAX_MOVES_WITHOUT_CAPTURES, PROM_RANKS, Board Edge Masks

#define MAX_MOVES_WITHOUT_CAPTURES 40

// Promotion Ranks: WHITE promotes on rank 8, BLACK on rank 1
static const uint64_t PROM_RANKS[2] = {
    0xFF00000000000000ULL,  // WHITE
    0x00000000000000FFULL   // BLACK
};

// Board Edge Masks
static const uint64_t NOT_FILE_A  = 0xfefefefefefefefeULL;
static const uint64_t NOT_FILE_H  = 0x7f7f7f7f7f7f7f7fULL;
static const uint64_t NOT_FILE_GH = 0x3f3f3f3f3f3f3f3fULL;
static const uint64_t NOT_FILE_AB = 0xfcfcfcfcfcfcfcfcULL;

// =============================================================================
// INLINE HELPERS
// =============================================================================
// set_bit, check_bit, get_all_*, shift_bitboard

static inline void set_bit(Bitboard *bb, int sq) { *bb |= (1ULL << sq); }
static inline int check_bit(Bitboard bb, int sq) { return (bb & (1ULL << sq)) ? 1 : 0; }

static inline Bitboard get_all_whites(const GameState *s) {
    return s->white_ladies | s->white_pieces;
}

static inline Bitboard get_all_black(const GameState *s) {
    return s->black_ladies | s->black_pieces;
}

static inline Bitboard get_all_occupied(const GameState *s) {
    return get_all_whites(s) | get_all_black(s);
}

static inline Bitboard get_empty_squares(const GameState *s) {
    return ~get_all_occupied(s);
}

static inline Bitboard shift_bitboard(Bitboard b, int offset) {
    return (offset > 0) ? (b << offset) : (b >> (-offset));
}

// =============================================================================
// ZOBRIST HASHING
// =============================================================================
// extern declarations + zobrist_init()

extern uint64_t zobrist_keys[2][2][64];  // [Color][PieceType][Square]
extern uint64_t zobrist_black_move;

void zobrist_init(void);
void init_move_tables(void);

// =============================================================================
// GAME API
// =============================================================================
// init_game, apply_move, generate_moves, is_square_threatened

void init_game(GameState *state);
void apply_move(GameState *state, const Move *move);
void generate_moves(const GameState *s, MoveList *list);
int is_square_threatened(const GameState *state, int square);

#endif // GAME_H