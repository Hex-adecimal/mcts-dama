/**
 * game.c - Core Game State Management for Italian Checkers
 */

#include "dama/engine/game.h"
#include <stdio.h>

// --- Zobrist Hashing ---
uint64_t zobrist_keys[NUM_COLORS][NUM_PIECE_TYPES][NUM_SQUARES];
uint64_t zobrist_black_move;

static uint64_t rand64(void) {
    static uint64_t seed = ZOBRIST_SEED;
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
}

void zobrist_init(void) {
    for (Color c = WHITE; c <= BLACK; c++) {
        for (Piece t = PAWN; t <= LADY; t++) {
            for (int sq = 0; sq < NUM_SQUARES; sq++) {
                zobrist_keys[c][t][sq] = rand64();
            }
        }
    }
    zobrist_black_move = rand64();
}

static uint64_t compute_full_hash(const GameState *s) {
    uint64_t h = 0;
    for (Color c = WHITE; c <= BLACK; c++) {
        for (Piece t = PAWN; t <= LADY; t++) {
            Bitboard bb = s->piece[c][t];
            while (bb) {
                const int sq = __builtin_ctzll(bb);
                h ^= zobrist_keys[c][t][sq];
                POP_LSB(bb);
            }
        }
    }
    if (s->current_player == BLACK) h ^= zobrist_black_move;
    return h;
}

// --- Game Initialization ---
void init_game(GameState *state) {
    state->piece[WHITE][PAWN] = INITIAL_WHITE_PAWNS;
    state->piece[BLACK][PAWN] = INITIAL_BLACK_PAWNS;
    state->piece[WHITE][LADY] = 0;
    state->piece[BLACK][LADY] = 0;
    state->current_player = WHITE;
    state->moves_without_captures = 0;
    state->hash = compute_full_hash(state);
}

// --- Move Execution ---
static void perform_movement(GameState *s, const int from, const int to) {
    const Color us = s->current_player;
    const Bitboard from_to_mask = BIT(from) | BIT(to);
    
    const Piece type = TEST_BIT(s->piece[us][LADY], from) ? LADY : PAWN;
    s->piece[us][type] ^= from_to_mask;
    
    if (type == PAWN && TEST_BIT(PROM_RANKS[us], to)) {
        CLEAR_BIT(s->piece[us][PAWN], to);
        SET_BIT(s->piece[us][LADY], to);
    }
}

void apply_move(GameState *state, const Move *move) {
    const Color us = state->current_player;
    const Color them = us ^ 1;
    const int from = move->path[0];
    const int to = (move->length == 0) ? move->path[1] : move->path[move->length];

    state->hash ^= zobrist_keys[us][move->is_lady_move][from];
    perform_movement(state, from, to);
    
    const Piece piece_now = TEST_BIT(state->piece[us][LADY], to) ? LADY : PAWN;
    state->hash ^= zobrist_keys[us][piece_now][to];

    if (move->length > 0) {
        for (int i = 0; i < move->length; i++) {
            const int cap_sq = move->captured_squares[i];
            const Piece cap_type = TEST_BIT(state->piece[them][LADY], cap_sq) ? LADY : PAWN;
            state->hash ^= zobrist_keys[them][cap_type][cap_sq];
            CLEAR_BIT(state->piece[them][cap_type], cap_sq);
        }
        state->moves_without_captures = 0;
    } else {
        if (state->piece[WHITE][LADY] || state->piece[BLACK][LADY])
            state->moves_without_captures++;
        else
            state->moves_without_captures = 0;
    }

    state->current_player = them;
    state->hash ^= zobrist_black_move;
}

// --- Debug Utilities ---
void print_board(const GameState *s) {
    printf("\n   A B C D E F G H\n  +---------------+\n");
    for (int r = 7; r >= 0; r--) {
        printf("%d |", r + 1);
        for (int f = 0; f < 8; f++) {
            const int sq = SQUARE(r, f);
            char c = '.';
            if (TEST_BIT(s->piece[WHITE][PAWN], sq))      c = 'w';
            else if (TEST_BIT(s->piece[WHITE][LADY], sq)) c = 'W';
            else if (TEST_BIT(s->piece[BLACK][PAWN], sq)) c = 'b';
            else if (TEST_BIT(s->piece[BLACK][LADY], sq)) c = 'B';
            printf("%c|", c);
        }
        printf(" %d\n", r + 1);
    }
    printf("  +---------------+\n   A B C D E F G H\n\n");
    printf("Turn: %s | Hash: %llx | Draw: %d\n", 
           (s->current_player == WHITE) ? "WHITE" : "BLACK", 
           (unsigned long long)s->hash, s->moves_without_captures);
}

void print_coords(const int sq) {
    printf("%c%d", COL(sq) + 'A', ROW(sq) + 1);
}

void print_move_description(const Move m) {
    print_coords(m.path[0]);
    if (m.length == 0) {
        printf("-");
        print_coords(m.path[1]);
    } else {
        for (int i = 0; i < m.length; i++) {
            printf("x");
            print_coords(m.path[i + 1]);
        }
    }
}
