/**
 * game.c - Core Game State Management
 * 
 * Contains: Zobrist hashing, game initialization, move application.
 */

#include "game.h"
#include <string.h>
#include <stdio.h>

// =============================================================================
// ZOBRIST HASHING
// =============================================================================

// [Color][PieceType][Square]
// Color: 0=White, 1=Black
// PieceType: 0=Pawn, 1=Lady
uint64_t zobrist_keys[2][2][64];
uint64_t zobrist_black_move;

// Simple PRNG for reproducible runs
static uint64_t rand64(void) {
    static uint64_t seed = 0x987654321ULL;
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
}

void zobrist_init(void) {
    for (int c = 0; c < 2; c++) {
        for (int pt = 0; pt < 2; pt++) {
            for (int sq = 0; sq < 64; sq++) {
                zobrist_keys[c][pt][sq] = rand64();
            }
        }
    }
    zobrist_black_move = rand64();
}

// Helper: hash a bitboard for Zobrist
static uint64_t hash_bitboard(Bitboard bb, int color, int piece_type) {
    uint64_t h = 0;
    while (bb) {
        int sq = __builtin_ctzll(bb);
        h ^= zobrist_keys[color][piece_type][sq];
        bb &= (bb - 1);
    }
    return h;
}

static uint64_t compute_full_hash(const GameState *state) {
    uint64_t hash = hash_bitboard(state->white_pieces, WHITE, 0)
                  ^ hash_bitboard(state->white_ladies, WHITE, 1)
                  ^ hash_bitboard(state->black_pieces, BLACK, 0)
                  ^ hash_bitboard(state->black_ladies, BLACK, 1);
    
    // Turn encoding: XOR with zobrist_black_move only when it's Black's turn.
    if (state->current_player == BLACK) {
        hash ^= zobrist_black_move;
    }
    return hash;
}

// =============================================================================
// INITIALIZATION
// =============================================================================

/**
 * Initializes the game state.
 * Sets up the board with pieces in their starting positions and sets the turn to White.
 * @param state Pointer to the GameState structure to initialize.
 */
void init_game(GameState *state) {
    // Clear the board state
    state->white_ladies = 0;
    state->white_pieces = 0;
    state->black_ladies = 0;
    state->black_pieces = 0;

    state->current_player = WHITE;
    state->moves_without_captures = 0;

    // Initialize White pieces:
    // Row 0: 0xAA (10101010)
    // Row 1: 0x55 (01010101) shifted by 8
    // Row 2: 0xAA (10101010) shifted by 16
    state->white_pieces = 0x0000000000AA55AAULL;

    // Initialize Black pieces:
    // Row 5: 0x55 shifted by 40
    // Row 6: 0xAA shifted by 48
    // Row 7: 0x55 shifted by 56
    state->black_pieces = 0x55AA550000000000ULL;
    
    state->hash = compute_full_hash(state);
}

// =============================================================================
// CORE ENGINE (Apply Move)
// =============================================================================

/**
 * Updates bitboards and switches the turn.
 * @param state Pointer to the current GameState.
 * @param from Source square index.
 * @param to Destination square index.
 */
static void perform_movement(GameState *s, int from, int to, int us) {
    Bitboard *own_pieces = (us == WHITE) ? &s->white_pieces : &s->black_pieces;
    Bitboard *own_ladies = (us == WHITE) ? &s->white_ladies : &s->black_ladies;
    
    Bitboard move_mask = (1ULL << from) | (1ULL << to);
    Bitboard is_piece = (*own_pieces & (1ULL << from));

    if (is_piece) *own_pieces ^= move_mask;
    else          *own_ladies ^= move_mask;

    // Promotion
    if (is_piece && ((1ULL << to) & PROM_RANKS[us])) {
        *own_pieces &= ~(1ULL << to);
        *own_ladies |= (1ULL << to);
    }
}

/**
 * Applies a generic move (simple or chain capture) to the game state.
 * Unified logic for both simple moves and captures.
 * @param state Pointer to the current GameState.
 * @param move Pointer to the Move structure to apply.
 */
void apply_move(GameState *state, const Move *move) {
    int us = state->current_player;
    int from = move->path[0];
    
    // Determine destination
    // If simple move (length 0), dest is path[1].
    // If capture (length > 0), dest is path[length].
    int to = (move->length == 0) ? move->path[1] : move->path[move->length];

    // --- ZOBRIST: Remove moving piece from source ---
    int is_lady = move->is_lady_move; // Trusted from move generation
    
    state->hash ^= zobrist_keys[us][is_lady][from];

    // Move the piece (and handle promotion)
    perform_movement(state, from, to, us);

    // --- ZOBRIST: Add piece at destination ---
    // Check if it promoted.
    int now_lady = (us == WHITE) ? check_bit(state->white_ladies, to) : check_bit(state->black_ladies, to);
    state->hash ^= zobrist_keys[us][now_lady][to];

    // Remove captured pieces (if any)
    if (move->length > 0) {
        int them = us ^ 1;
        Bitboard *enemy_pieces = (them == BLACK) ? &state->black_pieces : &state->white_pieces;
        Bitboard *enemy_ladies = (them == BLACK) ? &state->black_ladies : &state->white_ladies;
        
        for (int i = 0; i < move->length; i++) {
            int cap_sq = move->captured_squares[i];
            
            // Check what is currently there
             int is_l = check_bit(*enemy_ladies, cap_sq);
             
             if (is_l) state->hash ^= zobrist_keys[them][1][cap_sq];
             else      state->hash ^= zobrist_keys[them][0][cap_sq];

            Bitboard remove_mask = ~(1ULL << cap_sq);
            *enemy_pieces &= remove_mask;
            *enemy_ladies &= remove_mask;
        }
        state->moves_without_captures = 0;
    } else {
        state->moves_without_captures++;
    }

    // Switch Turn
    state->current_player = (Color)(us ^ 1);
    state->hash ^= zobrist_black_move; // Toggle turn hash
}

// =============================================================================
// PRINTERS & DEBUG
// =============================================================================

void print_coords(int square_idx) {
    int rank = (square_idx / 8) + 1;
    char file = (square_idx % 8) + 'A';
    printf("%c%d", file, rank);
}

void print_board(const GameState *state) {
    printf("\n   A B C D E F G H\n");
    printf("  +---------------+\n");

    for (int rank = 7; rank >= 0; rank--) {
        printf("%d |", rank + 1);
        
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            char c = '.';

            if (check_bit(state->white_pieces, sq)) c = 'w';
            else if (check_bit(state->black_pieces, sq)) c = 'b';
            else if (check_bit(state->white_ladies, sq)) c = 'W';
            else if (check_bit(state->black_ladies, sq)) c = 'B';
            
            printf("%c|", c);
        }
        printf(" %d\n", rank + 1);
    }
    printf("  +---------------+\n");
    printf("   A B C D E F G H\n\n");
    
    printf("Turn: %s\n", (state->current_player == WHITE) ? "WHITE" : "BLACK");
    printf("White pieces bitboard: %llu\n", state->white_pieces);
}

void print_move_list(MoveList *list) {
    printf("------------------------------------------------\n");
    printf("Found %d possible moves:\n", list->count);
    
    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        printf("%d) ", i + 1);
        
        if (m.length == 0) {
            print_coords(m.path[0]);
            printf(" -> ");
            print_coords(m.path[1]);
        } else {
            print_coords(m.path[0]);
            for (int j = 0; j < m.length; j++) {
                printf(" x ");
                print_coords(m.captured_squares[j]);
                printf(" -> ");
                print_coords(m.path[j+1]);
            }
            printf(" (Len: %d, Ladies: %d, First Lady: %d, Is Lady: %d)", 
                   m.length, m.captured_ladies_count, m.first_captured_is_lady, m.is_lady_move);
        }
        printf("\n");
    }
    printf("------------------------------------------------\n");
}

void print_move_description(Move m) {
    print_coords(m.path[0]);
    
    if (m.length == 0) {
        printf("-");
        print_coords(m.path[1]);
    } else {
        for (int i=0; i<m.length; i++) {
            printf("x");
            print_coords(m.path[i+1]);
        }
    }
}
