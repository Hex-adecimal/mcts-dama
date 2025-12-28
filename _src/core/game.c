/**
 * =============================================================================
 * game.c - Core Game State Management for Italian Checkers
 * =============================================================================
 * 
 * This module handles the fundamental game state operations:
 * 
 *   - Zobrist hashing for transposition table support
 *   - Game state initialization
 *   - Move application (simple moves and captures)
 *   - Debug printing utilities
 * 
 * State Representation:
 *   The board uses 4 bitboards (64-bit integers) to track piece positions:
 *     - white_pieces: White pawns
 *     - white_ladies: White promoted pieces (Queens)
 *     - black_pieces: Black pawns  
 *     - black_ladies: Black promoted pieces (Queens)
 * 
 * Square Indexing:
 *   Squares are numbered 0-63, where:
 *     - Square 0 = A1 (bottom-left)
 *     - Square 7 = H1 (bottom-right)
 *     - Square 56 = A8 (top-left)
 *     - Square 63 = H8 (top-right)
 * 
 * =============================================================================
 */

#include "game.h"
#include <string.h>
#include <stdio.h>

/* =============================================================================
 * ZOBRIST HASHING
 * =============================================================================
 * 
 * Zobrist hashing provides fast incremental hash updates for game states.
 * Each position has a unique 64-bit hash derived from XORing random values
 * for each piece on each square.
 * 
 * Key Structure: zobrist_keys[color][piece_type][square]
 *   - color: WHITE(0) or BLACK(1)
 *   - piece_type: PAWN(0) or LADY(1)
 *   - square: 0-63
 * 
 * The hash is updated incrementally when:
 *   - A piece moves (XOR out old position, XOR in new position)
 *   - A piece is captured (XOR out captured piece)
 *   - A pawn promotes (XOR out pawn, XOR in lady)
 *   - Turn changes (XOR with zobrist_black_move)
 */

/** Random values for each piece at each square: [color][piece_type][square] */
uint64_t zobrist_keys[2][2][64];

/** Random value XORed when it's Black's turn */
uint64_t zobrist_black_move;

/**
 * Simple 64-bit PRNG using XorShift algorithm.
 * Uses a fixed seed for reproducible hash values across runs.
 */
static uint64_t rand64(void) {
    static uint64_t seed = 0x987654321ULL;
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
}

/**
 * Initialize Zobrist hash tables.
 * 
 * Must be called once at program startup before any game operations.
 * Generates random 64-bit values for all piece/square combinations.
 */
void zobrist_init(void) {
    for (int color = 0; color < 2; color++) {
        for (int piece_type = 0; piece_type < 2; piece_type++) {
            for (int square = 0; square < 64; square++) {
                zobrist_keys[color][piece_type][square] = rand64();
            }
        }
    }
    zobrist_black_move = rand64();
}

/**
 * Compute hash contribution for all pieces in a bitboard.
 * 
 * @param bb          Bitboard with piece positions
 * @param color       Piece color (WHITE or BLACK)
 * @param piece_type  Piece type (0=pawn, 1=lady)
 * @return            XOR of all Zobrist keys for pieces in the bitboard
 */
static uint64_t hash_bitboard(Bitboard bb, int color, int piece_type) {
    uint64_t hash = 0;
    while (bb) {
        int sq = __builtin_ctzll(bb);
        hash ^= zobrist_keys[color][piece_type][sq];
        bb &= (bb - 1);  /* Clear lowest set bit */
    }
    return hash;
}

/**
 * Compute complete Zobrist hash from scratch.
 * 
 * Used for initialization and verification. During normal play,
 * the hash is updated incrementally for better performance.
 * 
 * @param state  Game state to hash
 * @return       64-bit Zobrist hash
 */
static uint64_t compute_full_hash(const GameState *state) {
    uint64_t hash = hash_bitboard(state->white_pieces, WHITE, 0)
                  ^ hash_bitboard(state->white_ladies, WHITE, 1)
                  ^ hash_bitboard(state->black_pieces, BLACK, 0)
                  ^ hash_bitboard(state->black_ladies, BLACK, 1);
    
    /* Include turn information in hash */
    if (state->current_player == BLACK) {
        hash ^= zobrist_black_move;
    }
    return hash;
}

/* =============================================================================
 * GAME INITIALIZATION
 * ============================================================================= */

/**
 * Initialize a new game with standard starting position.
 * 
 * Sets up the board with:
 *   - White pawns on rows 1-3 (dark squares)
 *   - Black pawns on rows 6-8 (dark squares)
 *   - White to move first
 *   - No Ladies on the board
 * 
 * Board Layout (after init):
 *   Row 8: b . b . b . b .   (Black)
 *   Row 7: . b . b . b . b
 *   Row 6: b . b . b . b .
 *   Row 5: . . . . . . . .   (Empty)
 *   Row 4: . . . . . . . .
 *   Row 3: . w . w . w . w   (White)
 *   Row 2: w . w . w . w .
 *   Row 1: . w . w . w . w
 * 
 * @param state  Pointer to GameState to initialize
 */
void init_game(GameState *state) {
    /* Clear all bitboards */
    state->white_ladies = 0;
    state->white_pieces = 0;
    state->black_ladies = 0;
    state->black_pieces = 0;

    state->current_player = WHITE;
    state->moves_without_captures = 0;

    /* 
     * White starting position (rows 1-3):
     *   Row 1 (bits 0-7):   0xAA = 10101010 (A1, C1, E1, G1)
     *   Row 2 (bits 8-15):  0x55 = 01010101 (B2, D2, F2, H2)  
     *   Row 3 (bits 16-23): 0xAA = 10101010 (A3, C3, E3, G3)
     */
    state->white_pieces = 0x0000000000AA55AAULL;

    /* 
     * Black starting position (rows 6-8):
     *   Row 6 (bits 40-47): 0x55 = 01010101 (B6, D6, F6, H6)
     *   Row 7 (bits 48-55): 0xAA = 10101010 (A7, C7, E7, G7)
     *   Row 8 (bits 56-63): 0x55 = 01010101 (B8, D8, F8, H8)
     */
    state->black_pieces = 0x55AA550000000000ULL;
    
    /* Compute initial hash */
    state->hash = compute_full_hash(state);
}

/* =============================================================================
 * MOVE APPLICATION
 * ============================================================================= */

/**
 * Move a piece from one square to another (internal helper).
 * 
 * Updates the appropriate bitboards and handles pawn promotion.
 * Does NOT update hash or captured pieces - those are handled by apply_move.
 * 
 * @param s     Game state to modify
 * @param from  Source square (0-63)
 * @param to    Destination square (0-63)
 * @param us    Color of moving player (WHITE or BLACK)
 */
static void perform_movement(GameState *s, int from, int to, int us) {
    // use helper
    Bitboard *own_pieces = (us == WHITE) ? &s->white_pieces : &s->black_pieces;
    Bitboard *own_ladies = (us == WHITE) ? &s->white_ladies : &s->black_ladies;
    
    // use lookup tables
    Bitboard move_mask = (1ULL << from) | (1ULL << to);
    Bitboard is_piece = (*own_pieces & (1ULL << from));

    /* Toggle piece position (removes from 'from', adds to 'to') */
    if (is_piece) *own_pieces ^= move_mask;
    else          *own_ladies ^= move_mask;

    /* Handle pawn promotion to Lady */
    if (is_piece && ((1ULL << to) & PROM_RANKS[us])) {
        *own_pieces &= ~(1ULL << to);  /* Remove from pawns */
        *own_ladies |= (1ULL << to);   /* Add to ladies */
    }
}

/**
 * Apply a move to the game state.
 * 
 * This is the main entry point for executing moves. Handles:
 *   - Simple moves (length = 0): Single diagonal step
 *   - Capture chains (length > 0): Sequence of jumps with captures
 *   - Pawn promotion when reaching back rank
 *   - Incremental Zobrist hash update
 *   - Turn switching
 *   - 40-move draw counter (only with Ladies on board)
 * 
 * @param state  Game state to modify
 * @param move   Move to apply (generated by movegen)
 */
void apply_move(GameState *state, const Move *move) {
    int us = state->current_player;
    int from = move->path[0];
    
    /* 
     * Determine destination square:
     *   - Simple move (length=0): destination is path[1]
     *   - Capture chain (length>0): destination is path[length]
     */
    int to = (move->length == 0) ? move->path[1] : move->path[move->length];

    /* Zobrist: Remove moving piece from source square */
    int is_lady = move->is_lady_move;
    state->hash ^= zobrist_keys[us][is_lady][from];

    /* Execute the piece movement (handles promotion internally) */
    perform_movement(state, from, to, us);

    /* 
     * Zobrist: Add piece at destination
     * Must check if promotion occurred to use correct piece type
     */
    int now_lady = (us == WHITE) 
        ? check_bit(state->white_ladies, to) 
        : check_bit(state->black_ladies, to);
    state->hash ^= zobrist_keys[us][now_lady][to];

    /* Process captured pieces (if this is a capture move) */
    if (move->length > 0) {
        int them = us ^ 1;
        Bitboard *enemy_pieces = (them == BLACK) ? &state->black_pieces : &state->white_pieces;
        Bitboard *enemy_ladies = (them == BLACK) ? &state->black_ladies : &state->white_ladies;
        
        for (int i = 0; i < move->length; i++) {
            int cap_sq = move->captured_squares[i];
            
            /* Zobrist: Remove captured piece from hash */
            int is_captured_lady = check_bit(*enemy_ladies, cap_sq);
            state->hash ^= zobrist_keys[them][is_captured_lady][cap_sq];

            /* Remove captured piece from bitboards */
            Bitboard remove_mask = ~(1ULL << cap_sq);
            *enemy_pieces &= remove_mask;
            *enemy_ladies &= remove_mask;
        }
        
        /* Reset 40-move draw counter on capture */
        state->moves_without_captures = 0;
    } else {
        /* 
         * 40-move draw rule (Italian Checkers):
         * Only count moves without captures when Ladies exist.
         * This prevents premature draws in pawn-only endings.
         */
        if (state->white_ladies || state->black_ladies) {
            state->moves_without_captures++;
        } else {
            state->moves_without_captures = 0;
        }
    }

    /* Switch turn and update hash */
    state->current_player = (Color)(us ^ 1);
    state->hash ^= zobrist_black_move;
}

/* =============================================================================
 * DEBUG PRINTING UTILITIES
 * ============================================================================= */

/**
 * Print a square index as algebraic notation (e.g., "A1", "H8").
 * 
 * @param square_idx  Square index (0-63)
 */
void print_coords(int square_idx) {
    int rank = (square_idx / 8) + 1;
    char file = (square_idx % 8) + 'A';
    printf("%c%d", file, rank);
}

/**
 * Print the current board state in ASCII format.
 * 
 * Display format:
 *   - 'w' = white pawn
 *   - 'W' = white lady
 *   - 'b' = black pawn
 *   - 'B' = black lady
 *   - '.' = empty square
 * 
 * @param state  Game state to display
 */
void print_board(const GameState *state) {
    printf("\n   A B C D E F G H\n");
    printf("  +---------------+\n");

    for (int rank = 7; rank >= 0; rank--) {
        printf("%d |", rank + 1);
        
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            char c = '.';

            if (check_bit(state->white_pieces, sq))      c = 'w';
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

/**
 * Print all moves in a move list with detailed information.
 * 
 * Shows path, captured squares, and priority metrics for each move.
 * Useful for debugging move generation.
 * 
 * @param list  Move list to display
 */
void print_move_list(MoveList *list) {
    printf("------------------------------------------------\n");
    printf("Found %d possible moves:\n", list->count);
    
    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        printf("%d) ", i + 1);
        
        if (m.length == 0) {
            /* Simple move: from -> to */
            print_coords(m.path[0]);
            printf(" -> ");
            print_coords(m.path[1]);
        } else {
            /* Capture chain: from x cap1 -> sq1 x cap2 -> sq2 ... */
            print_coords(m.path[0]);
            for (int j = 0; j < m.length; j++) {
                printf(" x ");
                print_coords(m.captured_squares[j]);
                printf(" -> ");
                print_coords(m.path[j + 1]);
            }
            printf(" (Len: %d, Ladies: %d, FirstLady: %d, IsLady: %d)", 
                   m.length, m.captured_ladies_count, 
                   m.first_captured_is_lady, m.is_lady_move);
        }
        printf("\n");
    }
    printf("------------------------------------------------\n");
}

/**
 * Print a single move in compact notation.
 * 
 * Format examples:
 *   - Simple move: "A3-B4"
 *   - Capture: "A3xB4xC5"
 * 
 * @param m  Move to display
 */
void print_move_description(Move m) {
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
