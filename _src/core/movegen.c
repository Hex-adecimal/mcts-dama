/**
 * =============================================================================
 * movegen.c - Move Generation for Italian Checkers (Dama Italiana)
 * =============================================================================
 * 
 * This module handles all legal move generation for the game. It implements
 * the complete Italian Checkers ruleset including:
 * 
 *   - Simple diagonal moves (pawns forward, ladies in all directions)
 *   - Capture chains with recursive jump sequences
 *   - Mandatory capture rule (must capture if possible)
 *   - Italian priority rules for capture selection
 *   - Pawn promotion to Lady on back rank
 * 
 * Key Design Decisions:
 *   - Uses pre-computed lookup tables for fast move generation
 *   - Bitboard representation for efficient board state queries
 *   - Recursive capture chain finding with backtracking
 *   - Priority-based filtering per Italian rules
 * 
 * Italian Checkers Priority Rules (in order):
 *   1. Must take the longest capture sequence
 *   2. Ladies must capture before pawns (if equal length)
 *   3. Prefer capturing more Ladies
 *   4. Prefer capturing a Lady first in the sequence
 * 
 * Board Layout (64-square, 8x8):
 *   - Squares 0-7:   Row 0 (Black's back rank)
 *   - Squares 56-63: Row 7 (White's back rank)
 *   - White moves upward (increasing row), Black moves downward
 * 
 * =============================================================================
 */

#include "movegen.h"
#include "game.h"

/** Maximum depth for capture chain recursion (safety limit) */
#define MAX_CHAIN_LENGTH 12

/* =============================================================================
 * DIRECTION CONSTANTS
 * =============================================================================
 * 
 * The board uses a linear index (0-63) where:
 *   - Moving +1 = move right on same row
 *   - Moving +8 = move up one row
 * 
 * Diagonal directions are combinations:
 *   - NE (North-East): +9  = up-right
 *   - NW (North-West): +7  = up-left
 *   - SE (South-East): -7  = down-right
 *   - SW (South-West): -9  = down-left
 */

/** All 4 diagonal directions: NE(+9), NW(+7), SE(-7), SW(-9) */
static const int ALL_DIRS[4] = { 9, 7, -7, -9 };

/** 
 * Border masks for jump validity checks.
 * Prevents wrapping around board edges during 2-square jumps.
 * Index corresponds to ALL_DIRS order.
 */
static const uint64_t JUMP_MASKS[4] = { NOT_FILE_GH, NOT_FILE_AB, NOT_FILE_GH, NOT_FILE_AB };

/* =============================================================================
 * MOVE LOOKUP TABLES
 * =============================================================================
 * 
 * Pre-computed tables for fast move generation. Initialized once at startup.
 * Using tables eliminates repeated boundary calculations during move gen.
 * 
 * Table Structure:
 *   PAWN_MOVE_TARGETS[color][square][direction] -> target bitboard
 *   LADY_MOVE_TARGETS[square][direction] -> target bitboard
 *   JUMP_LANDING[square][direction] -> landing square after capture
 *   JUMP_OVER_SQ[square][direction] -> captured piece square
 */

static Bitboard PAWN_MOVE_TARGETS[2][64][2];   /**< Pawn targets: [color][from][dir] */
static Bitboard LADY_MOVE_TARGETS[64][4];      /**< Lady targets: [from][dir] 4 directions */
static Bitboard JUMP_LANDING[64][4];           /**< Landing square after jump */
static Bitboard JUMP_OVER_SQ[64][4];           /**< Captured square in jump */
static int move_tables_initialized = 0;

/**
 * Initialize move lookup tables.
 * 
 * Must be called once before any move generation. Pre-computes all valid
 * move targets for every square and piece type. This eliminates boundary
 * checking during actual move generation.
 * 
 * Table contents:
 *   - Pawn moves: 2 forward diagonals per color
 *   - Lady moves: 4 diagonals (all directions)
 *   - Jump targets: Landing square + captured square for each direction
 */
void init_move_tables(void) {
    if (move_tables_initialized) return;
    
    for (int sq = 0; sq < 64; sq++) {
        int row = sq / 8;
        int col = sq % 8;
        
        /* Clear all entries for this square */
        for (int d = 0; d < 4; d++) {
            LADY_MOVE_TARGETS[sq][d] = 0;
            JUMP_LANDING[sq][d] = 0;
            JUMP_OVER_SQ[sq][d] = 0;
        }
        PAWN_MOVE_TARGETS[WHITE][sq][0] = 0;
        PAWN_MOVE_TARGETS[WHITE][sq][1] = 0;
        PAWN_MOVE_TARGETS[BLACK][sq][0] = 0;
        PAWN_MOVE_TARGETS[BLACK][sq][1] = 0;
        
        /* 
         * WHITE pawn moves (forward = increasing row)
         * Direction 0: NE (+9), Direction 1: NW (+7)
         */
        if (row < 7) {
            if (col < 7) PAWN_MOVE_TARGETS[WHITE][sq][0] = 1ULL << (sq + 9);
            if (col > 0) PAWN_MOVE_TARGETS[WHITE][sq][1] = 1ULL << (sq + 7);
        }
        
        /* 
         * BLACK pawn moves (forward = decreasing row)
         * Direction 0: SE (-7), Direction 1: SW (-9)
         */
        if (row > 0) {
            if (col < 7) PAWN_MOVE_TARGETS[BLACK][sq][0] = 1ULL << (sq - 7);
            if (col > 0) PAWN_MOVE_TARGETS[BLACK][sq][1] = 1ULL << (sq - 9);
        }
        
        /* Lady moves: all 4 diagonal directions, 1 step each */
        if (row < 7 && col < 7) LADY_MOVE_TARGETS[sq][0] = 1ULL << (sq + 9);  /* NE */
        if (row < 7 && col > 0) LADY_MOVE_TARGETS[sq][1] = 1ULL << (sq + 7);  /* NW */
        if (row > 0 && col < 7) LADY_MOVE_TARGETS[sq][2] = 1ULL << (sq - 7);  /* SE */
        if (row > 0 && col > 0) LADY_MOVE_TARGETS[sq][3] = 1ULL << (sq - 9);  /* SW */
        
        /* 
         * Jump targets: 2 squares in each direction
         * Requires 2 rows/cols of clearance to avoid board edge wrapping
         */
        if (row < 6 && col < 6) {  /* NE jump (+18) */
            JUMP_LANDING[sq][0] = 1ULL << (sq + 18);
            JUMP_OVER_SQ[sq][0] = 1ULL << (sq + 9);
        }
        if (row < 6 && col > 1) {  /* NW jump (+14) */
            JUMP_LANDING[sq][1] = 1ULL << (sq + 14);
            JUMP_OVER_SQ[sq][1] = 1ULL << (sq + 7);
        }
        if (row > 1 && col < 6) {  /* SE jump (-14) */
            JUMP_LANDING[sq][2] = 1ULL << (sq - 14);
            JUMP_OVER_SQ[sq][2] = 1ULL << (sq - 7);
        }
        if (row > 1 && col > 1) {  /* SW jump (-18) */
            JUMP_LANDING[sq][3] = 1ULL << (sq - 18);
            JUMP_OVER_SQ[sq][3] = 1ULL << (sq - 9);
        }
    }
    
    move_tables_initialized = 1;
}

/* =============================================================================
 * INTERNAL HELPER FUNCTIONS
 * ============================================================================= */

/**
 * Add a simple (non-capture) move to the move list.
 * 
 * @param list   Move list to append to
 * @param from   Source square (0-63)
 * @param to     Destination square (0-63)
 * @param is_lady  1 if moving piece is a Lady, 0 for pawn
 */
static void add_simple_move(MoveList *list, int from, int to, int is_lady) {
    if (list->count >= MAX_MOVES) return;
    
    Move *m = &list->moves[list->count++];
    m->path[0] = from;
    m->path[1] = to;
    m->length = 0;  /* 0 = simple move (no captures) */
    m->captured_ladies_count = 0;
    m->is_lady_move = is_lady;
    m->first_captured_is_lady = 0;
}

/**
 * Save a completed capture chain to the move list.
 * 
 * Computes priority metrics needed for Italian rules filtering:
 *   - Number of Ladies captured
 *   - Whether first capture was a Lady
 * 
 * @param list      Move list to append to
 * @param s         Current game state (for checking captured piece types)
 * @param path      Array of squares in the capture path
 * @param captured  Array of captured piece squares
 * @param depth     Number of captures in this chain
 * @param is_lady   1 if moving piece is a Lady
 */
static void save_move(MoveList *list, const GameState *s, 
                      int path[], int captured[], int depth, int is_lady) {
    if (list->count >= MAX_MOVES) return;
    if (depth > 11) depth = 11;  /* Safety cap */
    
    Move *m = &list->moves[list->count++];
    
    /* Copy path and captured squares */
    for (int i = 0; i <= depth; i++) {
        m->path[i] = (uint8_t)path[i];
    }
    for (int i = 0; i < depth; i++) {
        m->captured_squares[i] = (uint8_t)captured[i];
    }
    
    m->length = (uint8_t)depth;
    m->is_lady_move = (uint8_t)is_lady;
    m->captured_ladies_count = 0;
    m->first_captured_is_lady = 0;

    /* 
     * Compute priority metrics for Italian rules.
     * Check each captured square to determine if it held a Lady.
     */
    Bitboard all_ladies = s->white_ladies | s->black_ladies;
    for (int k = 0; k < depth; k++) {
        if (all_ladies & (1ULL << captured[k])) {
            m->captured_ladies_count++;
            if (k == 0) m->first_captured_is_lady = 1;
        }
    }
}

/**
 * Recursively find all capture chains from a given position.
 * 
 * Implements depth-first search to find all possible capture sequences.
 * Each branch of the search tree represents a different capture chain.
 * When no more captures are possible, the current chain is saved.
 * 
 * Special Rules Handled:
 *   - Pawns can only capture forward (color-dependent direction)
 *   - Pawns cannot capture Ladies (Italian rule)
 *   - Promotion ends the capture chain immediately
 *   - Same piece cannot be captured twice (tracked via modified enemy bitboards)
 * 
 * @param s             Current game state
 * @param current_sq    Current position of the capturing piece
 * @param path          Array tracking squares visited in this chain
 * @param captured      Array tracking captured piece squares
 * @param depth         Current depth (number of captures so far)
 * @param is_lady       1 if capturing piece is a Lady
 * @param enemy_pieces  Remaining enemy pawns (excludes already captured)
 * @param enemy_ladies  Remaining enemy Ladies (excludes already captured)
 * @param occupied      All occupied squares (excludes captured pieces)
 * @param list          Move list to save completed chains to
 */
static void find_captures(const GameState *s, int current_sq, 
                          int path[], int captured[], int depth, 
                          int is_lady, Bitboard enemy_pieces, Bitboard enemy_ladies, 
                          Bitboard occupied, MoveList *list) {
    
    int found_continuation = 0;
    int us = s->current_player;

    /* Setup direzioni: Dame 0-3, Pedine dipendenti dal colore */
    int start_dir = is_lady ? 0 : (us == WHITE ? 0 : 2);
    int end_dir   = is_lady ? 4 : (us == WHITE ? 2 : 4);

    for (int dir = start_dir; dir < end_dir; dir++) {
        /* OTTIMIZZAZIONE 1: Lookup immediato della casella da saltare */
        uint64_t over_mask = JUMP_OVER_SQ[current_sq][dir];
        
        /* Se è 0, siamo sul bordo o il salto è impossibile -> skip */
        if (!over_mask) continue;

        /* OTTIMIZZAZIONE 2: Controllo rapido presenza nemico */
        if (!((over_mask & enemy_pieces) || (over_mask & enemy_ladies))) continue;
        
        /* Regola Italiana: Pedina non mangia Dama */
        if (!is_lady && (over_mask & enemy_ladies)) continue;

        /* OTTIMIZZAZIONE 3: Lookup atterraggio */
        uint64_t land_mask = JUMP_LANDING[current_sq][dir];
        
        /* La casella di atterraggio deve essere vuota */
        if (occupied & land_mask) continue;

        /* --- SALTO VALIDO TROVATO --- */

        /* Calcoliamo gli indici solo ora che servono davvero */
        int bridge_sq = __builtin_ctzll(over_mask);
        int land_sq   = __builtin_ctzll(land_mask);

        path[depth + 1] = land_sq;
        captured[depth] = bridge_sq;
        found_continuation = 1;

        /* Controllo Promozione */
        int promoted = (!is_lady && (land_mask & PROM_RANKS[us]));

        if (promoted) {
            /* Stop immediato per promozione */
            save_move(list, s, path, captured, depth + 1, is_lady);
        } else {
            /* Ricorsione: cerca salti multipli */
            find_captures(s, land_sq, path, captured, depth + 1, is_lady, 
                          enemy_pieces & ~over_mask, 
                          enemy_ladies & ~over_mask, 
                          occupied & ~over_mask, 
                          list);
        }
    }

    /* CRITICO: Salva la catena quando non ci sono più catture possibili */
    if (!found_continuation && depth > 0) {
        save_move(list, s, path, captured, depth, is_lady);
    }
}

/**
 * Calculate priority score for a capture move.
 * 
 * Encodes Italian Checkers priority rules into a single comparable integer.
 * Higher score = higher priority. Bit layout:
 *   - Bits 24+: Capture chain length (most important)
 *   - Bits 20-23: Is Lady moving (Ladies have priority)
 *   - Bits 10-19: Number of Ladies captured
 *   - Bits 0-9: First capture was Lady
 * 
 * @param m  Move to score
 * @return   Integer score for comparison
 */
static int calculate_score(const Move *m) {
    return (m->length << 24) | 
           (m->is_lady_move << 20) | 
           (m->captured_ladies_count << 10) | 
           m->first_captured_is_lady;
}

/**
 * Filter moves based on Italian Checkers priority rules.
 * 
 * Italian rules require selecting only the "best" captures when multiple
 * are available. This function removes all moves except those with the
 * maximum priority score.
 * 
 * @param list  Move list to filter (modified in place)
 */
static void filter_moves(MoveList *list) {
    if (list->count <= 1) return;
    
    /* Pass 1: Find maximum score */
    int max_score = -1;
    for (int i = 0; i < list->count; i++) {
        int score = calculate_score(&list->moves[i]);
        if (score > max_score) max_score = score;
    }
    
    /* Pass 2: Keep only moves matching max score */
    int write_idx = 0;
    for (int i = 0; i < list->count; i++) {
        if (calculate_score(&list->moves[i]) == max_score) {
            list->moves[write_idx++] = list->moves[i];
        }
    }
    list->count = write_idx;
}

/* =============================================================================
 * PUBLIC API
 * ============================================================================= */

/**
 * Generate all legal simple (non-capture) moves.
 * 
 * Uses pre-computed lookup tables for fast generation.
 * Does not check for captures - caller should check generate_captures first.
 * 
 * @param s     Current game state
 * @param list  Move list to populate (appends to existing)
 */
void generate_simple_moves(const GameState *s, MoveList *list) {
    int us = s->current_player;
    Bitboard own_pieces = (us == WHITE) ? s->white_pieces : s->black_pieces;
    Bitboard own_ladies = (us == WHITE) ? s->white_ladies : s->black_ladies;
    Bitboard empty = get_empty_squares(s);

    /* Generate pawn moves using lookup tables */
    Bitboard pawns = own_pieces;
    while (pawns) {
        int from = __builtin_ctzll(pawns);
        
        for (int dir = 0; dir < 2; dir++) {
            Bitboard target = PAWN_MOVE_TARGETS[us][from][dir];
            if (target & empty) {
                int to = __builtin_ctzll(target);
                add_simple_move(list, from, to, 0);
            }
        }
        pawns &= (pawns - 1);  /* Clear lowest set bit */
    }

    /* Generate Lady moves (all 4 directions) */
    Bitboard ladies = own_ladies;
    while (ladies) {
        int from = __builtin_ctzll(ladies);
        
        for (int dir = 0; dir < 4; dir++) {
            Bitboard target = LADY_MOVE_TARGETS[from][dir];
            if (target & empty) {
                int to = __builtin_ctzll(target);
                add_simple_move(list, from, to, 1);
            }
        }
        ladies &= (ladies - 1);
    }
}

/**
 * Generate all legal capture moves.
 * 
 * Finds all possible capture chains for all pieces of the current player.
 * Does NOT apply Italian priority filtering - use filter_moves() after.
 * 
 * @param s     Current game state
 * @param list  Move list to populate (appends to existing)
 */
void generate_captures(const GameState *s, MoveList *list) {
    int us = s->current_player;
    int them = us ^ 1;
    
    Bitboard own_pieces = (us == WHITE) ? s->white_pieces : s->black_pieces;
    Bitboard own_ladies = (us == WHITE) ? s->white_ladies : s->black_ladies;
    Bitboard enemy_pieces = (them == WHITE) ? s->white_pieces : s->black_pieces;
    Bitboard enemy_ladies = (them == WHITE) ? s->white_ladies : s->black_ladies;
    Bitboard occupied = get_all_occupied(s);
    
    /* Process pawns then Ladies */
    Bitboard pieces[2] = { own_pieces, own_ladies };
    int is_lady_flag[2] = { 0, 1 };
    
    for (int t = 0; t < 2; t++) {
        Bitboard bb = pieces[t];
        while (bb) {
            int sq = __builtin_ctzll(bb);
            
            int path[MAX_CHAIN_LENGTH + 1];
            int captured[MAX_CHAIN_LENGTH];
            path[0] = sq;
            
            /* Find all capture chains starting from this piece */
            find_captures(s, sq, path, captured, 0, is_lady_flag[t], 
                          enemy_pieces, enemy_ladies, 
                          occupied & ~(1ULL << sq),  /* Remove self from occupied */
                          list);
            
            bb &= (bb - 1);
        }
    }
}

/**
 * Generate all legal moves for the current player.
 * 
 * This is the main entry point for move generation. Implements the
 * mandatory capture rule: if any captures are available, only capture
 * moves are returned (filtered by Italian priority rules).
 * 
 * @param s     Current game state
 * @param list  Move list to populate (cleared first)
 */
void generate_moves(const GameState *s, MoveList *list) {
    list->count = 0;
    
    /* Step 1: Try to generate captures */
    generate_captures(s, list);
    
    if (list->count > 0) {
        /* Captures exist - filter by Italian priority rules */
        filter_moves(list);
    } else {
        /* No captures - generate simple moves instead */
        generate_simple_moves(s, list);
    }
}

/**
 * Check if a square is threatened by the opponent.
 * 
 * Useful for AI evaluation - checks if a piece on the given square
 * could be captured by the opponent on their next move.
 * 
 * @param state   Current game state
 * @param square  Square to check (0-63)
 * @return        1 if square is threatened, 0 otherwise
 */
int is_square_threatened(const GameState *state, int square) {
    MoveList enemy_moves;
    generate_moves(state, &enemy_moves);
    
    for (int i = 0; i < enemy_moves.count; i++) {
        Move *m = &enemy_moves.moves[i];
        
        /* Check if any capture in this move takes the target square */
        if (m->length > 0) {
            for (int k = 0; k < m->length; k++) {
                if (m->captured_squares[k] == square) return 1;
            }
        }
    }
    return 0;
}
