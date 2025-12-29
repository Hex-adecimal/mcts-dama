/**
 * movegen.c - Move Generation for Italian Checkers
 * Uses lookup tables for fast move/capture generation.
 * Implements Italian priority rules for mandatory captures.
 */

#include "movegen.h"
#include "game.h"

// --- Lookup Tables ---
static Bitboard PAWN_MOVE_TARGETS[NUM_COLORS][NUM_SQUARES][2];
static Bitboard LADY_MOVE_TARGETS[NUM_SQUARES][NUM_DIRECTIONS];
static Bitboard JUMP_LANDING[NUM_SQUARES][NUM_DIRECTIONS];
static Bitboard JUMP_OVER_SQ[NUM_SQUARES][NUM_DIRECTIONS];

// Cached mobility: all squares a piece can move/jump to from each square
static Bitboard PAWN_MOBILITY[NUM_COLORS][NUM_SQUARES];  // All reachable squares for pawns
static Bitboard LADY_MOBILITY[NUM_SQUARES];               // All reachable squares for ladies
static Bitboard CAN_JUMP_FROM[NUM_SQUARES];               // Non-zero if any jump possible

static int move_tables_initialized = 0;

// --- CaptureContext with optimized caching ---
typedef struct {
    const GameState *state;
    MoveList *list;
    int path[MAX_CHAIN_LENGTH + 1];
    int captured[MAX_CHAIN_LENGTH];
    Bitboard enemy_pieces;
    Bitboard enemy_ladies;
    Bitboard all_enemy;      // Cached: enemy_pieces | enemy_ladies
    Bitboard occupied;
    uint8_t is_lady;
} CaptureContext;

// --- Forward declarations ---
static void find_captures(CaptureContext *ctx, int sq, int depth);

// --- Initialization ---
void init_move_tables(void) {
    if (move_tables_initialized) return;
    
    for (int sq = 0; sq < NUM_SQUARES; sq++) {
        const int row = ROW(sq);
        const int col = COL(sq);
        
        // Clear all entries
        for (int d = 0; d < NUM_DIRECTIONS; d++) {
            LADY_MOVE_TARGETS[sq][d] = 0;
            JUMP_LANDING[sq][d] = 0;
            JUMP_OVER_SQ[sq][d] = 0;
        }
        for (int c = 0; c < NUM_COLORS; c++) {
            PAWN_MOVE_TARGETS[c][sq][0] = 0;
            PAWN_MOVE_TARGETS[c][sq][1] = 0;
            PAWN_MOBILITY[c][sq] = 0;
        }
        LADY_MOBILITY[sq] = 0;
        CAN_JUMP_FROM[sq] = 0;
        
        // WHITE pawns: NE, NW
        if (row < 7) {
            if (col < 7) PAWN_MOVE_TARGETS[WHITE][sq][0] = BIT(sq + OFFSET_NE);
            if (col > 0) PAWN_MOVE_TARGETS[WHITE][sq][1] = BIT(sq + OFFSET_NW);
        }
        PAWN_MOBILITY[WHITE][sq] = PAWN_MOVE_TARGETS[WHITE][sq][0] | PAWN_MOVE_TARGETS[WHITE][sq][1];
        
        // BLACK pawns: SE, SW
        if (row > 0) {
            if (col < 7) PAWN_MOVE_TARGETS[BLACK][sq][0] = BIT(sq + OFFSET_SE);
            if (col > 0) PAWN_MOVE_TARGETS[BLACK][sq][1] = BIT(sq + OFFSET_SW);
        }
        PAWN_MOBILITY[BLACK][sq] = PAWN_MOVE_TARGETS[BLACK][sq][0] | PAWN_MOVE_TARGETS[BLACK][sq][1];
        
        // Lady moves: all 4 diagonals
        if (row < 7 && col < 7) LADY_MOVE_TARGETS[sq][DIR_NE] = BIT(sq + OFFSET_NE);
        if (row < 7 && col > 0) LADY_MOVE_TARGETS[sq][DIR_NW] = BIT(sq + OFFSET_NW);
        if (row > 0 && col < 7) LADY_MOVE_TARGETS[sq][DIR_SE] = BIT(sq + OFFSET_SE);
        if (row > 0 && col > 0) LADY_MOVE_TARGETS[sq][DIR_SW] = BIT(sq + OFFSET_SW);
        
        for (int d = 0; d < NUM_DIRECTIONS; d++) {
            LADY_MOBILITY[sq] |= LADY_MOVE_TARGETS[sq][d];
        }
        
        // Jump targets
        if (row < 6 && col < 6) {
            JUMP_LANDING[sq][DIR_NE] = BIT(sq + JUMP_NE);
            JUMP_OVER_SQ[sq][DIR_NE] = BIT(sq + OFFSET_NE);
        }
        if (row < 6 && col > 1) {
            JUMP_LANDING[sq][DIR_NW] = BIT(sq + JUMP_NW);
            JUMP_OVER_SQ[sq][DIR_NW] = BIT(sq + OFFSET_NW);
        }
        if (row > 1 && col < 6) {
            JUMP_LANDING[sq][DIR_SE] = BIT(sq + JUMP_SE);
            JUMP_OVER_SQ[sq][DIR_SE] = BIT(sq + OFFSET_SE);
        }
        if (row > 1 && col > 1) {
            JUMP_LANDING[sq][DIR_SW] = BIT(sq + JUMP_SW);
            JUMP_OVER_SQ[sq][DIR_SW] = BIT(sq + OFFSET_SW);
        }
        
        // Cache: can any jump happen from this square?
        for (int d = 0; d < NUM_DIRECTIONS; d++) {
            CAN_JUMP_FROM[sq] |= JUMP_LANDING[sq][d];
        }
    }
    
    move_tables_initialized = 1;
}

// --- Internal Helpers ---
static void add_simple_move(MoveList *list, const int from, const int to, const int is_lady) {
    if (list->count >= MAX_MOVES) return;
    
    Move *m = &list->moves[list->count++];
    m->path[0] = from;
    m->path[1] = to;
    m->length = 0;
    m->captured_ladies_count = 0;
    m->is_lady_move = is_lady;
    m->first_captured_is_lady = 0;
}

static void save_move(CaptureContext *ctx, const int depth) {
    if (ctx->list->count >= MAX_MOVES) return;
    const int safe_depth = (depth > MAX_CHAIN_LENGTH - 1) ? MAX_CHAIN_LENGTH - 1 : depth;
    
    Move *m = &ctx->list->moves[ctx->list->count++];
    
    for (int i = 0; i <= safe_depth; i++) {
        m->path[i] = (uint8_t)ctx->path[i];
    }
    for (int i = 0; i < safe_depth; i++) {
        m->captured_squares[i] = (uint8_t)ctx->captured[i];
    }
    
    m->length = (uint8_t)safe_depth;
    m->is_lady_move = ctx->is_lady;
    m->captured_ladies_count = 0;
    m->first_captured_is_lady = 0;

    // Compute Italian priority metrics using cached all_ladies
    const Bitboard all_ladies = ctx->state->piece[WHITE][LADY] | ctx->state->piece[BLACK][LADY];
    for (int k = 0; k < safe_depth; k++) {
        if (TEST_BIT(all_ladies, ctx->captured[k])) {
            m->captured_ladies_count++;
            if (k == 0) m->first_captured_is_lady = 1;
        }
    }
}

static void find_captures(CaptureContext *ctx, const int current_sq, const int depth) {
    int found_continuation = 0;
    const Color us = ctx->state->current_player;

    // Early exit: if no jumps possible from this square, skip
    if (!CAN_JUMP_FROM[current_sq]) {
        if (depth > 0) save_move(ctx, depth);
        return;
    }

    const int start_dir = ctx->is_lady ? 0 : (us == WHITE ? WHITE_DIR_START : BLACK_DIR_START);
    const int end_dir   = ctx->is_lady ? NUM_DIRECTIONS : (us == WHITE ? WHITE_DIR_END : BLACK_DIR_END);

    for (int dir = start_dir; dir < end_dir; dir++) {
        const Bitboard over_mask = JUMP_OVER_SQ[current_sq][dir];
        if (!over_mask) continue;

        // Use cached all_enemy for faster check
        if (!(over_mask & ctx->all_enemy)) continue;
        
        // Italian rule: Pawn cannot capture Lady
        if (!ctx->is_lady && (over_mask & ctx->enemy_ladies)) continue;

        const Bitboard land_mask = JUMP_LANDING[current_sq][dir];
        if (ctx->occupied & land_mask) continue;

        // Prefetch next square's lookup tables
        const int land_sq = __builtin_ctzll(land_mask);
        __builtin_prefetch(&JUMP_OVER_SQ[land_sq], 0, 3);
        __builtin_prefetch(&JUMP_LANDING[land_sq], 0, 3);

        // Valid jump found
        const int bridge_sq = __builtin_ctzll(over_mask);

        ctx->path[depth + 1] = land_sq;
        ctx->captured[depth] = bridge_sq;
        found_continuation = 1;

        const int promoted = (!ctx->is_lady && TEST_BIT(PROM_RANKS[us], land_sq));

        if (promoted) {
            save_move(ctx, depth + 1);
        } else {
            // Save state, recurse, restore
            const Bitboard saved_pieces = ctx->enemy_pieces;
            const Bitboard saved_ladies = ctx->enemy_ladies;
            const Bitboard saved_all = ctx->all_enemy;
            const Bitboard saved_occupied = ctx->occupied;
            
            ctx->enemy_pieces &= ~over_mask;
            ctx->enemy_ladies &= ~over_mask;
            ctx->all_enemy &= ~over_mask;
            ctx->occupied &= ~over_mask;
            
            find_captures(ctx, land_sq, depth + 1);
            
            ctx->enemy_pieces = saved_pieces;
            ctx->enemy_ladies = saved_ladies;
            ctx->all_enemy = saved_all;
            ctx->occupied = saved_occupied;
        }
    }

    if (!found_continuation && depth > 0) {
        save_move(ctx, depth);
    }
}

static int calculate_score(const Move *m) {
    return (m->length << 24) | (m->is_lady_move << 20) | 
           (m->captured_ladies_count << 10) | m->first_captured_is_lady;
}

static void filter_moves(MoveList *list) {
    if (list->count <= 1) return;
    
    int max_score = -1;
    for (int i = 0; i < list->count; i++) {
        const int score = calculate_score(&list->moves[i]);
        if (score > max_score) max_score = score;
    }
    
    int write_idx = 0;
    for (int i = 0; i < list->count; i++) {
        if (calculate_score(&list->moves[i]) == max_score) {
            list->moves[write_idx++] = list->moves[i];
        }
    }
    list->count = write_idx;
}

// --- Public API ---
void generate_simple_moves(const GameState *s, MoveList *list) {
    const Color us = s->current_player;
    const Bitboard empty = get_empty_squares(s);

    // Prefetch pawn mobility table
    __builtin_prefetch(&PAWN_MOBILITY[us][0], 0, 2);

    // Pawn moves
    Bitboard pawns = s->piece[us][PAWN];
    while (pawns) {
        const int from = __builtin_ctzll(pawns);
        
        // Prefetch next piece's data
        const Bitboard next_pawns = pawns & (pawns - 1);
        if (next_pawns) {
            const int next_sq = __builtin_ctzll(next_pawns);
            __builtin_prefetch(&PAWN_MOVE_TARGETS[us][next_sq], 0, 2);
        }
        
        for (int dir = 0; dir < NUM_PAWN_DIRS; dir++) {
            const Bitboard target = PAWN_MOVE_TARGETS[us][from][dir];
            if (target & empty) {
                add_simple_move(list, from, __builtin_ctzll(target), 0);
            }
        }
        POP_LSB(pawns);
    }

    // Lady moves
    Bitboard ladies = s->piece[us][LADY];
    while (ladies) {
        const int from = __builtin_ctzll(ladies);
        
        for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
            const Bitboard target = LADY_MOVE_TARGETS[from][dir];
            if (target & empty) {
                add_simple_move(list, from, __builtin_ctzll(target), 1);
            }
        }
        POP_LSB(ladies);
    }
}

void generate_captures(const GameState *s, MoveList *list) {
    const Color us = s->current_player;
    const Color them = us ^ 1;
    
    const Bitboard enemy_p = s->piece[them][PAWN];
    const Bitboard enemy_l = s->piece[them][LADY];
    
    CaptureContext ctx = {
        .state = s,
        .list = list,
        .enemy_pieces = enemy_p,
        .enemy_ladies = enemy_l,
        .all_enemy = enemy_p | enemy_l,  // Cached combined
        .occupied = get_all_occupied(s),
        .is_lady = 0
    };
    
    // Process pawns
    Bitboard pawns = s->piece[us][PAWN];
    while (pawns) {
        const int sq = __builtin_ctzll(pawns);
        
        // Skip if no jumps possible from this square
        if (CAN_JUMP_FROM[sq]) {
            ctx.path[0] = sq;
            ctx.is_lady = 0;
            ctx.occupied = get_all_occupied(s) & ~BIT(sq);
            ctx.enemy_pieces = enemy_p;
            ctx.enemy_ladies = enemy_l;
            ctx.all_enemy = enemy_p | enemy_l;
            find_captures(&ctx, sq, 0);
        }
        POP_LSB(pawns);
    }
    
    // Process ladies
    Bitboard ladies = s->piece[us][LADY];
    while (ladies) {
        const int sq = __builtin_ctzll(ladies);
        
        if (CAN_JUMP_FROM[sq]) {
            ctx.path[0] = sq;
            ctx.is_lady = 1;
            ctx.occupied = get_all_occupied(s) & ~BIT(sq);
            ctx.enemy_pieces = enemy_p;
            ctx.enemy_ladies = enemy_l;
            ctx.all_enemy = enemy_p | enemy_l;
            find_captures(&ctx, sq, 0);
        }
        POP_LSB(ladies);
    }
}

void generate_moves(const GameState *s, MoveList *list) {
    list->count = 0;
    generate_captures(s, list);
    
    if (list->count > 0) {
        filter_moves(list);
    } else {
        generate_simple_moves(s, list);
    }
}

int is_square_threatened(const GameState *state, const int square) {
    MoveList enemy_moves;
    generate_moves(state, &enemy_moves);
    
    for (int i = 0; i < enemy_moves.count; i++) {
        const Move *m = &enemy_moves.moves[i];
        if (m->length > 0) {
            for (int k = 0; k < m->length; k++) {
                if (m->captured_squares[k] == square) return 1;
            }
        }
    }
    return 0;
}
