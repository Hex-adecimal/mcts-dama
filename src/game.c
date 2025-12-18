#include "game.h"
#include <string.h>

#define MAX_CHAIN_LENGTH 12

// =============================================================================
// CONSTANTS (Direction Tables for Capture Generation)
// =============================================================================

// All 4 diagonal directions: NE(+9), NW(+7), SE(-7), SW(-9)
static const int ALL_DIRS[4] = { 9, 7, -7, -9 };

// Masks for jumps (2 steps in each direction)
static const uint64_t JUMP_MASKS[4] = { NOT_FILE_GH, NOT_FILE_AB, NOT_FILE_GH, NOT_FILE_AB };

// =============================================================================
// MOVE LOOKUP TABLES (Pre-computed move targets)
// =============================================================================
// [color][from_square][direction] -> target square bitboard

static Bitboard PAWN_MOVE_TARGETS[2][64][2];   // Pawn simple moves
static Bitboard LADY_MOVE_TARGETS[64][4];      // Lady simple moves (1 step)
static Bitboard JUMP_LANDING[64][4];           // Landing square after jump
static Bitboard JUMP_OVER_SQ[64][4];           // Captured square in jump
static int move_tables_initialized = 0;

void init_move_tables(void) {
    if (move_tables_initialized) return;
    
    for (int sq = 0; sq < 64; sq++) {
        int row = sq / 8;
        int col = sq % 8;
        
        // Clear all entries for this square
        for (int d = 0; d < 4; d++) {
            LADY_MOVE_TARGETS[sq][d] = 0;
            JUMP_LANDING[sq][d] = 0;
            JUMP_OVER_SQ[sq][d] = 0;
        }
        PAWN_MOVE_TARGETS[WHITE][sq][0] = 0;
        PAWN_MOVE_TARGETS[WHITE][sq][1] = 0;
        PAWN_MOVE_TARGETS[BLACK][sq][0] = 0;
        PAWN_MOVE_TARGETS[BLACK][sq][1] = 0;
        
        // WHITE pawn moves: +9 (NE), +7 (NW)
        if (row < 7) {
            if (col < 7) PAWN_MOVE_TARGETS[WHITE][sq][0] = 1ULL << (sq + 9);  // NE
            if (col > 0) PAWN_MOVE_TARGETS[WHITE][sq][1] = 1ULL << (sq + 7);  // NW
        }
        
        // BLACK pawn moves: -7 (SE), -9 (SW)
        if (row > 0) {
            if (col < 7) PAWN_MOVE_TARGETS[BLACK][sq][0] = 1ULL << (sq - 7);  // SE
            if (col > 0) PAWN_MOVE_TARGETS[BLACK][sq][1] = 1ULL << (sq - 9);  // SW
        }
        
        // Lady moves (all 4 directions, 1 step)
        // Dir 0: NE (+9), Dir 1: NW (+7), Dir 2: SE (-7), Dir 3: SW (-9)
        if (row < 7 && col < 7) LADY_MOVE_TARGETS[sq][0] = 1ULL << (sq + 9);
        if (row < 7 && col > 0) LADY_MOVE_TARGETS[sq][1] = 1ULL << (sq + 7);
        if (row > 0 && col < 7) LADY_MOVE_TARGETS[sq][2] = 1ULL << (sq - 7);
        if (row > 0 && col > 0) LADY_MOVE_TARGETS[sq][3] = 1ULL << (sq - 9);
        
        // Jump targets (2 squares in each direction)
        // Dir 0: NE (+18), over (+9)
        if (row < 6 && col < 6) {
            JUMP_LANDING[sq][0] = 1ULL << (sq + 18);
            JUMP_OVER_SQ[sq][0] = 1ULL << (sq + 9);
        }
        // Dir 1: NW (+14), over (+7)
        if (row < 6 && col > 1) {
            JUMP_LANDING[sq][1] = 1ULL << (sq + 14);
            JUMP_OVER_SQ[sq][1] = 1ULL << (sq + 7);
        }
        // Dir 2: SE (-14), over (-7)
        if (row > 1 && col < 6) {
            JUMP_LANDING[sq][2] = 1ULL << (sq - 14);
            JUMP_OVER_SQ[sq][2] = 1ULL << (sq - 7);
        }
        // Dir 3: SW (-18), over (-9)
        if (row > 1 && col > 1) {
            JUMP_LANDING[sq][3] = 1ULL << (sq - 18);
            JUMP_OVER_SQ[sq][3] = 1ULL << (sq - 9);
        }
    }
    
    move_tables_initialized = 1;
}

// =============================================================================
// ZOBRIST HASHING
// =============================================================================

// [Color][PieceType][Square]
// Color: 0=White, 1=Black
// PieceType: 0=Pawn, 1=Lady
uint64_t zobrist_keys[2][2][64];
uint64_t zobrist_black_move;

// Simple PRNG for reproducible runs
static uint64_t rand64() {
    static uint64_t seed = 0x987654321ULL;
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return seed;
}

void zobrist_init() {
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
    // This ensures identical positions with different turns have different hashes.
    // The key is toggled on/off each move via XOR in apply_move().
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
    // We need to know what was at 'from' (Pawn or Lady)
    int is_lady = move->is_lady_move; // Trusted from move generation
    
    state->hash ^= zobrist_keys[us][is_lady][from];

    // Move the piece (and handle promotion)
    perform_movement(state, from, to, us);

    // --- ZOBRIST: Add piece at destination ---
    // Check if it promoted. perform_movement handles the bitboard change.
    // We check the destination bitboard to see what it is now.
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
// MOVE GENERATION
// =============================================================================

/**
 * Helper to add a simple move to the move list.
 * @param list Pointer to the MoveList to add the move to.
 * @param from Source square index.
 * @param to Destination square index.
 * @param is_lady Boolean flag: 1 if the moving piece is a Lady, 0 if a Pawn.
 */
static void add_simple_move(MoveList *list, int from, int to, int is_lady) {
    if (list->count >= MAX_MOVES) return;
    Move *m = &list->moves[list->count++];
    m->path[0] = from;
    m->path[1] = to;
    m->length = 0; // 0 indicates simple move
    m->captured_ladies_count = 0;
    m->is_lady_move = is_lady;
    m->first_captured_is_lady = 0;
}


// Helper to save a capture move and compute priority metrics
static void save_move(MoveList *list, const GameState *s, 
                             int path[], int captured[], int depth, int is_lady) {
    if (list->count >= MAX_MOVES) return;
    if (depth > 11) depth = 11; // Safety cap
    
    Move *m = &list->moves[list->count++];
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

    // Compute priority metrics (Quality)
    for(int k = 0; k < depth; k++) {
        // Check if captured piece was a Lady
        if ((s->white_ladies | s->black_ladies) & (1ULL << captured[k])) {
            m->captured_ladies_count++;
            if (k == 0) m->first_captured_is_lady = 1;
        }
    }
}

/**
 * Recursive function to find capture chains.
 * Explores all possible jump sequences using DFS and Bitwise checks.
 * @param s Pointer to the current GameState.
 * @param current_sq The square index where the piece currently is.
 * @param path Array storing the sequence of squares visited.
 * @param captured Array storing the sequence of captured squares.
 * @param depth Current recursion depth (number of jumps made).
 * @param is_lady Boolean flag: 1 if moving a Lady, 0 if Pawn.
 * @param enemy_pieces Bitboard of enemy pieces (updated to prevent re-capture).
 * @param enemy_ladies Bitboard of enemy ladies (updated to prevent re-capture).
 * @param occupied Bitboard of all occupied squares (updated to allow landing on captured squares).
 * @param list Pointer to the MoveList to populate.
 */
static void find_captures(const GameState *s, int current_sq, 
                          int path[], int captured[], int depth, 
                          int is_lady, Bitboard enemy_pieces, Bitboard enemy_ladies, 
                          Bitboard occupied, MoveList *list) {
    
    int found_continuation = 0;
    int us = s->current_player;

    // Loop ranges based on piece type and color
    // White Pawn: 0-2 (+9, +7)
    // Black Pawn: 2-4 (-7, -9)
    // Lady: 0-4 (All)
    int start_dir = is_lady ? 0 : (us == WHITE ? 0 : 2);
    int end_dir   = is_lady ? 4 : (us == WHITE ? 2 : 4);

    for (int i = start_dir; i < end_dir; i++) {
        // BITWISE BOUNDARY CHECK
        // If current square is too close to the edge for a double jump, skip.
        if (!((1ULL << current_sq) & JUMP_MASKS[i])) continue;

        int step = ALL_DIRS[i];
        int jump = step * 2;
        
        int bridge_sq = current_sq + step;
        int land_sq   = current_sq + jump;

        // Safety vertical check (mostly for debug, bitmasks usually cover this)
        if (land_sq < 0 || land_sq > 63) continue;

        // CHECK CONTENT
        Bitboard bridge_mask = (1ULL << bridge_sq);
        Bitboard land_mask   = (1ULL << land_sq);

        // Bridge must contain enemy
        int is_enemy_p = (enemy_pieces & bridge_mask) != 0;
        int is_enemy_l = (enemy_ladies & bridge_mask) != 0;
        
        if (!is_enemy_p && !is_enemy_l) continue;
        
        // Rule: Pawn cannot capture Lady
        if (!is_lady && is_enemy_l) continue;
        
        // Landing must be empty (relative to current recursion state)
        if (occupied & land_mask) continue;

        // VALID CAPTURE -> RECURSE
        found_continuation = 1;
        
        path[depth + 1] = land_sq;
        captured[depth] = bridge_sq;

        // Check Promotion
        int promoted = (!is_lady && (land_mask & PROM_RANKS[us]));

        if (promoted) {
            // Promotion stops the chain immediately
            save_move(list, s, path, captured, depth + 1, is_lady);
        } else {
            // Continue recursion
            find_captures(s, land_sq, path, captured, depth + 1, is_lady, 
                          enemy_pieces & ~bridge_mask, // Update masks
                          enemy_ladies & ~bridge_mask, 
                          occupied & ~bridge_mask, 
                          list);
        }
    }

    // END OF CHAIN
    if (!found_continuation && depth > 0) {
        save_move(list, s, path, captured, depth, is_lady);
    }
}

/**
 * Helper to calculate a score for a move based on Italian Checkers priority rules.
 * The score encodes move properties to allow easy comparison for filtering.
 * @param m Pointer to the Move to score.
 * @return An integer representing the move's priority score.
 */
static int calculate_score(const Move *m) {
    // Score = (Length << 24) | (IsLady << 20) | (CapturedLadies << 10) | (FirstIsLady)
    // Assuming max length 12, max captured ladies 12.
    // Length: Bits 24-27 (Max 15)
    // IsLady: Bit 20 (0 or 1)
    // CapturedLadies: Bits 10-13 (Max 15)
    // FirstIsLady: Bit 0 (0 or 1)
    
    return (m->length << 24) | 
           (m->is_lady_move << 20) | 
           (m->captured_ladies_count << 10) | 
           m->first_captured_is_lady;
}

/**
 * Filters the list of moves based on Italian Checkers priority rules.
 * Rules (in order):
 * 1. Most pieces captured (Quantity).
 * 2. Lady captures > Pawn captures (Quality of piece moving).
 * 3. Most Ladies captured (Quality of captured pieces).
 * 4. First captured piece is a Lady.
 * * Uses a score-based approach to filter in 2 passes.
 * @param list Pointer to the MoveList to filter.
 */
static void filter_moves(MoveList *list) {
    if (list->count <= 1) return;
    
    int max_score = -1;
    
    // Pass 1: Find Max Score
    for (int i = 0; i < list->count; i++) {
        int score = calculate_score(&list->moves[i]);
        if (score > max_score) max_score = score;
    }
    
    // Pass 2: Filter
    int w = 0;
    for (int i = 0; i < list->count; i++) {
        if (calculate_score(&list->moves[i]) == max_score) {
            list->moves[w++] = list->moves[i];
        }
    }
    list->count = w;
}

/**
 * Generates all simple moves (non-captures) for the current player.
 * Uses pre-computed lookup tables for faster move generation.
 */
void generate_simple_moves(const GameState *s, MoveList *list) {
    int us = s->current_player;
    Bitboard own_pieces = (us == WHITE) ? s->white_pieces : s->black_pieces;
    Bitboard own_ladies = (us == WHITE) ? s->white_ladies : s->black_ladies;
    Bitboard empty      = get_empty_squares(s);

    // Pawns (using lookup tables)
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
        pawns &= (pawns - 1);
    }

    // Ladies (using lookup tables, all 4 directions)
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
 * Generates all capture moves (including chains) for the current player.
 * Uses recursive search to find all possible capture sequences.
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void generate_captures(const GameState *s, MoveList *list) {
    int us = s->current_player;
    int them = us ^ 1;
    
    Bitboard own_pieces = (us == WHITE) ? s->white_pieces : s->black_pieces;
    Bitboard own_ladies = (us == WHITE) ? s->white_ladies : s->black_ladies;
    Bitboard enemy_pieces = (them == WHITE) ? s->white_pieces : s->black_pieces;
    Bitboard enemy_ladies = (them == WHITE) ? s->white_ladies : s->black_ladies;
    
    // Calculate initial occupied board
    Bitboard occupied = get_all_occupied(s);
    
    // Helper to iterate pieces and find capture chains
    Bitboard pieces[2] = { own_pieces, own_ladies };
    int is_lady_flag[2] = { 0, 1 };
    
    for (int t = 0; t < 2; t++) {
        Bitboard bb = pieces[t];
        while (bb) {
            int sq = __builtin_ctzll(bb);
            int path[MAX_CHAIN_LENGTH + 1];
            int captured[MAX_CHAIN_LENGTH];
            path[0] = sq;
            find_captures(s, sq, path, captured, 0, is_lady_flag[t], 
                          enemy_pieces, enemy_ladies, occupied & ~(1ULL << sq), list);
            bb &= (bb - 1);
        }
    }
}

/**
 * Generates all legal moves for the current player.
 * Enforces Italian Checkers rules: mandatory captures and priority rules.
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void generate_moves(const GameState *s, MoveList *list) {
    list->count = 0;
    
    // Generate Captures
    generate_captures(s, list);
    
    // If captures exist, filter them and return
    if (list->count > 0) {
        filter_moves(list);
    } else {
        // Otherwise generate simple moves
        generate_simple_moves(s, list);
    }
}

/**
 * Checks if a specific square is under potential attack by the opponent in the NEXT turn.
 * Used for Safety Heuristics.
 * @param state The state AFTER our move.
 * @param square The square index to check.
 * @return 1 if attacked, 0 otherwise.
 */
int is_square_threatened(const GameState *state, int square) {
    // Generate all opponent moves from this state
    MoveList enemy_moves;
    // We can reuse generate_moves (expensive) or a lighter version.
    // For now, use generate_moves for correctness.
    generate_moves(state, &enemy_moves);
    
    for (int i = 0; i < enemy_moves.count; i++) {
        Move *m = &enemy_moves.moves[i];
        
        if (m->length > 0) {
            for (int k = 0; k < m->length; k++) {
                if (m->captured_squares[k] == square) return 1;
            }
        }
    }
    return 0;
}