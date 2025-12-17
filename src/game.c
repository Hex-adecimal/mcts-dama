#include "game.h"
#include <stdio.h>
#include <string.h> // For memcpy

#define MAX_CHAIN_LENGTH 12

// --- GLOBAL CONSTANTS ---

// Offsets for moving "Forward"
// White moves towards higher indices (+9, +7), Black towards lower (-7, -9)
static const int PAWN_DIRS[2][2] = {
    { 9, 7 },    // WHITE offsets (NE, NW)
    { -7, -9 }   // BLACK offsets (SE, SW)
};

// Masks required for these movements
// White NE (+9) and Black SE (-7) go RIGHT -> NO column H
// White NW (+7) and Black SW (-9) go LEFT -> NO column A
static const uint64_t PAWN_MASKS[2][2] = {
    { NOT_FILE_H, NOT_FILE_A }, // WHITE (for +9, for +7)
    { NOT_FILE_H, NOT_FILE_A }  // BLACK (for -7, for -9)
};

// --- ZOBRIST HASHING ---

// [Color][PieceType][Square]
// Color: 0=White, 1=Black
// PieceType: 0=Pawn, 1=Lady
static uint64_t zobrist_keys[2][2][64];
static uint64_t zobrist_black_move;

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

static uint64_t compute_full_hash(const GameState *state) {
    uint64_t hash = 0;
    
    // White Pieces
    Bitboard wp = state->white_pieces;
    while(wp) {
        int sq = __builtin_ctzll(wp);
        hash ^= zobrist_keys[WHITE][0][sq];
        wp &= (wp - 1);
    }
    // White Ladies
    Bitboard wl = state->white_ladies;
    while(wl) {
        int sq = __builtin_ctzll(wl);
        hash ^= zobrist_keys[WHITE][1][sq];
        wl &= (wl - 1);
    }
    // Black Pieces
    Bitboard bp = state->black_pieces;
    while(bp) {
        int sq = __builtin_ctzll(bp);
        hash ^= zobrist_keys[BLACK][0][sq];
        bp &= (bp - 1);
    }
    // Black Ladies
    Bitboard bl = state->black_ladies;
    while(bl) {
        int sq = __builtin_ctzll(bl);
        hash ^= zobrist_keys[BLACK][1][sq];
        bl &= (bl - 1);
    }
    
    if (state->current_player == BLACK) {
        hash ^= zobrist_black_move;
    }
    
    return hash;
}


// --- INITIALIZATION AND PRINTING ---

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

/**
 * Prints the current board state to the console.
 * Displays the board with coordinates, piece positions, and current turn info.
 * @param state Pointer to the GameState to display.
 */
void print_board(const GameState *state) {
    printf("\n   A B C D E F G H\n");
    printf("  +---------------+\n");

    // Iterate from top rank (7) to bottom rank (0) to print correctly
    for (int rank = 7; rank >= 0; rank--) {
        printf("%d |", rank + 1); // Print rank label (1-8)
        
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file; // Map rank/file to square index (0-63)
            char c = '.'; // Default: empty square

            // Check for piece presence on the current square
            if (check_bit(state->white_pieces, sq)) c = 'w';
            else if (check_bit(state->black_pieces, sq)) c = 'b';
            else if (check_bit(state->white_ladies, sq)) c = 'W';
            else if (check_bit(state->black_ladies, sq)) c = 'B';
            
            printf("%c|", c); // Print the character and a separator
        }
        printf(" %d\n", rank + 1);
    }
    printf("  +---------------+\n");
    printf("   A B C D E F G H\n\n");
    
    // Additional debug info
    printf("Turn: %s\n", (state->current_player == WHITE) ? "WHITE" : "BLACK");
    printf("White pieces bitboard: %llu\n", state->white_pieces);
}

/**
 * Prints the algebraic coordinates (e.g., "A1") of a square index.
 * @param square_idx The 0-63 index of the square.
 */
void print_coords(int square_idx) {
    int rank = (square_idx / 8) + 1; // Riga 1-8
    char file = (square_idx % 8) + 'A'; // Colonna A-H
    printf("%c%d", file, rank);
}

/**
 * Prints the list of generated moves to the console.
 * Useful for debugging and verifying move generation logic.
 * @param list Pointer to the MoveList containing moves to print.
 */
void print_move_list(MoveList *list) {

    printf("------------------------------------------------\n");
    printf("Found %d possible moves:\n", list->count);
    
    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        printf("%d) ", i + 1);
        
        if (m.length == 0) { // Simple move
            print_coords(m.path[0]);
            printf(" -> ");
            print_coords(m.path[1]);
        } else { // Capture chain
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

// --- CORE ENGINE ---

/**
 * Updates bitboards and switches the turn.
 * @param state Pointer to the current GameState.
 * @param from Source square index.
 * @param to Destination square index.
 */
// Helper privato: sposta e promuove
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

    // 1. Move the piece (and handle promotion)
    perform_movement(state, from, to, us);

    // --- ZOBRIST: Add piece at destination ---
    // Check if it promoted. perform_movement handles the bitboard change.
    // We check the destination bitboard to see what it is now.
    int now_lady = (us == WHITE) ? check_bit(state->white_ladies, to) : check_bit(state->black_ladies, to);
    state->hash ^= zobrist_keys[us][now_lady][to];

    // 2. Remove captured pieces (if any)
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

    // 3. Switch Turn
    state->current_player = (Color)(us ^ 1);
    state->hash ^= zobrist_black_move; // Toggle turn hash
}

// --- MOVE GENERATION ---

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


// Evita la duplicazione del codice di salvataggio e calcolo score
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

    // Calcolo metriche per la priorità (Quality)
    for(int k = 0; k < depth; k++) {
        // Controlliamo se il pezzo catturato era una dama (Bianca o Nera)
        if ((s->white_ladies | s->black_ladies) & (1ULL << captured[k])) {
            m->captured_ladies_count++;
            if (k == 0) m->first_captured_is_lady = 1;
        }
    }
}

/**
 * Recursive function to find capture chains.
 * Explores all possible jump sequences using DFS and Bitwise checks.
 * * @param s Pointer to the current GameState.
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

    // Directions: 0=NE(+9), 1=NW(+7), 2=SE(-7), 3=SW(-9)
    static const int DIRS[4] = { 9, 7, -7, -9 };
    // Masks to check if JUMP start is valid (e.g. to jump NE, cannot be in GH)
    static const uint64_t JUMP_MASKS[4] = { NOT_FILE_GH, NOT_FILE_AB, NOT_FILE_GH, NOT_FILE_AB };

    // Loop ranges based on piece type and color
    // White Pawn: 0-2 (+9, +7)
    // Black Pawn: 2-4 (-7, -9)
    // Lady: 0-4 (All)
    int start_dir = is_lady ? 0 : (us == WHITE ? 0 : 2);
    int end_dir   = is_lady ? 4 : (us == WHITE ? 2 : 4);

    for (int i = start_dir; i < end_dir; i++) {
        // 1. BITWISE BOUNDARY CHECK
        // If current square is too close to the edge for a double jump, skip.
        if (!((1ULL << current_sq) & JUMP_MASKS[i])) continue;

        int step = DIRS[i];
        int jump = step * 2;
        
        int bridge_sq = current_sq + step;
        int land_sq   = current_sq + jump;

        // Safety vertical check (mostly for debug, bitmasks usually cover this)
        if (land_sq < 0 || land_sq > 63) continue;

        // 2. CHECK CONTENT
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

        // 3. VALID CAPTURE -> RECURSE
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

    // 4. END OF CHAIN
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
 * @param s Pointer to the current GameState.
 * @param list Pointer to the MoveList to populate.
 */
void generate_simple_moves(const GameState *s, MoveList *list) {
    int us = s->current_player;
    Bitboard own_pieces = (us == WHITE) ? s->white_pieces : s->black_pieces;
    Bitboard own_ladies = (us == WHITE) ? s->white_ladies : s->black_ladies;
    Bitboard empty      = get_empty_squares(s);

    // Pawns
    for (int i = 0; i < 2; i++) {
        int offset = PAWN_DIRS[us][i];
        Bitboard mask = PAWN_MASKS[us][i];
        Bitboard movers = own_pieces & mask;
        Bitboard valid = shift_bitboard(movers, offset) & empty;
        
        while (valid) {
            int to = __builtin_ctzll(valid);
            int from = to - offset;
            add_simple_move(list, from, to, 0);
            valid &= (valid - 1);
        }
    }

    // Ladies
    int directions[4] = { 9, 7, -7, -9 };
    Bitboard dir_masks[4] = { NOT_FILE_H, NOT_FILE_A, NOT_FILE_H, NOT_FILE_A };

    for (int i = 0; i < 4; i++) {
        int offset = directions[i];
        Bitboard mask = dir_masks[i];
        Bitboard movers = own_ladies & mask;
        Bitboard valid = shift_bitboard(movers, offset) & empty;

        while (valid) {
            int to = __builtin_ctzll(valid);
            int from = to - offset;
            add_simple_move(list, from, to, 1);
            valid &= (valid - 1);
        }
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
    
    // Iterate all pieces and try to find chains
    Bitboard p = own_pieces;
    while (p) {
        int sq = __builtin_ctzll(p);
        int path[MAX_CHAIN_LENGTH + 1]; 
        int captured[MAX_CHAIN_LENGTH]; 
        path[0] = sq;
        // Pass occupied MINUS the starting square (it's floating)
        find_captures(s, sq, path, captured, 0, 0, enemy_pieces, enemy_ladies, occupied & ~(1ULL << sq), list);
        p &= (p - 1);
    }
    
    Bitboard l = own_ladies;
    while (l) {
        int sq = __builtin_ctzll(l);
        int path[MAX_CHAIN_LENGTH + 1];
        int captured[MAX_CHAIN_LENGTH];
        path[0] = sq;
        find_captures(s, sq, path, captured, 0, 1, enemy_pieces, enemy_ladies, occupied & ~(1ULL << sq), list);
        l &= (l - 1);
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
    
    // 1. Generate Captures
    generate_captures(s, list);
    
    // 2. If captures exist, filter them and return
    if (list->count > 0) {
        filter_moves(list);
    } else {
        // 3. Otherwise generate simple moves
        generate_simple_moves(s, list);
    }
}