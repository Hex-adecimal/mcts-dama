#include <stdio.h>
#include <stdlib.h>
#include "game.h"
#include "movegen.h"
#include "cnn.h"

#include "dataset.h"

// --- DATASET STATISTICS ---
void analyze_dataset_stats() {
    const char *filename = "data/train.bin";
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("\n[Stats] No training dataset found at %s.\n", filename);
        return;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Read and skip header
    DatasetHeader header;
    if (fread(&header, sizeof(DatasetHeader), 1, f) != 1) {
        printf("\n[Stats] Error reading header from %s.\n", filename);
        fclose(f);
        return;
    }
    
    int count = header.num_samples;
    
    // We don't load everything to RAM to save memory, just iterate nicely?
    // Actually 100k samples is ~100MB. Can load chunks.
    TrainingSample buffer[1000];
    int wins = 0, losses = 0, draws = 0;
    
    int processed = 0;
    while (processed < count) {
        int to_read = (count - processed > 1000) ? 1000 : (count - processed);
        fread(buffer, sizeof(TrainingSample), to_read, f);
        
        for (int i = 0; i < to_read; i++) {
            if (buffer[i].target_value > 0.1f) wins++;
            else if (buffer[i].target_value < -0.1f) losses++;
            else draws++;
        }
        processed += to_read;
    }
    
    fclose(f);
    
    printf("\n=== Training Dataset Statistics ===\n");
    printf("Total Samples: %d\n", count);
    printf("  Wins   (> 0.1): %d (%.1f%%)\n", wins, 100.0f * wins / count);
    printf("  Losses (< -0.1): %d (%.1f%%)\n", losses, 100.0f * losses / count);
    printf("  Draws  (Near 0): %d (%.1f%%)\n", draws, 100.0f * draws / count);
    
    float balance = (float)wins / (wins + losses + draws);
    if (draws > count * 0.8) {
        printf("  [WARNING] High Draw Rate! Value Head might learn 0.0 constant.\n");
    } else if (wins > count * 0.95 || losses > count * 0.95) {
        printf("  [WARNING] Unbalanced Dataset! (One side always wins?)\n");
    } else {
        printf("  [OK] Dataset looks balanced.\n");
    }
}
void print_bitboard(Bitboard b) {
    for (int r = 7; r >= 0; r--) {
        for (int c = 0; c < 8; c++) {
            if ((b >> (r*8 + c)) & 1) printf("X "); else printf(". ");
        }
        printf("\n");
    }
}

void print_features(float *features) {
    printf("\n--- CNN Input Features (Channel 0: My Pawns) ---\n");
    // Channel 0 is first 32 floats (only dark squares)
    // We map back to 8x8 to visualize
    int idx = 0;
    for (int r = 7; r >= 0; r--) {
        for (int c = 0; c < 8; c++) {
             if ((r+c)%2==1) {
                 printf("%.0f ", features[idx]);
                 idx++;
             } else {
                 printf(". ");
             }
        }
        printf("\n");
    }
    printf("(If Canonical is working, BLACK pieces should appear at BOTTOM for Black)\n");
}

int main() {
    printf("=== CNN DEBUGGER ===\n");
    
    // Analyze dataset first
    analyze_dataset_stats();
    printf("\n------------------------------------------------\n");
    
    init_move_tables();
    zobrist_init();

    CNNWeights weights;
    cnn_init(&weights);
    if (cnn_load_weights(&weights, "bin/cnn_weights.bin") != 0) {
        printf("Failed to load weights!\n");
        return 1;
    }
    printf("Weights loaded.\n");
    
    GameState state;
    init_game(&state); // Standard start
    
    CNNOutput out;

    // --- TEST: CUSTOM SCENARIO (Forced Capture) ---
    printf("\n\n=== CUSTOM SCENARIO: WHITE To Move (C3 Capture D4) ===\n");
    
    // Clear Board
    state.white_pieces = 0;
    state.white_ladies = 0;
    state.black_pieces = 0;
    state.black_ladies = 0;
    state.current_player = WHITE;
    
    // Place Pieces
    // C3 = Index 18 (Row 2, Col 2). (2*8 + 2)
    // D4 = Index 27 (Row 3, Col 3). (3*8 + 3)
    state.white_pieces |= (1ULL << 18);
    state.black_pieces |= (1ULL << 27);
    
    print_board(&state);
    
    float features[CNN_BOARD_SIZE * CNN_BOARD_SIZE * CNN_INPUT_CHANNELS];
    float player_val;
    cnn_encode_state(&state, features, &player_val);
    print_features(features);
    
    cnn_forward(&weights, &state, &out);
    printf("\n>>> Value Head: %.4f (Should be High if it sees the capture)\n", out.value);
    
    // --- RAW NETWORK OUTPUT ANALYSIS ---
    printf("\n>>> RAW Network Output (Top 5 Probabilities anywhere):\n");
    
    // Simple sort to find top 5
    typedef struct { int idx; float p; } ProbEntry;
    ProbEntry entries[CNN_POLICY_SIZE];
    for(int i=0; i<CNN_POLICY_SIZE; i++) {
        entries[i].idx = i;
        entries[i].p = out.policy[i];
    }
    
    // Bubble sort top 5 (lazy)
    for(int i=0; i<5; i++) {
        for(int j=i+1; j<CNN_POLICY_SIZE; j++) {
            if(entries[j].p > entries[i].p) {
                ProbEntry tmp = entries[i];
                entries[i] = entries[j];
                entries[j] = tmp;
            }
        }
    }
    
    for(int i=0; i<5; i++) {
        int idx = entries[i].idx;
        float p = entries[i].p;
        printf("  #%d Index %d: Prob %.4f\n", i+1, idx, p);
    }
    printf("  (Total Sum: %.4f)\n", 1.0); // Softmax sums to 1

    MoveList moves;
    generate_moves(&state, &moves);
    printf("\nLegal Moves for White (Masked & Normalized):\n");
    
    // 1. Calculate Sum of Legal Probabilities
    float sum_legal = 0.0f;
    for (int i=0; i<moves.count; i++) {
        int idx = cnn_move_to_index(&moves.moves[i], state.current_player);
        sum_legal += out.policy[idx];
    }
    printf("  Sum of Legal Probs: %.6f (If near 0, network hates legal moves)\n", sum_legal);
    
    // 2. Print Normalized
    for (int i=0; i<moves.count; i++) {
        int idx = cnn_move_to_index(&moves.moves[i], state.current_player);
        float raw_prob = out.policy[idx];
        float norm_prob = (sum_legal > 1e-9) ? (raw_prob / sum_legal) : 0.0f;
        
        int from = moves.moves[i].path[0];
        // Capture dest is path[length]
        int to = (moves.moves[i].length > 0) ? moves.moves[i].path[moves.moves[i].length] : moves.moves[i].path[1];
        
        printf("Move %d->%d (Idx %d): Raw %.6f | Norm %.4f ", from, to, idx, raw_prob, norm_prob);
        if (moves.moves[i].length > 0) printf("[CAPTURE]");
        printf("\n");
        
        // Visualize the move index direction
        // from=18. to=36 (E5). Diff=18. Direction?
        // idx = from*4 + dir.
        // Let's verify what index it *should* be.
        // 18 / 2 = 9 (Packed). 
        // No, cnn_move_to_index uses sparse 0-63 if we fixed it? 
        // Or 32 packed?
        // I'll trust the output for now.
    }
    
    cnn_free(&weights);
    return 0;
}
