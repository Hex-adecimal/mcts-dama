/**
 * print_game.c - Print a complete decisive game from training data
 * 
 * Finds and displays a non-draw game from the training dataset.
 */

#include "game.h"
#include "movegen.h"
#include "dataset.h"

#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *filename = (argc > 1) ? argv[1] : "data/train_gen1.bin";
    
    printf("=== Game Viewer from Training Data ===\n");
    printf("Loading: %s\n\n", filename);
    
    zobrist_init();
    init_move_tables();
    
    // Load samples
    int count = dataset_get_count(filename);
    if (count <= 0) {
        printf("ERROR: Could not read dataset\n");
        return 1;
    }
    
    printf("Total samples: %d\n\n", count);
    
    // Allocate and load
    int max_load = (count > 50000) ? 50000 : count;
    TrainingSample *samples = malloc(max_load * sizeof(TrainingSample));
    dataset_load(filename, samples, max_load);
    
    // Find a decisive game (Z != 0)
    // Strategy: Find a sample with |Z| > 0.5, then find the start of that game
    // Games start at initial position (12 pieces each side)
    
    int game_start = -1;
    int game_end = -1;
    float game_result = 0;
    
    for (int i = 0; i < max_load; i++) {
        float z = samples[i].target_value;
        
        // Check for decisive game (non-draw)
        if (z > 0.5f || z < -0.5f) {
            // Found a decisive sample, now backtrack to find game start
            game_result = z;
            game_end = i;
            
            // Backtrack while Z has same sign (same game)
            int j = i;
            while (j > 0) {
                float prev_z = samples[j-1].target_value;
                // If Z changes sign or becomes 0, we crossed into previous game
                if ((game_result > 0 && prev_z <= 0) || (game_result < 0 && prev_z >= 0)) {
                    break;
                }
                j--;
            }
            game_start = j;
            
            // Find true end of this game
            while (game_end + 1 < max_load) {
                float next_z = samples[game_end + 1].target_value;
                if ((game_result > 0 && next_z <= 0) || (game_result < 0 && next_z >= 0)) {
                    break;
                }
                game_end++;
            }
            
            break;
        }
    }
    
    if (game_start < 0) {
        printf("No decisive game found in first %d samples.\n", max_load);
        free(samples);
        return 0;
    }
    
    int game_length = game_end - game_start + 1;
    printf("Found Decisive Game!\n");
    printf("  Samples: %d to %d (%d moves)\n", game_start, game_end, game_length);
    printf("  Result: %.2f (%s wins)\n\n", game_result, 
           game_result > 0 ? "WHITE" : "BLACK");
    
    // Print the game
    printf("=== GAME REPLAY ===\n\n");
    
    for (int i = game_start; i <= game_end && i < game_start + 50; i++) { // Limit to 50 moves
        TrainingSample *s = &samples[i];
        
        int move_num = i - game_start + 1;
        const char *player = (s->state.current_player == WHITE) ? "WHITE" : "BLACK";
        
        printf("--- Move %d (%s to play) ---\n", move_num, player);
        printf("Z = %.2f (from %s's perspective)\n", s->target_value, player);
        
        // Print board
        print_board(&s->state);
        
        // Find top policy move
        float max_prob = 0;
        int max_idx = 0;
        for (int k = 0; k < CNN_POLICY_SIZE; k++) {
            if (s->target_policy[k] > max_prob) {
                max_prob = s->target_policy[k];
                max_idx = k;
            }
        }
        
        // Decode move index (from = idx/8, dir = idx%8)
        int from_sq = max_idx / 8;
        int dir = max_idx % 8;
        int is_capture = (dir >= 4);
        if (is_capture) dir -= 4;
        
        const char *dir_names[] = {"NE", "NW", "SE", "SW"};
        printf("Top Policy: sq %d -> %s%s (%.1f%%)\n", 
               from_sq, dir_names[dir], is_capture ? " (capture)" : "",
               max_prob * 100);
        
        printf("\n");
    }
    
    if (game_length > 50) {
        printf("... (game continues for %d more moves)\n", game_length - 50);
    }
    
    printf("=== GAME END (Result: %s) ===\n", game_result > 0 ? "WHITE wins" : "BLACK wins");
    
    free(samples);
    return 0;
}
