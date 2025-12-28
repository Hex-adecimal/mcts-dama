/**
 * init_weights.c
 * 
 * Helper tool to initialize fresh CNN weights and save them to 'models/cnn_weights.bin'.
 * Useful when weights are lost or to start from scratch.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"

int main(int argc, char **argv) {
    printf("=== CNN Weights Initializer ===\n");
    
    // Seed random
    srand((unsigned)time(NULL));
    
    CNNWeights weights;
    printf("Initializing random weights...\n");
    cnn_init(&weights); // Allocates and randomizes
    
    const char *path = "models/cnn_weights.bin";
    if (argc > 1) {
        path = argv[1];
    }
    
    printf("Saving to '%s'...\n", path);
    cnn_save_weights(&weights, path);
    printf("✓ Success! Weights saved.\n");
    
    cnn_free(&weights);
    return 0;
}
