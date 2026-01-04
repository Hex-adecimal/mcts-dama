/**
 * cnn_io.c - CNN Weight I/O Operations
 * 
 * Contains: cnn_save_weights, cnn_load_weights
 */

#include "dama/neural/cnn.h"
#include <stdio.h>

// =============================================================================
// WEIGHT I/O
// =============================================================================

void cnn_save_weights(const CNNWeights *w, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(w->conv1_w, sizeof(float), 64*CNN_INPUT_CHANNELS*9, f); fwrite(w->conv1_b, sizeof(float), 64, f);
    fwrite(w->conv2_w, sizeof(float), 64*64*9, f); fwrite(w->conv2_b, sizeof(float), 64, f);
    fwrite(w->conv3_w, sizeof(float), 64*64*9, f); fwrite(w->conv3_b, sizeof(float), 64, f);
    fwrite(w->conv4_w, sizeof(float), 64*64*9, f); fwrite(w->conv4_b, sizeof(float), 64, f);
    fwrite(w->bn1_gamma, sizeof(float), 64, f); fwrite(w->bn1_beta, sizeof(float), 64, f);
    fwrite(w->bn2_gamma, sizeof(float), 64, f); fwrite(w->bn2_beta, sizeof(float), 64, f);
    fwrite(w->bn3_gamma, sizeof(float), 64, f); fwrite(w->bn3_beta, sizeof(float), 64, f);
    fwrite(w->bn4_gamma, sizeof(float), 64, f); fwrite(w->bn4_beta, sizeof(float), 64, f);
    fwrite(w->bn1_mean, sizeof(float), 64, f); fwrite(w->bn1_var, sizeof(float), 64, f);
    fwrite(w->bn2_mean, sizeof(float), 64, f); fwrite(w->bn2_var, sizeof(float), 64, f);
    fwrite(w->bn3_mean, sizeof(float), 64, f); fwrite(w->bn3_var, sizeof(float), 64, f);
    fwrite(w->bn4_mean, sizeof(float), 64, f); fwrite(w->bn4_var, sizeof(float), 64, f);
    fwrite(w->policy_w, sizeof(float), 512*4097, f); fwrite(w->policy_b, sizeof(float), 512, f);
    fwrite(w->value_w1, sizeof(float), 256*4097, f); fwrite(w->value_b1, sizeof(float), 256, f);
    fwrite(w->value_w2, sizeof(float), 1*256, f); fwrite(w->value_b2, sizeof(float), 1, f);
    fclose(f);
}

int cnn_load_weights(CNNWeights *w, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fread(w->conv1_w, sizeof(float), 64*CNN_INPUT_CHANNELS*9, f); fread(w->conv1_b, sizeof(float), 64, f);
    fread(w->conv2_w, sizeof(float), 64*64*9, f); fread(w->conv2_b, sizeof(float), 64, f);
    fread(w->conv3_w, sizeof(float), 64*64*9, f); fread(w->conv3_b, sizeof(float), 64, f);
    fread(w->conv4_w, sizeof(float), 64*64*9, f); fread(w->conv4_b, sizeof(float), 64, f);
    fread(w->bn1_gamma, sizeof(float), 64, f); fread(w->bn1_beta, sizeof(float), 64, f);
    fread(w->bn2_gamma, sizeof(float), 64, f); fread(w->bn2_beta, sizeof(float), 64, f);
    fread(w->bn3_gamma, sizeof(float), 64, f); fread(w->bn3_beta, sizeof(float), 64, f);
    fread(w->bn4_gamma, sizeof(float), 64, f); fread(w->bn4_beta, sizeof(float), 64, f);
    fread(w->bn1_mean, sizeof(float), 64, f); fread(w->bn1_var, sizeof(float), 64, f);
    fread(w->bn2_mean, sizeof(float), 64, f); fread(w->bn2_var, sizeof(float), 64, f);
    fread(w->bn3_mean, sizeof(float), 64, f); fread(w->bn3_var, sizeof(float), 64, f);
    fread(w->bn4_mean, sizeof(float), 64, f); fread(w->bn4_var, sizeof(float), 64, f);
    fread(w->policy_w, sizeof(float), 512*4097, f); fread(w->policy_b, sizeof(float), 512, f);
    fread(w->value_w1, sizeof(float), 256*4097, f); fread(w->value_b1, sizeof(float), 256, f);
    fread(w->value_w2, sizeof(float), 1*256, f); fread(w->value_b2, sizeof(float), 1, f);
    fclose(f);
    return 0;
}
