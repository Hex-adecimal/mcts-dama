#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dataset.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <dataset.bin>\n", argv[0]);
        return 1;
    }

    int count_int = dataset_get_count(argv[1]);
    if (count_int < 0) {
        printf("Error getting sample count.\n");
        return 1;
    }
    size_t count = (size_t)count_int;
    
    TrainingSample *samples = malloc(count * sizeof(TrainingSample));
    if (!samples) {
        printf("Memory allocation failed for %zu samples\n", count);
        return 1;
    }
    
    int loaded = dataset_load(argv[1], samples, count);
    printf("Loaded %d samples. Scanning for NaN/Inf...\n", loaded);

    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t invalid_policy_sum = 0;

    for (size_t i = 0; i < (size_t)loaded; i++) {
        TrainingSample *s = &samples[i];

        // Check Value
        if (isnan(s->target_value)) {
            printf("Sample %zu has NaN value!\n", i);
            nan_count++;
        }
        if (isinf(s->target_value)) {
            printf("Sample %zu has Inf value!\n", i);
            inf_count++;
        }

        // Check Policy
        float p_sum = 0;
        int has_nan_policy = 0;
        for (int j = 0; j < CNN_POLICY_SIZE; j++) {
            if (isnan(s->target_policy[j])) {
                if (!has_nan_policy) printf("Sample %zu has NaN in policy at index %d\n", i, j);
                has_nan_policy = 1;
                nan_count++;
            }
            if (isinf(s->target_policy[j])) {
                printf("Sample %zu has Inf in policy at index %d\n", i, j);
                inf_count++;
            }
            p_sum += s->target_policy[j];
        }

        if (fabsf(p_sum - 1.0f) > 0.01f && p_sum > 0.001f) {
            // printf("Sample %zu has invalid policy sum: %f\n", i, p_sum);
            invalid_policy_sum++;
        }
        
        if (nan_count > 10) break;
    }

    printf("\nScan Complete:\n");
    printf("- NaN Issues: %zu\n", nan_count);
    printf("- Inf Issues: %zu\n", inf_count);
    printf("- Invalid Sums: %zu\n", invalid_policy_sum);

    free(samples);
    return 0;
}
