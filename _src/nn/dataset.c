/**
 * dataset.c - Dataset I/O for Neural Network Training
 */

#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int dataset_save(const char *filename, const TrainingSample *samples, size_t count) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "[Dataset] Error: Cannot open %s for writing\n", filename);
        return -1;
    }
    
    // Write header
    DatasetHeader header;
    memcpy(header.magic, DATASET_MAGIC, 4);
    header.version = DATASET_VERSION;
    header.num_samples = (uint32_t)count;
    
    if (fwrite(&header, sizeof(DatasetHeader), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    
    // Write samples
    if (fwrite(samples, sizeof(TrainingSample), count, f) != count) {
        fclose(f);
        return -1;
    }
    
    fclose(f);
    printf("[Dataset] Saved %zu samples to %s\n", count, filename);
    return 0;
}

TrainingSample* dataset_load_alloc(const char *filename, int *out_count) {
    *out_count = 0;
    
    int count = dataset_get_count(filename);
    if (count <= 0) {
        fprintf(stderr, "[Dataset] Error: Cannot get count from %s\n", filename);
        return NULL;
    }
    
    TrainingSample *data = malloc(count * sizeof(TrainingSample));
    if (!data) {
        fprintf(stderr, "[Dataset] Error: Out of memory\n");
        return NULL;
    }
    
    int loaded = dataset_load(filename, data, count);
    if (loaded <= 0) {
        free(data);
        return NULL;
    }
    
    *out_count = loaded;
    return data;
}

int dataset_save_append(const char *filename, const TrainingSample *samples, size_t count) {
    FILE *f = fopen(filename, "r+b");
    if (!f) {
        // File doesn't exist? Create new.
        f = fopen(filename, "wb");
        if (!f) return -1;
        
        // Write fresh header
        DatasetHeader header;
        memcpy(header.magic, DATASET_MAGIC, 4);
        header.version = DATASET_VERSION;
        header.num_samples = 0;
        fwrite(&header, sizeof(DatasetHeader), 1, f);
        
        // Reopen in update mode to allow seeking if needed later (or just continue writing)
        // Actually w+b allows reading too. 
        // But let's close and strictly follow r+b flow or just use the handle handles.
        // For 'wb', we are at 0. Wrote header. Now at sizeof(header).
        // We can just write samples.
        fwrite(samples, sizeof(TrainingSample), count, f);
        
        // Update header count in memory and rewrite (easy way)
        header.num_samples = (uint32_t)count;
        fseek(f, 0, SEEK_SET);
        fwrite(&header, sizeof(DatasetHeader), 1, f);
        fclose(f);
        return 0;
    }
    
    // File exists, read header
    DatasetHeader header;
    if (fread(&header, sizeof(DatasetHeader), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    
    // Validate
    if (memcmp(header.magic, DATASET_MAGIC, 4) != 0 || header.version != DATASET_VERSION) {
        fclose(f);
        return -1;
    }
    
    // Seek to end
    fseek(f, 0, SEEK_END);
    fwrite(samples, sizeof(TrainingSample), count, f);
    
    // Update header
    header.num_samples += (uint32_t)count;
    fseek(f, 0, SEEK_SET);
    fwrite(&header, sizeof(DatasetHeader), 1, f);
    
    fclose(f);
    return 0;
}

int dataset_load(const char *filename, TrainingSample *samples, size_t max_samples) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "[Dataset] Error: Cannot open %s for reading\n", filename);
        return -1;
    }
    
    // Read header
    DatasetHeader header;
    if (fread(&header, sizeof(DatasetHeader), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    
    // Validate magic
    if (memcmp(header.magic, DATASET_MAGIC, 4) != 0) {
        fprintf(stderr, "[Dataset] Error: Invalid file format\n");
        fclose(f);
        return -1;
    }
    
    // Validate version
    if (header.version != DATASET_VERSION) {
        fprintf(stderr, "[Dataset] Error: Version mismatch (%u vs %u)\n", 
                header.version, DATASET_VERSION);
        fclose(f);
        return -1;
    }
    
    // Load samples
    size_t to_load = (header.num_samples < max_samples) ? header.num_samples : max_samples;
    size_t loaded = fread(samples, sizeof(TrainingSample), to_load, f);
    
    fclose(f);
    printf("[Dataset] Loaded %zu samples from %s\n", loaded, filename);
    return (int)loaded;
}

int dataset_get_count(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return -1;
    
    DatasetHeader header;
    if (fread(&header, sizeof(DatasetHeader), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    
    fclose(f);
    
    if (memcmp(header.magic, DATASET_MAGIC, 4) != 0) return -1;
    
    return (int)header.num_samples;
}

void dataset_shuffle(TrainingSample *samples, size_t count) {
    for (size_t i = count - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        TrainingSample temp = samples[i];
        samples[i] = samples[j];
        samples[j] = temp;
    }
}

void dataset_split(TrainingSample *samples, size_t count, float train_ratio,
                   TrainingSample **train_out, TrainingSample **val_out,
                   size_t *train_count, size_t *val_count) {
    
    *train_count = (size_t)(count * train_ratio);
    *val_count = count - *train_count;
    
    *train_out = samples;
    *val_out = samples + *train_count;
}
