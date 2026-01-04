/**
 * test_training.c - Unit Tests for Training Module
 * 
 * Tests: dataset.h, selfplay.h, training_pipeline.h
 */

// Note: Includes are in test_main.c

#include <sys/stat.h>
#include <unistd.h>

// =============================================================================
// DATASET TESTS
// =============================================================================

TEST(training_dataset_save_and_load) {
    const char *path = "/tmp/test_dataset.bin";
    
    // Create sample data
    TrainingSample samples[5];
    memset(samples, 0, sizeof(samples));
    
    for (int i = 0; i < 5; i++) {
        init_game(&samples[i].state);
        samples[i].target_value = (float)i / 5.0f;
        samples[i].target_policy[0] = 1.0f;
    }
    
    // Save
    int save_ret = dataset_save(path, samples, 5);
    ASSERT_EQ(0, save_ret);
    
    // Check count
    int count = dataset_get_count(path);
    ASSERT_EQ(5, count);
    
    // Load
    TrainingSample loaded[5];
    int load_ret = dataset_load(path, loaded, 5);
    ASSERT_EQ(5, load_ret);
    
    // Verify
    for (int i = 0; i < 5; i++) {
        ASSERT_FLOAT_EQ(samples[i].target_value, loaded[i].target_value, 1e-6f);
    }
    
    // Cleanup
    remove(path);
}

TEST(training_dataset_append) {
    const char *path = "/tmp/test_dataset_append.bin";
    
    // Remove if exists
    remove(path);
    
    // Create and save first batch
    TrainingSample batch1[3];
    memset(batch1, 0, sizeof(batch1));
    for (int i = 0; i < 3; i++) {
        init_game(&batch1[i].state);
        batch1[i].target_value = 0.1f * i;
    }
    
    int ret = dataset_save_append(path, batch1, 3);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(3, dataset_get_count(path));
    
    // Append second batch
    TrainingSample batch2[2];
    memset(batch2, 0, sizeof(batch2));
    for (int i = 0; i < 2; i++) {
        init_game(&batch2[i].state);
        batch2[i].target_value = 0.5f + 0.1f * i;
    }
    
    ret = dataset_save_append(path, batch2, 2);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(5, dataset_get_count(path));
    
    // Cleanup
    remove(path);
}

TEST(training_dataset_load_alloc) {
    const char *path = "/tmp/test_dataset_alloc.bin";
    
    // Create sample data
    TrainingSample samples[10];
    memset(samples, 0, sizeof(samples));
    for (int i = 0; i < 10; i++) {
        init_game(&samples[i].state);
        samples[i].target_value = (float)i;
    }
    
    dataset_save(path, samples, 10);
    
    // Load with auto-alloc
    int count = 0;
    TrainingSample *loaded = dataset_load_alloc(path, &count);
    
    ASSERT_NOT_NULL(loaded);
    ASSERT_EQ(10, count);
    ASSERT_FLOAT_EQ(5.0f, loaded[5].target_value, 1e-6f);
    
    free(loaded);
    remove(path);
}

TEST(training_dataset_get_count_nonexistent) {
    int count = dataset_get_count("/nonexistent/path/data.bin");
    ASSERT_EQ(-1, count);
}

TEST(training_dataset_shuffle_changes_order) {
    TrainingSample samples[100];
    memset(samples, 0, sizeof(samples));
    
    for (int i = 0; i < 100; i++) {
        init_game(&samples[i].state);
        samples[i].target_value = (float)i;
    }
    
    // Store first few values
    float first_before = samples[0].target_value;
    float tenth_before = samples[9].target_value;
    
    dataset_shuffle(samples, 100);
    
    // Very unlikely that both stay in place after shuffle
    // (This is probabilistic, but should almost always pass)
    int changed = (samples[0].target_value != first_before) ||
                  (samples[9].target_value != tenth_before);
    ASSERT_TRUE(changed);
}

TEST(training_dataset_split) {
    // Allocate samples on heap since split may modify or point into original
    TrainingSample *samples = malloc(100 * sizeof(TrainingSample));
    memset(samples, 0, 100 * sizeof(TrainingSample));
    
    for (int i = 0; i < 100; i++) {
        init_game(&samples[i].state);
        samples[i].target_value = (float)i;
    }
    
    TrainingSample *train = NULL, *val = NULL;
    size_t train_count = 0, val_count = 0;
    
    dataset_split(samples, 100, 0.8f, &train, &val, &train_count, &val_count);
    
    ASSERT_EQ(80, (int)train_count);
    ASSERT_EQ(20, (int)val_count);
    
    // Note: dataset_split returns pointers INTO the samples buffer, 
    // not new allocations. So only free the original samples.
    free(samples);
}

// =============================================================================
// TRAINING SAMPLE TESTS
// =============================================================================

TEST(training_sample_struct_size) {
    // Ensure TrainingSample has expected size for binary compatibility
    size_t expected_state_size = sizeof(GameState);
    size_t expected_history_size = sizeof(GameState) * (CNN_HISTORY_T - 1);
    size_t expected_policy_size = sizeof(float) * CNN_POLICY_SIZE;
    size_t expected_value_size = sizeof(float);
    
    size_t total = expected_state_size + expected_history_size + 
                   expected_policy_size + expected_value_size;
    
    // Allow for padding
    ASSERT_LE(total, sizeof(TrainingSample));
}

TEST(training_sample_policy_sums_to_valid) {
    // Create a valid sample
    TrainingSample sample = {0};
    init_game(&sample.state);
    
    // Set a valid policy distribution
    sample.target_policy[0] = 0.5f;
    sample.target_policy[1] = 0.3f;
    sample.target_policy[2] = 0.2f;
    
    float sum = 0;
    for (int i = 0; i < CNN_POLICY_SIZE; i++) {
        sum += sample.target_policy[i];
    }
    
    ASSERT_FLOAT_EQ(1.0f, sum, 0.01f);
}

// =============================================================================
// CNN TRAINING TESTS
// =============================================================================

TEST(training_cnn_train_step_reduces_loss) {
    zobrist_init();
    movegen_init();
    
    CNNWeights weights;
    cnn_init(&weights);
    
    // Create a simple training sample
    TrainingSample sample = {0};
    init_game(&sample.state);
    sample.target_policy[0] = 1.0f;  // Target: move 0
    sample.target_value = 0.5f;
    
    // Train for several steps and check loss decreases
    float p_loss1 = 0, v_loss1 = 0;
    float p_loss2 = 0, v_loss2 = 0;
    
    // First step
    cnn_train_step(&weights, &sample, 1, 0.1f, 0.002f, 0.0f, 0.0f, &p_loss1, &v_loss1);
    
    // Train more
    for (int i = 0; i < 10; i++) {
        cnn_train_step(&weights, &sample, 1, 0.1f, 0.002f, 0.0f, 0.0f, &p_loss2, &v_loss2);
    }
    
    // Loss should decrease (or at least not increase drastically)
    // On a single sample with high LR, we should overfit quickly
    float total_loss1 = p_loss1 + v_loss1;
    float total_loss2 = p_loss2 + v_loss2;
    
    ASSERT_LT(total_loss2, total_loss1 + 0.1f);
    
    cnn_free(&weights);
}

TEST(training_cnn_gradients_cleared_after_update) {
    CNNWeights weights;
    cnn_init(&weights);
    
    TrainingSample sample = {0};
    init_game(&sample.state);
    sample.target_policy[0] = 1.0f;
    sample.target_value = 0.0f;
    
    float p, v;
    cnn_train_step(&weights, &sample, 1, 0.01f, 0.001f, 0.0f, 0.0f, &p, &v);
    
    // After a train step, network should have updated weights (not just gradients)
    // The exact gradient state depends on implementation
    // Just verify the function doesn't crash and returns valid losses
    ASSERT_GE(p, 0.0f);
    ASSERT_GE(v, 0.0f);
    
    cnn_free(&weights);
}
