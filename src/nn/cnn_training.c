/**
 * cnn_training.c - Training Logic (Backprop, Gradients, SGD) for CNN
 */

#include "cnn.h"
#include "conv_ops.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <Accelerate/Accelerate.h>

// =============================================================================
// BATCH NORM BACKWARD
// =============================================================================

void batch_norm_backward(
    const float *d_output, const float *input,
    const float *gamma, const float *batch_mean, const float *batch_var,
    float *d_input, float *d_gamma, float *d_beta,
    int C, int H, int W
) {
    int S = H * W;
    #pragma omp parallel for
    for (int c = 0; c < C; c++) {
        float mean = batch_mean[c];
        float var = batch_var[c];
        float inv_std = 1.0f / sqrtf(var + CNN_BN_EPSILON);
        
        float d_gamma_c = 0, d_beta_c = 0;
        for (int i = 0; i < S; i++) {
            float x_hat = (input[c * S + i] - mean) * inv_std;
            d_gamma_c += d_output[c * S + i] * x_hat;
            d_beta_c += d_output[c * S + i];
        }
        
        // Atomic add for gamma/beta gradients (may be accumulated from multiple samples)
        #pragma omp atomic
        d_gamma[c] += d_gamma_c;
        #pragma omp atomic
        d_beta[c] += d_beta_c;

        float common = gamma[c] * inv_std / S;
        for (int i = 0; i < S; i++) {
            float x_hat = (input[c * S + i] - mean) * inv_std;
            d_input[c * S + i] = common * (S * d_output[c * S + i] - d_beta_c - x_hat * d_gamma_c);
        }
    }
}

// =============================================================================
// TRAINING HELPERS
// =============================================================================

void cnn_zero_gradients(CNNWeights *w) {
    memset(w->d_conv1_w, 0, 64 * CNN_INPUT_CHANNELS * 9 * sizeof(float)); memset(w->d_conv1_b, 0, 64 * sizeof(float));
    memset(w->d_conv2_w, 0, 64 * 64 * 9 * sizeof(float)); memset(w->d_conv2_b, 0, 64 * sizeof(float));
    memset(w->d_conv3_w, 0, 64 * 64 * 9 * sizeof(float)); memset(w->d_conv3_b, 0, 64 * sizeof(float));
    memset(w->d_conv4_w, 0, 64 * 64 * 9 * sizeof(float)); memset(w->d_conv4_b, 0, 64 * sizeof(float));
    memset(w->d_bn1_gamma, 0, 64 * sizeof(float)); memset(w->d_bn1_beta, 0, 64 * sizeof(float));
    memset(w->d_bn2_gamma, 0, 64 * sizeof(float)); memset(w->d_bn2_beta, 0, 64 * sizeof(float));
    memset(w->d_bn3_gamma, 0, 64 * sizeof(float)); memset(w->d_bn3_beta, 0, 64 * sizeof(float));
    memset(w->d_bn4_gamma, 0, 64 * sizeof(float)); memset(w->d_bn4_beta, 0, 64 * sizeof(float));
    memset(w->d_policy_w, 0, 512 * 4097 * sizeof(float)); memset(w->d_policy_b, 0, 512 * sizeof(float));
    memset(w->d_value_w1, 0, 256 * 4097 * sizeof(float)); memset(w->d_value_b1, 0, 256 * sizeof(float));
    memset(w->d_value_w2, 0, 1 * 256 * sizeof(float)); memset(w->d_value_b2, 0, 1 * sizeof(float));
}

// Internal forward pass for training (saving intermediate activations)
void cnn_forward_train(
    CNNWeights *w, const float *input, float player,
    float *conv1_pre, float *bn1_out, float *bn1_pre_relu, float *bn1_mean, float *bn1_var,
    float *conv2_pre, float *bn2_out, float *bn2_pre_relu, float *bn2_mean, float *bn2_var,
    float *conv3_pre, float *bn3_out, float *bn3_pre_relu, float *bn3_mean, float *bn3_var,
    float *conv4_pre, float *bn4_out, float *bn4_pre_relu, float *bn4_mean, float *bn4_var,
    float *fc_input, float *policy_out, float *value_h, float *value_out
) {
    // 1. Layer 1 (Fused BN+ReLU)
    conv2d_forward(input, w->conv1_w, w->conv1_b, conv1_pre, 8, 8, CNN_INPUT_CHANNELS, 64, 3);
    batch_norm_forward_relu(conv1_pre, w->bn1_gamma, w->bn1_beta, bn1_out, bn1_pre_relu, bn1_mean, bn1_var, w->bn1_mean, w->bn1_var, 64, 8, 8, 1);

    // 2. Layer 2 (Fused BN+ReLU)
    conv2d_forward(bn1_out, w->conv2_w, w->conv2_b, conv2_pre, 8, 8, 64, 64, 3);
    batch_norm_forward_relu(conv2_pre, w->bn2_gamma, w->bn2_beta, bn2_out, bn2_pre_relu, bn2_mean, bn2_var, w->bn2_mean, w->bn2_var, 64, 8, 8, 1);

    // 3. Layer 3 (Fused BN+ReLU)
    conv2d_forward(bn2_out, w->conv3_w, w->conv3_b, conv3_pre, 8, 8, 64, 64, 3);
    batch_norm_forward_relu(conv3_pre, w->bn3_gamma, w->bn3_beta, bn3_out, bn3_pre_relu, bn3_mean, bn3_var, w->bn3_mean, w->bn3_var, 64, 8, 8, 1);

    // 4. Layer 4 (Fused BN+ReLU)
    conv2d_forward(bn3_out, w->conv4_w, w->conv4_b, conv4_pre, 8, 8, 64, 64, 3);
    batch_norm_forward_relu(conv4_pre, w->bn4_gamma, w->bn4_beta, bn4_out, bn4_pre_relu, bn4_mean, bn4_var, w->bn4_mean, w->bn4_var, 64, 8, 8, 1);

    // 5. Head
    memcpy(fc_input, bn4_out, 4096 * sizeof(float));
    fc_input[4096] = player;

    // Policy head: policy_out = W * fc_input + bias (BLAS optimized)
    memcpy(policy_out, w->policy_b, 512 * sizeof(float));  // Start with bias
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 512, 4097, 1.0f, 
                w->policy_w, 4097, fc_input, 1, 1.0f, policy_out, 1);
    
    // Softmax (vDSP vectorized)
    float max_p;
    vDSP_maxv(policy_out, 1, &max_p, 512);  // Find max
    float neg_max = -max_p;
    vDSP_vsadd(policy_out, 1, &neg_max, policy_out, 1, 512);  // Subtract max
    int n512 = 512;
    vvexpf(policy_out, policy_out, &n512);  // Vectorized exp
    float sum_exp;
    vDSP_sve(policy_out, 1, &sum_exp, 512);  // Sum
    vDSP_vsdiv(policy_out, 1, &sum_exp, policy_out, 1, 512);  // Divide by sum

    // Value head layer 1: value_h = ReLU(W1 * fc_input + b1) (BLAS optimized)
    memcpy(value_h, w->value_b1, 256 * sizeof(float));  // Start with bias
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 256, 4097, 1.0f,
                w->value_w1, 4097, fc_input, 1, 1.0f, value_h, 1);
    for (int i = 0; i < 256; i++) value_h[i] = relu(value_h[i]);
    
    // Value head layer 2: v_out = tanh(w2 * value_h + b2)
    float v_sum = w->value_b2[0];
    for (int i = 0; i < 256; i++) v_sum += value_h[i] * w->value_w2[i];
    *value_out = tanh_act(v_sum);
}

// =============================================================================
// THREAD-LOCAL GRADIENTS FOR DATA PARALLELISM
// =============================================================================

typedef struct {
    float *d_conv1_w, *d_conv1_b;
    float *d_conv2_w, *d_conv2_b;
    float *d_conv3_w, *d_conv3_b;
    float *d_conv4_w, *d_conv4_b;
    float *d_bn1_gamma, *d_bn1_beta;
    float *d_bn2_gamma, *d_bn2_beta;
    float *d_bn3_gamma, *d_bn3_beta;
    float *d_bn4_gamma, *d_bn4_beta;
    float *d_policy_w, *d_policy_b;
    float *d_value_w1, *d_value_b1;
    float *d_value_w2, *d_value_b2;
} LocalGradients;

static void local_grads_init(LocalGradients *lg) {
    lg->d_conv1_w = calloc(64 * CNN_INPUT_CHANNELS * 9, sizeof(float));
    lg->d_conv1_b = calloc(64, sizeof(float));
    lg->d_conv2_w = calloc(64 * 64 * 9, sizeof(float));
    lg->d_conv2_b = calloc(64, sizeof(float));
    lg->d_conv3_w = calloc(64 * 64 * 9, sizeof(float));
    lg->d_conv3_b = calloc(64, sizeof(float));
    lg->d_conv4_w = calloc(64 * 64 * 9, sizeof(float));
    lg->d_conv4_b = calloc(64, sizeof(float));
    lg->d_bn1_gamma = calloc(64, sizeof(float));
    lg->d_bn1_beta = calloc(64, sizeof(float));
    lg->d_bn2_gamma = calloc(64, sizeof(float));
    lg->d_bn2_beta = calloc(64, sizeof(float));
    lg->d_bn3_gamma = calloc(64, sizeof(float));
    lg->d_bn3_beta = calloc(64, sizeof(float));
    lg->d_bn4_gamma = calloc(64, sizeof(float));
    lg->d_bn4_beta = calloc(64, sizeof(float));
    lg->d_policy_w = calloc(512 * 4097, sizeof(float));
    lg->d_policy_b = calloc(512, sizeof(float));
    lg->d_value_w1 = calloc(256 * 4097, sizeof(float));
    lg->d_value_b1 = calloc(256, sizeof(float));
    lg->d_value_w2 = calloc(256, sizeof(float));
    lg->d_value_b2 = calloc(1, sizeof(float));
}

static void local_grads_free(LocalGradients *lg) {
    free(lg->d_conv1_w); free(lg->d_conv1_b);
    free(lg->d_conv2_w); free(lg->d_conv2_b);
    free(lg->d_conv3_w); free(lg->d_conv3_b);
    free(lg->d_conv4_w); free(lg->d_conv4_b);
    free(lg->d_bn1_gamma); free(lg->d_bn1_beta);
    free(lg->d_bn2_gamma); free(lg->d_bn2_beta);
    free(lg->d_bn3_gamma); free(lg->d_bn3_beta);
    free(lg->d_bn4_gamma); free(lg->d_bn4_beta);
    free(lg->d_policy_w); free(lg->d_policy_b);
    free(lg->d_value_w1); free(lg->d_value_b1);
    free(lg->d_value_w2); free(lg->d_value_b2);
}

static void local_grads_merge(CNNWeights *w, LocalGradients *lg) {
    // Merge all thread-local gradients into global (called once per thread end)
    for (int i = 0; i < 64 * CNN_INPUT_CHANNELS * 9; i++) w->d_conv1_w[i] += lg->d_conv1_w[i];
    for (int i = 0; i < 64; i++) w->d_conv1_b[i] += lg->d_conv1_b[i];
    for (int i = 0; i < 64 * 64 * 9; i++) w->d_conv2_w[i] += lg->d_conv2_w[i];
    for (int i = 0; i < 64; i++) w->d_conv2_b[i] += lg->d_conv2_b[i];
    for (int i = 0; i < 64 * 64 * 9; i++) w->d_conv3_w[i] += lg->d_conv3_w[i];
    for (int i = 0; i < 64; i++) w->d_conv3_b[i] += lg->d_conv3_b[i];
    for (int i = 0; i < 64 * 64 * 9; i++) w->d_conv4_w[i] += lg->d_conv4_w[i];
    for (int i = 0; i < 64; i++) w->d_conv4_b[i] += lg->d_conv4_b[i];
    for (int i = 0; i < 64; i++) { w->d_bn1_gamma[i] += lg->d_bn1_gamma[i]; w->d_bn1_beta[i] += lg->d_bn1_beta[i]; }
    for (int i = 0; i < 64; i++) { w->d_bn2_gamma[i] += lg->d_bn2_gamma[i]; w->d_bn2_beta[i] += lg->d_bn2_beta[i]; }
    for (int i = 0; i < 64; i++) { w->d_bn3_gamma[i] += lg->d_bn3_gamma[i]; w->d_bn3_beta[i] += lg->d_bn3_beta[i]; }
    for (int i = 0; i < 64; i++) { w->d_bn4_gamma[i] += lg->d_bn4_gamma[i]; w->d_bn4_beta[i] += lg->d_bn4_beta[i]; }
    for (int i = 0; i < 512 * 4097; i++) w->d_policy_w[i] += lg->d_policy_w[i];
    for (int i = 0; i < 512; i++) w->d_policy_b[i] += lg->d_policy_b[i];
    for (int i = 0; i < 256 * 4097; i++) w->d_value_w1[i] += lg->d_value_w1[i];
    for (int i = 0; i < 256; i++) w->d_value_b1[i] += lg->d_value_b1[i];
    for (int i = 0; i < 256; i++) w->d_value_w2[i] += lg->d_value_w2[i];
    w->d_value_b2[0] += lg->d_value_b2[0];
}

// NOTE: cnn_backward_sample was removed - logic is inlined in cnn_train_step


// =============================================================================
// SGD OPTIMIZER
// =============================================================================

void cnn_clip_gradients(CNNWeights *w, float threshold) {
    float *grads[] = {w->d_conv1_w, w->d_conv1_b, w->d_conv2_w, w->d_conv2_b, w->d_conv3_w, w->d_conv3_b, w->d_conv4_w, w->d_conv4_b,
                      w->d_bn1_gamma, w->d_bn1_beta, w->d_bn2_gamma, w->d_bn2_beta, w->d_bn3_gamma, w->d_bn3_beta, w->d_bn4_gamma, w->d_bn4_beta,
                      w->d_policy_w, w->d_policy_b, w->d_value_w1, w->d_value_b1, w->d_value_w2, w->d_value_b2};
    size_t sizes[] = {64*CNN_INPUT_CHANNELS*9, 64, 64*64*9, 64, 64*64*9, 64, 64*64*9, 64,
                      64, 64, 64, 64, 64, 64, 64, 64,
                      512*4097, 512, 256*4097, 256, 256, 1};

    for (int i = 0; i < 22; i++) {
        float *g = grads[i];
        size_t sz = sizes[i];
        #pragma omp parallel for
        for (size_t j = 0; j < sz; j++) {
            if (g[j] > threshold) g[j] = threshold;
            if (g[j] < -threshold) g[j] = -threshold;
        }
    }
}

// Parallelized weight update with momentum SGD
static void update_layer(float *w, float *d_w, float *v_w, size_t size, float lr, float momentum, float l1, float l2) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        float grad = d_w[i] + l2 * w[i] + l1 * (w[i] > 0 ? 1.0f : -1.0f);
        v_w[i] = momentum * v_w[i] + lr * grad / size; 
        w[i] -= v_w[i];
    }
}


void cnn_update_weights(CNNWeights *w, float policy_lr, float value_lr, float momentum, float l1, float l2, int batch_size) {
    float scaled_policy_lr = policy_lr / batch_size;
    float scaled_value_lr = value_lr / batch_size;
    
    // Shared backbone uses geometric mean of both LRs for balanced learning
    float scaled_backbone_lr = sqrtf(scaled_policy_lr * scaled_value_lr);
    
    // Backbone (conv layers, BN) - shared between both heads
    update_layer(w->conv1_w, w->d_conv1_w, w->v_conv1_w, 64*CNN_INPUT_CHANNELS*9, scaled_backbone_lr, momentum, l1, l2);
    update_layer(w->conv1_b, w->d_conv1_b, w->v_conv1_b, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->conv2_w, w->d_conv2_w, w->v_conv2_w, 64*64*9, scaled_backbone_lr, momentum, l1, l2);
    update_layer(w->conv2_b, w->d_conv2_b, w->v_conv2_b, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->conv3_w, w->d_conv3_w, w->v_conv3_w, 64*64*9, scaled_backbone_lr, momentum, l1, l2);
    update_layer(w->conv3_b, w->d_conv3_b, w->v_conv3_b, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->conv4_w, w->d_conv4_w, w->v_conv4_w, 64*64*9, scaled_backbone_lr, momentum, l1, l2);
    update_layer(w->conv4_b, w->d_conv4_b, w->v_conv4_b, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->bn1_gamma, w->d_bn1_gamma, w->v_bn1_gamma, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->bn1_beta, w->d_bn1_beta, w->v_bn1_beta, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->bn2_gamma, w->d_bn2_gamma, w->v_bn2_gamma, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->bn2_beta, w->d_bn2_beta, w->v_bn2_beta, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->bn3_gamma, w->d_bn3_gamma, w->v_bn3_gamma, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->bn3_beta, w->d_bn3_beta, w->v_bn3_beta, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->bn4_gamma, w->d_bn4_gamma, w->v_bn4_gamma, 64, scaled_backbone_lr, momentum, 0, 0);
    update_layer(w->bn4_beta, w->d_bn4_beta, w->v_bn4_beta, 64, scaled_backbone_lr, momentum, 0, 0);
    
    // Policy head - uses policy_lr
    update_layer(w->policy_w, w->d_policy_w, w->v_policy_w, 512*4097, scaled_policy_lr, momentum, l1, l2);
    update_layer(w->policy_b, w->d_policy_b, w->v_policy_b, 512, scaled_policy_lr, momentum, 0, 0);
    
    // Value head - uses value_lr
    update_layer(w->value_w1, w->d_value_w1, w->v_value_w1, 256*4097, scaled_value_lr, momentum, l1, l2);
    update_layer(w->value_b1, w->d_value_b1, w->v_value_b1, 256, scaled_value_lr, momentum, 0, 0);
    update_layer(w->value_w2, w->d_value_w2, w->v_value_w2, 256, scaled_value_lr, momentum, l1, l2);
    update_layer(w->value_b2, w->d_value_b2, w->v_value_b2, 1, scaled_value_lr, momentum, 0, 0);
}

// =============================================================================
// FULLY PARALLELIZED TRAINING STEP (Data Parallel)
// =============================================================================

float cnn_train_step(CNNWeights *w, const TrainingSample *batch, int batch_size, float policy_lr, float value_lr, float l1, float l2, float *out_policy_loss, float *out_value_loss) {
    cnn_zero_gradients(w);
    
    float total_loss = 0, total_p_loss = 0, total_v_loss = 0;

    // True data parallelism: each thread has its own gradient buffer
    #pragma omp parallel reduction(+:total_loss, total_p_loss, total_v_loss)
    {
        // Thread-local gradient accumulator
        LocalGradients local;
        local_grads_init(&local);
        
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < batch_size; i++) {
            // 1. Encode input
            float player;
            float input[CNN_INPUT_CHANNELS * 64];
            cnn_encode_sample(&batch[i], input, &player);
            
            // 2. UNIFIED Forward pass (training mode BN stats)
            float conv1_pre[64*64], bn1_out[64*64], bn1_pre_relu[64*64], bn1_mean[64], bn1_var[64];
            float conv2_pre[64*64], bn2_out[64*64], bn2_pre_relu[64*64], bn2_mean[64], bn2_var[64];
            float conv3_pre[64*64], bn3_out[64*64], bn3_pre_relu[64*64], bn3_mean[64], bn3_var[64];
            float conv4_pre[64*64], bn4_out[64*64], bn4_pre_relu[64*64], bn4_mean[64], bn4_var[64];
            float fc_input[4097], policy_out[512], value_h[256], value_out;

            cnn_forward_train(w, input, player, 
                conv1_pre, bn1_out, bn1_pre_relu, bn1_mean, bn1_var,
                conv2_pre, bn2_out, bn2_pre_relu, bn2_mean, bn2_var,
                conv3_pre, bn3_out, bn3_pre_relu, bn3_mean, bn3_var,
                conv4_pre, bn4_out, bn4_pre_relu, bn4_mean, bn4_var,
                fc_input, policy_out, value_h, &value_out);
            
            // 3. Compute loss using SAME forward outputs
            float p_loss = 0;
            for (int j = 0; j < 512; j++) {
                p_loss -= batch[i].target_policy[j] * logf(policy_out[j] + 1e-10f);
            }
            float v_loss = (value_out - batch[i].target_value) * (value_out - batch[i].target_value);
            
            total_p_loss += p_loss;
            total_v_loss += v_loss;
            total_loss += (p_loss + v_loss);

            // 4. Backward pass using SAME forward buffers (no recomputation!)
            float d_policy[512], d_value;
            for (int j = 0; j < 512; j++) d_policy[j] = policy_out[j] - batch[i].target_policy[j];
            d_value = 2.0f * (value_out - batch[i].target_value) * (1.0f - value_out * value_out);

            // FC gradients
            float d_fc_input[4097];
            memset(d_fc_input, 0, sizeof(d_fc_input));
            
            // Value Head backward
            float d_v_h[256];
            for (int j = 0; j < 256; j++) {
                float dv_act = d_value * w->value_w2[j] * (value_h[j] > 0 ? 1.0f : 0.0f);
                d_v_h[j] = dv_act;
                local.d_value_w2[j] += d_value * value_h[j];
                local.d_value_b2[0] += d_value;
            }
            
            // Value FC1 backward (BLAS optimized)
            // d_W1 += d_v_h ⊗ fc_input (outer product)
            cblas_sger(CblasRowMajor, 256, 4097, 1.0f,
                       d_v_h, 1, fc_input, 1, local.d_value_w1, 4097);
            // d_fc_input += W1^T * d_v_h (transposed matvec)
            cblas_sgemv(CblasRowMajor, CblasTrans, 256, 4097, 1.0f,
                        w->value_w1, 4097, d_v_h, 1, 1.0f, d_fc_input, 1);
            // d_bias1 += d_v_h
            for (int j = 0; j < 256; j++) local.d_value_b1[j] += d_v_h[j];

            // Policy Head backward (BLAS optimized)
            // d_W += d_policy ⊗ fc_input (outer product)
            cblas_sger(CblasRowMajor, 512, 4097, 1.0f,
                       d_policy, 1, fc_input, 1, local.d_policy_w, 4097);
            // d_fc_input += W^T * d_policy (transposed matvec)
            cblas_sgemv(CblasRowMajor, CblasTrans, 512, 4097, 1.0f,
                        w->policy_w, 4097, d_policy, 1, 1.0f, d_fc_input, 1);
            // d_bias += d_policy
            for (int j = 0; j < 512; j++) local.d_policy_b[j] += d_policy[j];

            // Conv backward using SAME pre_relu buffers
            float d_bn4[64*64], d_conv4_pre[64*64];
            memcpy(d_bn4, d_fc_input, 4096 * sizeof(float)); 
            for (int j = 0; j < 64 * 64; j++) if (bn4_pre_relu[j] <= 0) d_bn4[j] = 0;
            batch_norm_backward(d_bn4, conv4_pre, w->bn4_gamma, bn4_mean, bn4_var, d_conv4_pre, local.d_bn4_gamma, local.d_bn4_beta, 64, 8, 8);

            float d_bn3[64*64], d_conv3_pre[64*64];
            conv2d_backward(bn3_out, w->conv4_w, d_conv4_pre, d_bn3, local.d_conv4_w, local.d_conv4_b, 8, 8, 64, 64, 3);
            for (int j = 0; j < 64 * 64; j++) if (bn3_pre_relu[j] <= 0) d_bn3[j] = 0;
            batch_norm_backward(d_bn3, conv3_pre, w->bn3_gamma, bn3_mean, bn3_var, d_conv3_pre, local.d_bn3_gamma, local.d_bn3_beta, 64, 8, 8);

            float d_bn2[64*64], d_conv2_pre[64*64];
            conv2d_backward(bn2_out, w->conv3_w, d_conv3_pre, d_bn2, local.d_conv3_w, local.d_conv3_b, 8, 8, 64, 64, 3);
            for (int j = 0; j < 64 * 64; j++) if (bn2_pre_relu[j] <= 0) d_bn2[j] = 0;
            batch_norm_backward(d_bn2, conv2_pre, w->bn2_gamma, bn2_mean, bn2_var, d_conv2_pre, local.d_bn2_gamma, local.d_bn2_beta, 64, 8, 8);

            float d_bn1[64*64], d_conv1_pre[64*64], d_input[CNN_INPUT_CHANNELS * 64];
            conv2d_backward(bn1_out, w->conv2_w, d_conv2_pre, d_bn1, local.d_conv2_w, local.d_conv2_b, 8, 8, 64, 64, 3);
            for (int j = 0; j < 64 * 64; j++) if (bn1_pre_relu[j] <= 0) d_bn1[j] = 0;
            batch_norm_backward(d_bn1, conv1_pre, w->bn1_gamma, bn1_mean, bn1_var, d_conv1_pre, local.d_bn1_gamma, local.d_bn1_beta, 64, 8, 8);
            conv2d_backward(input, w->conv1_w, d_conv1_pre, d_input, local.d_conv1_w, local.d_conv1_b, 8, 8, CNN_INPUT_CHANNELS, 64, 3);
        }
        
        // Merge thread-local gradients into global (once per thread at end)
        #pragma omp critical
        {
            local_grads_merge(w, &local);
        }
        
        local_grads_free(&local);
    }

    cnn_clip_gradients(w, 5.0f);
    cnn_update_weights(w, policy_lr, value_lr, 0.9f, l1, l2, batch_size);

    if (out_policy_loss) *out_policy_loss = total_p_loss / batch_size;
    if (out_value_loss) *out_value_loss = total_v_loss / batch_size;
    return total_loss / batch_size;
}

// =============================================================================
// CLEANUP (Call at training end to free all thread-local buffers)
// =============================================================================

void cnn_training_cleanup(void) {
    // Spawn parallel region so each thread can free its own TLS buffer
    #pragma omp parallel
    {
        conv_ops_cleanup();
    }
}
