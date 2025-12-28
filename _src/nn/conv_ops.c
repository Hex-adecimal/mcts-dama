/**
 * conv_ops.c - Convolution Operations Implementation
 * 
 * 2D Convolution with "same" padding (output size = input size).
 * Optimized for 3×3 kernels with OpenMP parallelization and Accelerate BLAS.
 */

#include <stdlib.h>
#include <string.h>
#include <Accelerate/Accelerate.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// BUFFER MANAGEMENT (Thread-Local Pre-Allocation)
// =============================================================================

// Maximum buffer size for 8x8 board, 64 channels, 3x3 kernel
#define MAX_COL_BUFFER_SIZE (64 * 64 * 3 * 3 * 8 * 8)  // ~2.4 MB

// Thread-local buffer for im2col/col2im operations
static __thread float *tls_col_buffer = NULL;
static __thread int tls_buffer_initialized = 0;

// Initialize thread-local buffer (call once per thread before training)
static void ensure_buffer_initialized(void) {
    if (!tls_buffer_initialized) {
        tls_col_buffer = (float*)malloc(MAX_COL_BUFFER_SIZE * sizeof(float));
        tls_buffer_initialized = 1;
    }
}

// Public cleanup function (call after training)
void conv_ops_cleanup(void) {
    if (tls_buffer_initialized && tls_col_buffer) {
        free(tls_col_buffer);
        tls_col_buffer = NULL;
        tls_buffer_initialized = 0;
    }
}

// =============================================================================
// HELPER: im2col (Parallelized)
// =============================================================================

// Unrolls input [Ci, H, W] into column matrix [Ci*K*K, H*W]
static void im2col(const float *data_im, int channels, int height, int width, int kernel_size, int pad, float *data_col) {
    int height_col = height;
    int width_col = width;
    int channels_col = channels * kernel_size * kernel_size;
    
    // Parallelize across output channels - each channel is independent
    #pragma omp parallel for
    for (int c = 0; c < channels_col; c++) {
        int w_offset = c % kernel_size;
        int h_offset = (c / kernel_size) % kernel_size;
        int c_im = c / kernel_size / kernel_size;
        
        for (int h = 0; h < height_col; h++) {
            for (int w = 0; w < width_col; w++) {
                int im_row = h_offset + h - pad;
                int im_col = w_offset + w - pad;
                int col_index = (c * height_col + h) * width_col + w;
                
                if (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) {
                    data_col[col_index] = data_im[(c_im * height + im_row) * width + im_col];
                } else {
                    data_col[col_index] = 0.0f;
                }
            }
        }
    }
}

// =============================================================================
// HELPER: col2im (Parallelized with atomic)
// =============================================================================


static void col2im(const float *data_col, int channels, int height, int width, int kernel_size, int pad, float *data_im) {
    memset(data_im, 0, sizeof(float) * channels * height * width);
    int height_col = height;
    int width_col = width;
    int channels_col = channels * kernel_size * kernel_size;
    
    // Parallelize - use atomic to handle overlapping writes
    #pragma omp parallel for
    for (int c = 0; c < channels_col; c++) {
        int w_offset = c % kernel_size;
        int h_offset = (c / kernel_size) % kernel_size;
        int c_im = c / kernel_size / kernel_size;
        
        for (int h = 0; h < height_col; h++) {
            for (int w = 0; w < width_col; w++) {
                int im_row = h_offset + h - pad;
                int im_col = w_offset + w - pad;
                int col_index = (c * height_col + h) * width_col + w;
                
                if (im_row >= 0 && im_col >= 0 && im_row < height && im_col < width) {
                    int idx = (c_im * height + im_row) * width + im_col;
                    #pragma omp atomic
                    data_im[idx] += data_col[col_index];
                }
            }
        }
    }
}

// =============================================================================
// FORWARD PASS (BLAS SGEMM)
// =============================================================================

void conv2d_forward(
    const float *input,
    const float *kernel,
    const float *bias,
    float *output,
    int H, int W, int Ci, int Co, int K
) {
    int pad = K / 2;
    
    // Use thread-local pre-allocated buffer (zero allocation)
    ensure_buffer_initialized();
    float *col_buffer = tls_col_buffer;
    
    // 1. im2col (parallelized internally)
    im2col(input, Ci, H, W, K, pad, col_buffer);
    
    // 2. GEMM: Output = Weights * Col (uses Accelerate - already multi-threaded)
    int M = Co;
    int N = H * W;
    int K_dim = Ci * K * K;
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K_dim,
                1.0f, kernel, K_dim,
                col_buffer, N,
                0.0f, output, N);
    
    // 3. Add Bias (vDSP vectorized)
    #pragma omp parallel for
    for (int c = 0; c < Co; c++) {
        vDSP_vsadd(&output[c * H * W], 1, &bias[c], &output[c * H * W], 1, H * W);
    }
}

// =============================================================================
// BACKWARD PASS (BLAS SGEMM)
// =============================================================================

void conv2d_backward(
    const float *input,
    const float *kernel,
    const float *d_output,
    float *d_input,
    float *d_kernel,
    float *d_bias,
    int H, int W, int Ci, int Co, int K
) {
    int pad = K / 2;
    
    // Use thread-local pre-allocated buffer (zero allocation)
    ensure_buffer_initialized();
    float *col_buffer = tls_col_buffer;
    
    // 1. Bias Grad (parallelized with vDSP sum)
    #pragma omp parallel for
    for (int c = 0; c < Co; c++) {
        float sum = 0.0f;
        vDSP_sve(&d_output[c * H * W], 1, &sum, H * W);
        d_bias[c] += sum;
    }
    
    // 2. Re-compute im2col (parallelized internally)
    im2col(input, Ci, H, W, K, pad, col_buffer);
    
    // 3. Gradient wrt Weights: d_output * col^T (Accelerate GEMM)
    int M = Co;
    int N = Ci * K * K;
    int K_dim = H * W;
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K_dim,
                1.0f, d_output, K_dim,
                col_buffer, K_dim,
                1.0f, d_kernel, N);
    
    // 4. Gradient wrt Input
    if (d_input) {
        M = Ci * K * K;
        N = H * W;
        K_dim = Co;
        
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    M, N, K_dim,
                    1.0f, kernel, M,
                    d_output, N,
                    0.0f, col_buffer, N);
        
        // col2im (parallelized internally)
        col2im(col_buffer, Ci, H, W, K, pad, d_input);
    }
}

// =============================================================================
// TENSOR OPERATIONS (Parallelized)
// =============================================================================

void tensor_relu(float *data, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}

void tensor_relu_backward(const float *pre, float *d_out, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        if (pre[i] <= 0) d_out[i] = 0;
    }
}
