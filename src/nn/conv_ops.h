/**
 * conv_ops.h - Convolution Operations for CNN
 * 
 * Implements 2D convolution forward and backward passes.
 * Memory layout: NCHW (batch, channels, height, width) but we operate on CHW.
 */

#ifndef CONV_OPS_H
#define CONV_OPS_H

// =============================================================================
// TENSOR UTILITIES
// =============================================================================

// Index into a 3D tensor [C][H][W] stored as flat array
#define TENSOR_IDX(c, y, x, H, W) ((c) * (H) * (W) + (y) * (W) + (x))

// Index into a 4D kernel [Co][Ci][Ky][Kx] stored as flat array
#define KERNEL_IDX(co, ci, ky, kx, Ci, K) \
    ((co) * (Ci) * (K) * (K) + (ci) * (K) * (K) + (ky) * (K) + (kx))

// =============================================================================
// CONVOLUTION OPERATIONS
// =============================================================================

/**
 * Forward pass of 2D convolution with same padding.
 * 
 * @param input   Input tensor [Ci][H][W]
 * @param kernel  Convolution kernels [Co][Ci][K][K]
 * @param bias    Bias per output channel [Co]
 * @param output  Output tensor [Co][H][W] (same spatial size due to padding)
 * @param H       Input height
 * @param W       Input width
 * @param Ci      Input channels
 * @param Co      Output channels (number of filters)
 * @param K       Kernel size (K×K, typically 3)
 */
void conv2d_forward(
    const float *input,
    const float *kernel,
    const float *bias,
    float *output,
    int H, int W, int Ci, int Co, int K
);

/**
 * Backward pass of 2D convolution.
 * Computes gradients for input, kernel weights, and bias.
 * 
 * @param input       Input tensor from forward pass [Ci][H][W]
 * @param kernel      Kernel weights [Co][Ci][K][K]
 * @param d_output    Gradient from next layer [Co][H][W]
 * @param d_input     Gradient to propagate to previous layer [Ci][H][W] (output)
 * @param d_kernel    Gradient for kernel weights [Co][Ci][K][K] (output, accumulated)
 * @param d_bias      Gradient for bias [Co] (output, accumulated)
 * @param H, W, Ci, Co, K  Dimensions
 */
void conv2d_backward(
    const float *input,
    const float *kernel,
    const float *d_output,
    float *d_input,
    float *d_kernel,
    float *d_bias,
    int H, int W, int Ci, int Co, int K
);

// =============================================================================
// TENSOR OPERATIONS
// =============================================================================

/**
 * Apply ReLU activation in-place to tensor.
 */
void tensor_relu(float *data, int size);

/**
 * Backward pass of ReLU: multiply gradient by ReLU derivative.
 * Operates in-place on d_output.
 * 
 * @param pre_activation  Values BEFORE ReLU was applied
 * @param d_output        Gradient to modify in-place
 * @param size            Number of elements
 */
void tensor_relu_backward(const float *pre_activation, float *d_output, int size);

/**
 * Cleanup thread-local convolution buffers.
 * Call from each thread that used convolution operations.
 */
void conv_ops_cleanup(void);

#endif // CONV_OPS_H
