/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions

 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 * GitHub: https://github.com/Embed-ML/EmbedIA
 */

/*
 * Implementation Notes:
 * - All operations use floating-point arithmetic
 * - Memory management via swap_alloc() minimizes heap fragmentation
 * - Tensor format: CHW (Channels, Height, Width) for optimal memory access
 * - Three convolution variants for different performance/feature tradeoffs
 */

#include <stdlib.h>
#include <math.h>
#include "neural_net.h"


// ========================================================
// Core Internal Functions
// ========================================================


/*
 * Calculates symmetric padding for 'SAME' convolution mode
 * - Handles even/odd padding distribution
 * - dilation_rate currently fixed at 1 (could be parameterized)
 */
/*
static uint16_t compute_padding(int stride, int in_size, int filter_size, int out_size) {
    int dilation_rate = 1;
    int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    int total_padding = ((out_size - 1) * stride + effective_filter_size - in_size);
    total_padding = total_padding > 0 ? total_padding : 0;
    return total_padding / 2;
}

*/

static inline uint16_t compute_padding(int stride, int in_size, int filter_size, int out_size) {
    if (out_size <= 0) return 0;

    int total_padding = (out_size - 1) * stride + filter_size - in_size;
    if (total_padding < 0) total_padding = 0;

    return total_padding / 2;
}

/*
 * calc_alloc_conv2d_output()
 *   Calculates the output size for a convolutional layer.
 * Parameters:
 * n_filters => Number of filters in the convolutional layer
 * kernel_sz => Size of the convolutional kernel
 * strides => Strides for the convolution
 * padding => Type of padding (VALID or SAME)
 * input => Input data for the convolution
 * output => NULL
 */
static inline void calc_conv2d_output_size(uint16_t n_filters, size2d_t kernel_sz,
                                            size2d_t strides, uint8_t padding,
                                            data3d_t input, data3d_t *output) {
    if (padding == PAD_VALID) {
        output->height = (input.height + strides.h - kernel_sz.h) / strides.h;
        output->width  = (input.width  + strides.w - kernel_sz.w) / strides.w;
    } else {
        output->height = (input.height + strides.h - 1) / strides.h;
        output->width  = (input.width  + strides.w - 1) / strides.w;
    }
    output->channels = n_filters;
    output->data     = NULL;
}

/*
 * calc_alloc_conv2d_output()
 *   Calculates the output size for a convolutional layer and allocates memory for the output data.
 * Parameters:
 * n_filters => Number of filters in the convolutional layer
 * kernel_sz => Size of the convolutional kernel
 * strides => Strides for the convolution
 * padding => Type of padding (VALID or SAME)
 * input => Input data for the convolution
 * output => Pointer to store the output data
 */
static inline void calc_alloc_conv2d_output(uint16_t n_filters, size2d_t kernel_sz,
                                             size2d_t strides, uint8_t padding,
                                             data3d_t input, data3d_t *output) {
    calc_conv2d_output_size(n_filters, kernel_sz, strides, padding, input, output);
    output->data = (real_t*)swap_alloc(
        sizeof(real_t) * output->channels * output->height * output->width
    );
}

// ========================================================
// Convolution Layer Implementations
// ========================================================

/*
 * General convolution with padding and bounds checking
 * - Supports arbitrary strides and padding modes
 * - Includes explicit bounds checking for safe memory access
 * - Memory access pattern: Channel -> Height -> Width
 */

// Function: conv2d_padding_layer
// @embedia-include neural_net/_conv2d_padding_layer.c



/*
 * Optimized convolution for stride=1 without padding
 * - Removes bounds checking for maximum speed
 * - Uses simpler memory addressing
 */

// Function: conv2d_strides_layer
// @embedia-include neural_net/_conv2d_strides_layer.c



/*
 * Basic convolution implementation for stride=1 without padding
 * - Simplest form of convolution operation
 * - Direct implementation of sliding window approach
 */

// Function: conv2d_layer
// @embedia-include neural_net/_conv2d_layer.c



// ========================================================
// 1D Convolution Layer Implementation
// ========================================================

/*
 * Calculates symmetric padding for 'SAME' convolution mode (1D version)
 */

static uint16_t compute_padding_1d(int stride, int in_size, int filter_size, int out_size) {
    int dilation_rate = 1;
    int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    int total_padding = ((out_size - 1) * stride + effective_filter_size - in_size);
    total_padding = total_padding > 0 ? total_padding : 0;
    return total_padding / 2;
}


/*
 * Allocates and configures output tensor for 1D convolution
 */
static void calc_alloc_conv1d_output(uint16_t n_filters, uint16_t kernel_size, uint16_t stride,
                                     uint8_t padding, data2d_t input, data2d_t *output) {
    if (padding == PAD_VALID) {
        output->width = (input.width + stride - kernel_size) / stride;
    } else {
        output->width = (input.width + stride - 1) / stride;
    }
    output->channels = n_filters; // total of output channels
    output->data = (real_t*)swap_alloc(sizeof(real_t) * output->channels * output->width);
}

// ========================================================
// 1D Convolution Layer Implementations
// ========================================================

/*
 * General 1D convolution with padding and bounds checking
 */

// Function: conv1d_padding_layer
// @embedia-include neural_net/_conv1d_padding_layer.c


/*
 * Optimized 1D convolution for stride=1 without padding
 */

// Function: conv1d_strides_layer
// @embedia-include neural_net/_conv1d_strides_layer.c


/*
 * Basic 1D convolution implementation for stride=1 without padding
 */

// Function: conv1d_layer
// @embedia-include neural_net/_conv1d_layer.c



/*
 * Depthwise convolution operation for separable convolutions
 * - Applies single filter per input channel
 * - Includes padding and bounds checking
 * - Used as first step in separable convolutions
 */

// Function: depthwise
// @embedia-include neural_net/_depthwise.c



/*
 * Pointwise convolution operation for separable convolutions
 * - 1x1 convolution combining channels from depthwise step
 * - Efficient channel mixing with minimal computation
 */

// Function: pointwise
// @embedia-include neural_net/_pointwise.c



/*
 * Complete separable convolution implementation
 * - Combines depthwise and pointwise steps
 * - More efficient than standard convolution
 * - Reduces computation while maintaining similar capacity
 */

// Function: separable_conv2d_layer
// @embedia-include neural_net/_separable_conv2d_layer.c


/*
 * Standalone depthwise convolution layer
 * - Applies single filter per input channel
 * - Includes padding and bounds checking
 * - More efficient than standard convolution for certain architectures
 */

// Function: depthwise_bias
// @embedia-include neural_net/_depthwise_bias.c



/*
 * Depthwise Convolution 2D Layer
 * - Implements channel-wise spatial convolution with independent filters
 * - Each input channel has its own set of filter weights
 * - More efficient than standard convolution for depthwise operations
 *
 *   Input tensor in data3d_t format (channels, height, width)
 *   output  => Pointer to output tensor (pre-allocated by calc_alloc_conv2d_output)
 *
 * Operation:
 *   1. Calculates output dimensions and allocates memory
 *   2. Applies depthwise convolution with per-channel bias
 */

// Function: depthwise_conv2d_layer
// @embedia-include neural_net/_depthwise_conv2d_layer.c


/*
 * Fully connected dense layer implementation
 * - Each output neuron connects to all inputs
 * - Uses optimized dot product with bias
 * - Fundamental building block for MLPs
 */

// Function: dense_layer
// @embedia-include neural_net/_dense_layer.c



// ========================================================
// Local Pooling Functions
// ========================================================

/*
 * Max pooling layer implementation
 * - Downsamples input by taking maximum value in each window
 * - Preserves channel dimensions
 * - Commonly used for spatial invariance
 */

// Function: max_pooling2d_layer
// @embedia-include neural_net/_max_pooling2d_layer.c



/*
 * Average pooling layer implementation
 * - Downsamples input by averaging values in each window
 * - Preserves channel dimensions
 * - Smoother downsampling than max pooling
 */

// Function: average_pooling2d_layer
// @embedia-include neural_net/_average_pooling2d_layer.c


/*
 * Max pooling 1D layer implementation
 * - Downsamples 1D input by taking maximum value in each window
 * - Preserves channel dimensions
 * - Commonly used for temporal invariance in sequential data
 */

// Function: max_pooling1d_layer
// @embedia-include neural_net/_max_pooling1d_layer.c


/*
 * Average pooling 1D layer implementation
 * - Downsamples 1D input by averaging values in each window
 * - Preserves channel dimensions
 * - Smoother downsampling than max pooling for sequential data
 */

// Function: average_pooling1d_layer
// @embedia-include neural_net/_average_pooling1d_layer.c



// ========================================================
// Global Pooling Functions
// ========================================================
/*
 * Global Max Pooling 2D
 * Takes maximum value along spatial dimensions for each channel
 * Input: data3d_t (channels, height, width)
 * Output: data1d_t (channels)
 */

// Function: global_max_pooling2d_layer
// @embedia-include neural_net/_global_max_pooling2d_layer.c


/*
 * Global Average Pooling 2D
 * Reduces (channels, height, width) to (channels) by averaging over spatial dimensions
 * Input: data3d_t (channels, height, width)
 * Output: data1d_t (channels)
 */

// Function: global_average_pooling2d_layer
// @embedia-include neural_net/_global_average_pooling2d_layer.c


/*
 * Global Max Pooling 1D
 * Takes maximum value along width dimension for each channel
 * Input: data2d_t (channels, width)
 * Output: data1d_t (channels)
 */

// Function: global_max_pooling1d_layer
// @embedia-include neural_net/_global_max_pooling1d_layer.c


/*
 * Global Average Pooling 1D
 * Reduces (channels, width) to (channels) by averaging along width dimension
 * Input: data2d_t (channels, width)
 * Output: data1d_t (channels)
 */

// Function: global_average_pooling1d_layer
// @embedia-include neural_net/_global_average_pooling1d_layer.c



// ========================================================
// Activation Functions
// ========================================================

/*
 * Numerically stable softmax implementation
 * - Uses log-sum-exp trick to prevent overflow
 * - Three-pass algorithm: find max -> calculate sum -> normalize
 */

// Function: softmax_activation
// @embedia-include neural_net/_softmax_activation.c


/*
 * Rectified Linear Unit (ReLU) activation
 * - Simple thresholding at zero
 * - Computationally efficient with sparse activation
 */

// Function: relu_activation
// @embedia-include neural_net/_relu_activation.c


/*
 * ReLU6 activation (clipped ReLU)
 * - Thresholds activations at 0 and 6
 * - Used in quantization-aware training
 */

// Function: relu6_activation
// @embedia-include neural_net/_relu6_activation.c


/*
 * Leaky ReLU activation
 * - Small negative slope for negative inputs
 * - Helps prevent "dying ReLU" problem
 */

// Function: leakyrelu_activation
// @embedia-include neural_net/_leakyrelu_activation.c


/*
 * Hyperbolic tangent activation
 * - Outputs in range [-1, 1]
 * - Smooth S-shaped curve
 */

// Function: tanh_activation
// @embedia-include neural_net/_tanh_activation.c


/*
 * Sigmoid activation
 * - Outputs in range (0, 1)
 * - Classic activation for binary classification
 */

// Function: sigmoid_activation
// @embedia-include neural_net/_sigmoid_activation.c


/*
 * Softsign activation
 * - Similar to tanh but with slower asymptotes
 * - Computationally cheaper alternative to sigmoid
 */

// Function: softsign_activation
// @embedia-include neural_net/_softsign_activation.c


/*
 * softplus activation function: log(e^x + 1)
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */

// Function: softplus_activation
// @embedia-include neural_net/_softplus_activation.c



// ========================================================
// Normalization Functions
// ========================================================

/*
 * 1D Batch Normalization
 * - Normalizes activations using learned parameters
 * - Improves training stability and convergence
 */

// Function: batch_normalization1d_layer
// @embedia-include neural_net/_batch_normalization1d_layer.c


/*
 * 3D Batch Normalization
 * - Channel-wise normalization for convolutional outputs
 * - Uses per-channel scaling and shifting
 */

// Function: batch_normalization3d_layer
// @embedia-include neural_net/_batch_normalization3d_layer.c


// ========================================================
// Utility Functions
// ========================================================

/*
 * Converts 3D tensor (CHW) to 1D vector
 * - Used for transition between convolutional and dense layers
 * - Memory layout: Channel-major -> Row-major -> Column-major
 */
void flatten3d_layer(data3d_t input, data1d_t * output) {
    uint32_t i, j, c, idx = 0;
    output->length = input.channels * input.height * input.width;
    output->data = (real_t*)swap_alloc(sizeof(real_t) * output->length);
    for (i = 0; i < input.height; i++) {
        for (j = 0; j < input.width; j++) {
            for (c = 0; c < input.channels; c++) {
                output->data[idx++] = input.data[c * input.height * input.width + i * input.width + j];
            }
        }
    }
}



/*
 * Flattens a 2D tensor (channels × width) into a 1D vector.
 * - Used to connect 1D convolutional layers to dense layers.
 * - Memory layout of input: channel-major (channels stored sequentially).
 * - Output order: time-major with channels last, matching TensorFlow Flatten:
 *   [t0_c0, t0_c1, ..., t1_c0, t1_c1, ..., tN_c0, tN_c1, ...]
 */
void flatten2d_layer(data2d_t input, data1d_t * output) {
    uint32_t i, c, idx = 0;
    output->length = input.channels * input.width;
    output->data = (real_t*)swap_alloc(sizeof(real_t) * output->length);

    // Flatten in "time-major" order: positions first, then channels
    for (i = 0; i < input.width; i++) {
        for (c = 0; c < input.channels; c++) {
            output->data[idx++] = input.data[c * input.width + i];
        }
    }
}


/*
 * Initializes zero padding for 2D data
 * - Helper function for zero_padding2d_layer
 * - Sets border regions to zero
 */
static void zero_padding2d_init(uint8_t pad_h, uint8_t pad_w, data3d_t *output){
    uint32_t c, i, j;

    for (c = 0; c < output->channels; c++) {
        for (i = 0; i < output->height; i++) {
            for (j = 0; j < pad_w; j++) {
                output->data[(c * output->height + i) * output->width + j] = 0; // left
                output->data[(c * output->height + i) * output->width + output->width - 1 - j] = 0; // right
            }
        }
    }
    for (c = 0; c < output->channels; c++) {
        for (i = 0; i < pad_h; i++) {
            // top fill
            for (j = 0; j < output->width; j++) {
                output->data[(c * output->height + i) * output->width + j] = 0; // top
                output->data[(c * output->height + output->height - 1 - i) * output->width + j] = 0; // bottom
            }
        }
    }
}


/*
 * Zero padding for 2D data
 * - Adds border of zeros around input
 * - Preserves spatial dimensions for convolution
 */

void zero_padding2d_layer(uint8_t pad_h, uint8_t pad_w, data3d_t input, data3d_t *output) {
    uint32_t c, i, j, out_idx, in_idx;

    // Calc output dimension
    output->channels = input.channels;
    output->height = input.height + 2 * pad_h;
    output->width  = input.width  + 2 * pad_w;
    output->data = (real_t*)swap_alloc(sizeof(real_t) * output->channels * output->height * output->width);

    for (c = 0; c < input.channels; c++) {
        for (i = 0; i < input.height; i++) {
            for (j = 0; j < input.width; j++) {
                out_idx = (c * output->height + (i + pad_h)) * output->width + (j + pad_w);
                in_idx  = (c * input.height + i) * input.width + j;
                output->data[out_idx] = input.data[in_idx];
            }
        }
    }

    zero_padding2d_init(pad_h, pad_w, output);
}

/*
 * Channel adaptation layer
 * - Reorders input channels for compatibility
 * - Handles different channel ordering formats
 */
void channel_adapt_layer_3d(data3d_t input, data3d_t * output){

    uint32_t i, j, c, l;

    output->channels = input.channels;
    output->height   = input.height;
    output->width    = input.width;
    output->data     = (real_t*)swap_alloc( sizeof(real_t)*output->channels*output->height*output->width );

    for(c=0, l=0; c < input.channels; c++){
        for(i=0; i < input.height; i++) {
            for(j=0; j < input.width; j++, l++ ){
                output->data[l] = input.data[i*input.channels*input.width+input.channels*j+c];
            }
        }
    }
}




/*
 * Channel adaptation layer for 2D data (1D convolution)
 * - Reorders input channels for compatibility
 * - Handles different channel ordering formats
 * - Converts from interleaved to consecutive channel format
 */
void channel_adapt_layer_2d(data2d_t input, data2d_t * output){

    uint32_t i, c, l;

    output->channels = input.channels;
    output->width    = input.width;
    output->data     = (real_t*)swap_alloc(sizeof(real_t) * output->channels * output->width);

    // Convert from interleaved format to consecutive channels format
    for(i = 0; i < input.width; i++) {
        for(c = 0, l = 0; c < input.channels; c++, l++) {
            // Input : interleaved format [time0_ch0, time0_ch1, time1_ch0, time1_ch1, ...]
            // Output: consecutive format [ch0_time0, ch0_time1, ..., ch1_time0, ch1_time1, ...]
            output->data[c * input.width + i] = input.data[i * input.channels + c];
        }
    }
}

