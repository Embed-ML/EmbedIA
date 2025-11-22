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
static uint16_t compute_padding(int stride, int in_size, int filter_size, int out_size) {
    int dilation_rate = 1;
    int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    int total_padding = ((out_size - 1) * stride + effective_filter_size - in_size);
    total_padding = total_padding > 0 ? total_padding : 0;
    return total_padding / 2;
}

/*
 * Allocates and configures output tensor for convolution
 * - Handles both VALID and SAME padding modes
 * - Uses swap_alloc() for temporary memory buffers
 */
static void calc_alloc_conv2d_output(uint16_t n_filters, size2d_t kernel_sz, size2d_t strides,
                                     uint8_t padding, data3d_t input, data3d_t *output) {
    if (padding == PAD_VALID) {
        output->height = (input.height + strides.h - kernel_sz.h) / strides.h;
        output->width  = (input.width  + strides.w - kernel_sz.w) / strides.w;
    } else {
        output->height = (input.height + strides.h - 1) / strides.h;
        output->width  = (input.width  + strides.w - 1) / strides.w;
    }
    output->channels = n_filters; // total of output channels
    output->data = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );
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
void conv2d_padding_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output) {
    int32_t delta, i, j, k, l, f_pos, i_pos;
    int16_t f, c, i_pad, j_pad, pad_h, pad_w;
    float value;

    // calculate output size and allocate memory
    calc_alloc_conv2d_output(layer.n_filters, layer.kernel, layer.strides, layer.padding, input, output);

    pad_h = compute_padding(layer.strides.h, input.height, layer.kernel.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.kernel.w, output->width);

    for (f = 0; f < layer.n_filters; f++) {
        delta = f * output->height * output->width;
        for (i = 0; i < output->height; i++) {
            for (j = 0; j < output->width; j++) {
                value = 0;
                for (c = 0; c < layer.channels; c++) {
                    for (k = 0; k < layer.kernel.h; k++) {
                        for (l = 0; l < layer.kernel.w; l++) {
                            i_pad = i * layer.strides.h + k - pad_h;
                            j_pad = j * layer.strides.w + l - pad_w;
                            // Check for valid input access within padded bounds
                            if (i_pad >= 0 && i_pad < input.height && j_pad >= 0 && j_pad < input.width) {
                                f_pos = (c * layer.kernel.h * layer.kernel.w) + k * layer.kernel.w + l;
                                i_pos = (c * input.height * input.width) + i_pad * input.width + j_pad;
                                value += layer.filters[f].weights[f_pos] * input.data[i_pos];
                            }
                        }
                    }
                }
                output->data[delta + i * output->width + j] = value + layer.filters[f].bias;
            }
        }
    }
}


/*
 * Optimized convolution for stride=1 without padding
 * - Removes bounds checking for maximum speed
 * - Uses simpler memory addressing
 */
void conv2d_strides_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output){
    int32_t delta, i,j,k,l, f_pos, i_pos;
    int16_t f, c;
    float value;

    // calculate output size and allocate memory
    calc_alloc_conv2d_output(layer.n_filters, layer.kernel, layer.strides, layer.padding, input, output);

    for(f=0; f<layer.n_filters; f++){
        delta = f * output->height * output->width;
        for(i=0; i < output->height; i++) {
            for (j = 0; j < output->width; j++) {
                value = 0;
                for(c=0; c<layer.channels; c++){
                    for(k=0; k<layer.kernel.h; k++){
                        for(l=0; l<layer.kernel.w; l++){
                            f_pos = (c*layer.kernel.h*layer.kernel.w)+k*layer.kernel.w+l;
                            i_pos = (c * input.height * input.width) +      // start of channel
                                    (i*layer.strides.h + k) * input.width + // start of row
                                    (j*layer.strides.w + l);                // offset from start

                            value += layer.filters[f].weights[f_pos] * input.data[i_pos];
                        }
                    }
                }
                output->data[delta + i*output->width + j] = value + layer.filters[f].bias;
            }
        }
    }
}


/*
 * Basic convolution implementation for stride=1 without padding
 * - Simplest form of convolution operation
 * - Direct implementation of sliding window approach
 */
void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output) {
    int32_t delta, i, j, k, l, f_pos, i_pos;
    int16_t f, c;
    float value;

    // calculate output size and allocate memory
    calc_alloc_conv2d_output(layer.n_filters, layer.kernel, layer.strides, layer.padding, input, output);

    for(f=0; f<layer.n_filters; f++){
        delta = f*output->height*output->width;
        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){
                value = 0;
                for(c=0; c<layer.channels; c++){
                    for(k=0; k<layer.kernel.h; k++){
                        for(l=0; l<layer.kernel.w; l++){
                            f_pos = (c*layer.kernel.h*layer.kernel.w)+k*layer.kernel.w+l; // assumes strides=1
                            i_pos = (c * input.height * input.width) + // start of channel
                                    (i + k) * input.width +            // start of row
                                    (j + l);                           // offset from start

                            value += layer.filters[f].weights[f_pos] * input.data[i_pos];
                        }
                    }
                }
                output->data[delta + i*output->width + j] = value + layer.filters[f].bias;
            }
        }
    }
}

/*
 * Depthwise convolution operation for separable convolutions
 * - Applies single filter per input channel
 * - Includes padding and bounds checking
 * - Used as first step in separable convolutions
 */
static void depthwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t *output) {
    uint32_t i, j, k, l, c, f_pos, i_pos, pad_h, pad_w, i_pad, j_pad;
    float sum;

    pad_h = compute_padding(layer.strides.h, input.height, layer.depth_kernel_sz.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.depth_kernel_sz.w, output->width);

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            for (c = 0; c < layer.depth_channels; c++) {
                sum = 0;
                for (k = 0; k < layer.depth_kernel_sz.h; k++) {
                    for (l = 0; l < layer.depth_kernel_sz.w; l++) {
                            i_pad = i * layer.strides.h + k - pad_h;
                            j_pad = j * layer.strides.w + l - pad_w;
                            // Check for valid input access within padded bounds
                            if (i_pad >= 0 && i_pad < input.height && j_pad >= 0 && j_pad < input.width) {
                                f_pos = (c * layer.depth_kernel_sz.h * layer.depth_kernel_sz.w) + k * layer.depth_kernel_sz.w + l;
                                i_pos = (c * input.height * input.width) + i_pad * input.width + j_pad;
                                sum += filter.weights[f_pos] * input.data[i_pos];
                            }
                    }
                }
                output->data[c * output->width * output->height + i * output->width + j] = sum;
            }
        }
    }
}


/*
 * Pointwise convolution operation for separable convolutions
 * - 1x1 convolution combining channels from depthwise step
 * - Efficient channel mixing with minimal computation
 */
static void pointwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t *output, uint32_t delta) {
    uint32_t i, j, c, i_pos;
    float sum;

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            sum = 0;
            for (c = 0; c < layer.point_channels; c++) {
                i_pos = (c * input.height * input.width) + (i * 1) * input.width + (j * 1);
                sum += (filter.weights[c] * input.data[i_pos]);
            }
            output->data[delta + i * output->width + j] = sum + filter.bias;
        }
    }
}


/*
 * Complete separable convolution implementation
 * - Combines depthwise and pointwise steps
 * - More efficient than standard convolution
 * - Reduces computation while maintaining similar capacity
 */
void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t * output) {
    uint32_t delta, i;
    data3d_t depth_output;

    calc_alloc_conv2d_output(layer.depth_channels, layer.depth_kernel_sz, layer.strides, layer.padding, input, &depth_output);
    depthwise(layer, layer.depth_filter, input, &depth_output);

    output->channels = layer.n_filters; //cantidad de filtros
    output->height   = depth_output.height;
    output->width    = depth_output.width;
    output->data     = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );

    for(i=0; i<layer.n_filters; i++){
        delta = i*(output->height)*(output->width);
        pointwise(layer, layer.point_filters[i], depth_output,output,delta);
    }
}

/*
 * Standalone depthwise convolution layer
 * - Applies single filter per input channel
 * - Includes padding and bounds checking
 * - More efficient than standard convolution for certain architectures
 */
static void depthwise_bias(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output){
    uint32_t i, j, k, l, c, f_pos, i_pos, pad_h, pad_w, j_pad, i_pad;
    float sum;

    pad_h = compute_padding(layer.strides.h, input.height, layer.kernel_sz.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.kernel_sz.w, output->width);

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            for (c = 0; c < layer.channels; c++) {
                sum = 0;
                for (k = 0; k < layer.kernel_sz.h; k++) {
                    for (l = 0; l < layer.kernel_sz.w; l++) {

                            i_pad = i * layer.strides.h + k - pad_h;
                            j_pad = j * layer.strides.w + l - pad_w;
                            // Check for valid input access within padded bounds
                            if (i_pad >= 0 && i_pad < input.height && j_pad >= 0 && j_pad < input.width) {
                                f_pos = (c * layer.kernel_sz.h * layer.kernel_sz.w) + k * layer.kernel_sz.w + l;
                                i_pos = (c * input.height * input.width) + i_pad * input.width + j_pad;
                                sum += layer.weights[f_pos] * input.data[i_pos];
                            }


                    }
                }
                output->data[c * output->width * output->height + i * output->width + j] = sum + layer.bias[c];
            }
        }
    }
}


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
void depthwise_conv2d_layer(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output) {
    calc_alloc_conv2d_output(layer.channels, layer.kernel_sz, layer.strides, layer.padding, input, output);
    depthwise_bias(layer, input, output);
}

/*
 * Fully connected dense layer implementation
 * - Each output neuron connects to all inputs
 * - Uses optimized dot product with bias
 * - Fundamental building block for MLPs
 */
void dense_layer(dense_layer_t *layer, data1d_t *input, data1d_t *output) {
    output->length = layer->output_size;
    output->data = (float*)swap_alloc(sizeof(float) * output->length);

    for (uint32_t i = 0; i < layer->output_size; i++) {
        // Get the weights for the i-th neuron: strides across input_size
        const float *neuron_weights = &layer->weights[i * layer->input_size];
        output->data[i] = dot_product_bias(neuron_weights, input->data, input->length, layer->biases[i]);
    }
}


/*
 * Max pooling layer implementation
 * - Downsamples input by taking maximum value in each window
 * - Preserves channel dimensions
 * - Commonly used for spatial invariance
 */
void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
    uint32_t c,i,j,aux1,aux2;
    float max = -INFINITY;
    float num;

    output->height = ((uint16_t) ((input.height - pool.size)/pool.strides)) + 1;
    output->width  = ((uint16_t) ((input.width - pool.size)/pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (float*)swap_alloc(sizeof(float)*(output->channels)*(output->height)*(output->width));

    for(c=0; c<output->channels; c++){
        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){
                max = -INFINITY;
                for(aux1=0; aux1<pool.size; aux1++){
                    for(aux2=0; aux2<pool.size; aux2++){
                        num = input.data[c*input.width*input.height + (i*pool.strides + aux1)*input.width + j*pool.strides + aux2];
                        if(num>max){
                            max = num;
                        }
                    }
                }
                output->data[c*output->width*output->height + i*output->width + j] = max;
            }
        }
    }
}


/*
 * Average pooling layer implementation
 * - Downsamples input by averaging values in each window
 * - Preserves channel dimensions
 * - Smoother downsampling than max pooling
 */
void average_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
    uint32_t c,i,j,aux1,aux2;
    uint32_t cant = pool.size*pool.size;
    float avg = 0;
    float num;

    output->height = ((uint32_t)((input.height - pool.size) / pool.strides)) + 1;
    output->width  = ((uint32_t)((input.width  - pool.size) / pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (float*)swap_alloc(sizeof(float)*(output->channels)*(output->height)*(output->width));

    for(c=0; c<output->channels; c++){
        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){
                avg = 0;
                for(aux1=0; aux1<pool.size; aux1++){
                    for(aux2=0; aux2<pool.size; aux2++){
                        num = input.data[c*input.width*input.height + (i*pool.strides + aux1)*input.width + j*pool.strides + aux2];
                        avg += num;
                    }
                }
                output->data[c*output->width*output->height + i*output->width + j] = avg/cant;
            }
        }
    }
}


// ========================================================
// Activation Functions
// ========================================================

/*
 * Numerically stable softmax implementation
 * - Uses log-sum-exp trick to prevent overflow
 * - Three-pass algorithm: find max -> calculate sum -> normalize
 */
void softmax_activation(float *data, uint32_t length){
    uint32_t i;
    float m = -INFINITY;

    // Find max for numerical stability
    for (i = 0; i < length; i++) {
        if (data[i] > m) m = data[i];
    }

    // Compute sum of exponentials
    float sum = (0.0);
    for(i = 0; i < length; i++) {
        sum += exp(data[i] - m);
    }

    // Normalize

    float offset = m + log(sum);
    for(i = 0; i < length; i++) {
        data[i] = exp(data[i] - offset);
    }
}

/*
 * Rectified Linear Unit (ReLU) activation
 * - Simple thresholding at zero
 * - Computationally efficient with sparse activation
 */
void relu_activation(float *data, uint32_t length){
    uint32_t i;

    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? 0 : data[i];
    }
}

/*
 * ReLU6 activation (clipped ReLU)
 * - Thresholds activations at 0 and 6
 * - Used in quantization-aware training
 */
void relu6_activation(float *data, uint32_t length) {

    uint32_t i;
    for (i = 0; i < length; i++) {
        if (data[i] < 0.0)
            data[i] = 0.0;
        else if (data[i] > 6.0)
            data[i] = 6.0;
    }
}

/*
 * Leaky ReLU activation
 * - Small negative slope for negative inputs
 * - Helps prevent "dying ReLU" problem
 */
void leakyrelu_activation(float *data, uint32_t length, float alpha){
    uint32_t i;
    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? alpha*data[i] : data[i];
    }
}

/*
 * Hyperbolic tangent activation
 * - Outputs in range [-1, 1]
 * - Smooth S-shaped curve
 */
void tanh_activation(float *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = 2/(1+exp(-2*data[i])) - 1;
    }
}

/*
 * Sigmoid activation
 * - Outputs in range (0, 1)
 * - Classic activation for binary classification
 */
void sigmoid_activation(float *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = 1 / (1 + exp(-data[i]));
    }
}

/*
 * Softsign activation
 * - Similar to tanh but with slower asymptotes
 * - Computationally cheaper alternative to sigmoid
 */
void softsign_activation(float *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = data[i] / (fabs(data[i])+1);
    }
}

/*
 * softplus activation function: log(e^x + 1)
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softplus_activation(float *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = log( exp(data[i])+1 );
    }
}


// ========================================================
// Normalization Functions
// ========================================================

/*
 * Normalization type 1 (mean and variance)
 * - Subtracts mean and divides by standard deviation
 * - Input preprocessing for better training stability
 */
void normalization1(normalization_layer_t n, data1d_t input, data1d_t * output){
    uint32_t i;

    output->length = input.length;
    output->data = (float*)swap_alloc(sizeof(float)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = (input.data[i]-n.sub_val[i])*n.inv_div_val[i];
    }
}

/*
 * Normalization type 2 (variance only)
 * - Scales by standard deviation without mean subtraction
 * - Used in certain network architectures
 */
void normalization2(normalization_layer_t n, data1d_t input, data1d_t * output) {
    uint32_t i;
    output->length = input.length;
    output->data = (float*)swap_alloc(sizeof(float)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = input.data[i]*n.inv_div_val[i];
    }
}

/*
 * 1D Batch Normalization
 * - Normalizes activations using learned parameters
 * - Improves training stability and convergence
 */
void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data) {
    uint32_t i;
    for (i = 0; i < data->length; i++) {
        data->data[i] = data->data[i] * layer.moving_inv_std_dev[i] + layer.std_beta[i];
    }
}

/*
 * 3D Batch Normalization
 * - Channel-wise normalization for convolutional outputs
 * - Uses per-channel scaling and shifting
 */
void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data) {
    uint32_t c, j, base_idx;
    uint32_t plane_size = data->height * data->width;
    for (c = 0; c < data->channels; c++) {
        base_idx = c * plane_size;
        for (j = 0; j < plane_size; j++) {
            data->data[base_idx + j] = data->data[base_idx + j] * layer.moving_inv_std_dev[c] + layer.std_beta[c];
        }
    }
}

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
    output->data = (float*)swap_alloc(sizeof(float) * output->length);
    for (i = 0; i < input.height; i++) {
        for (j = 0; j < input.width; j++) {
            for (c = 0; c < input.channels; c++) {
                output->data[idx++] = input.data[c * input.height * input.width + i * input.width + j];
            }
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
    output->data = (float*)swap_alloc(sizeof(float) * output->channels * output->height * output->width);

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
void channel_adapt_layer(data3d_t input, data3d_t * output){

    uint32_t i, j, c, l;

    output->channels = input.channels;
    output->height   = input.height;
    output->width    = input.width;
    output->data     = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );

    for(c=0, l=0; c < input.channels; c++){
        for(i=0; i < input.height; i++) {
            for(j=0; j < input.width; j++, l++ ){
                output->data[l] = input.data[i*input.channels*input.width+input.channels*j+c];
            }
        }
    }
}
