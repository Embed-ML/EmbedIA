#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H
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

/**
 * @file neural_net.h
 * @brief EmbedIA - Embedded Machine Learning and Neural Networks Framework
 *
 * This library provides structures and functions for implementing neural networks on microcontrollers.
 * It supports convolutional, dense, pooling, normalization, and activation layers using fixed-point arithmetic.
 *
 * @author César Estrebou
 * @institution III-LIDI, Faculty of Informatics - National University of La Plata (UNLP)
 * @copyright BSD 3-Clause License. See LICENSE file for details.
 * @repository https://github.com/Embed-ML/EmbedIA
 */


#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "fixed.h"
#include "common.h"

//{includes}

/**
 * @defgroup layer_structures Layer Structures
 * @brief Definitions of layer data structures used in neural network models.
 * @{
 */

/**
 * @struct filter_t
 * @brief Represents a single filter (kernel) in a convolutional layer.
 *
 * Contains weights and bias for one filter.
 */
typedef struct {
    const fixed * weights;  /**< Pointer to the filter weights (kernel_size * channels) */
    fixed bias;             /**< Bias value for the filter */
} filter_t;


/**
 * @def PAD_SAME
 * @brief Padding mode: output size is same as input (padding added as needed).
 */
#define PAD_SAME 1

/**
 * @def PAD_VALID
 * @brief Padding mode: no padding applied, output size may shrink.
 */
#define PAD_VALID 0


/**
 * @struct conv2d_layer_t
 * @brief Represents a 2D convolutional layer.
 */
typedef struct {
    uint16_t n_filters;     /**< Number of filters in the layer */
    filter_t * filters;     /**< Array of filters */
    uint16_t channels;      /**< Number of input channels */
    size2d_t kernel;        /**< Kernel size (height, width) */
    uint8_t padding;        /**< Padding type: PAD_SAME or PAD_VALID */
    size2d_t strides;       /**< Stride size (vertical, horizontal) */
} conv2d_layer_t;


/**
 * @struct depthwise_conv2d_layer_t
 * @brief Represents a depthwise 2D convolutional layer.
 *
 * Each input channel is convolved with a separate filter.
 */
typedef struct {
    const fixed * weights;      /**< Weights for depthwise filters (channels * kernel_h * kernel_w) */
    const fixed * bias;         /**< Bias values per channel */
    uint16_t channels;          /**< Number of input channels (and filters) */
    size2d_t kernel_sz;         /**< Kernel size (height, width) */
    uint8_t padding;            /**< Padding type: PAD_SAME or PAD_VALID */
    size2d_t strides;           /**< Stride size (vertical, horizontal) */
} depthwise_conv2d_layer_t;


/**
 * @struct separable_conv2d_layer_t
 * @brief Represents a separable 2D convolutional layer.
 *
 * Composed of a depthwise convolution followed by a pointwise (1x1) convolution.
 */
typedef struct {
    uint16_t n_filters;             /**< Number of pointwise filters (output channels) */
    filter_t * point_filters;       /**< Array of 1x1 filters (pointwise) */
    uint16_t point_channels;        /**< Number of input channels for pointwise step */
    size2d_t point_kernel_sz;       /**< Kernel size for pointwise convolution (should be 1x1) */
    filter_t depth_filter;          /**< Depthwise filter (shared or per-channel) */
    uint16_t depth_channels;        /**< Number of input channels for depthwise step */
    size2d_t depth_kernel_sz;       /**< Kernel size for depthwise convolution */
    uint8_t padding;                /**< Padding type: PAD_SAME or PAD_VALID */
    size2d_t strides;               /**< Stride size for both steps */
} separable_conv2d_layer_t;


/**
 * @struct dense_layer_t
 * @brief Represents a fully connected (dense) layer.
 */
typedef struct {
    uint16_t input_size;        /**< Number of input neurons */
    uint16_t output_size;       /**< Number of output neurons */
    fixed *weights;             /**< Weight matrix [input_size][output_size] */
    fixed *biases;              /**< Bias vector [output_size] */
} dense_layer_t;


/**
 * @struct pooling2d_layer_t
 * @brief Configuration for 2D pooling layers (max or average).
 */
typedef struct {
    uint16_t size;      /**< Pooling window size (assumed square: size x size) */
    uint16_t strides;   /**< Stride of the pooling window */
} pooling2d_layer_t;

/** @} */ // end of layer_structures


/**
 * @defgroup normalization_structures Normalization Structures
 * @brief Structures used for various normalization techniques.
 * @{
 */

/**
 * @struct normalization_layer_t
 * @brief Generic normalization layer for element-wise normalization.
 *
 * Applies transformation: (x_i - sub_val[i]) / inv_div_val[i]
 *
 * Can be used for:
 * - Standard: (x - mean) / std_dev
 * - Min-Max: (x - min) / (max - min)
 * - Robust: (x - median) / (q3 - q1)
 */
typedef struct {
    const fixed *sub_val;         /**< Values to subtract (e.g., mean, min, median) */
    const fixed *inv_div_val;     /**< Inverse of divisor (e.g., 1/std_dev, 1/(max-min)) */
} normalization_layer_t;


/**
 * @struct batch_normalization_layer_t
 * @brief Parameters for batch normalization layer.
 *
 * Implements: output = (input - moving_mean) * moving_inv_std_dev + beta
 * Optimized as: output = input * moving_inv_std_dev + std_beta
 */
typedef struct {
    uint32_t length;                        /**< Number of channels (length of parameter vectors) */
    const fixed *moving_inv_std_dev;        /**< Precomputed: gamma / sqrt(variance + epsilon) */
    const fixed *std_beta;                  /**< Precomputed: beta - moving_mean * moving_inv_std_dev */
} batch_normalization_layer_t;

/** @} */ // end of normalization_structures


/**
 * @struct spectrogram_layer_t
 * @brief Configuration for spectrogram generation from audio signals.
 *
 * Groups all parameters needed to compute a mel-spectrogram from raw audio.
 */
typedef struct {
    uint16_t convert_to_db;     /**< Whether to convert magnitude to decibels */
    uint16_t n_fft;             /**< FFT size */
    uint16_t n_mels;            /**< Number of mel frequency bands */
    uint16_t frame_length;      /**< Length of each frame in samples */
    uint16_t sample_rate;       /**< Sampling rate of the audio signal */
    uint16_t n_blocks;          /**< Number of time frames (blocks) */
    uint16_t n_fft_table;       /**< Size of precomputed FFT sine/cosine table */
    uint16_t noverlap;          /**< Number of overlapping samples between frames */
    uint16_t step;              /**< Step size (hop length) between frames in samples */
    uint16_t len_nfft_nmels;    /**< Length of the averaging window from FFT to mel bins */
    uint16_t spec_size;         /**< Total size of the spectrogram (n_blocks * n_mels) */
    uint16_t ts_us;             /**< Time step between samples in microseconds */
} spectrogram_layer_t;


/**
 * @defgroup core_functions Core Layer Functions
 * @brief Fundamental operations for neural network inference.
 * @{
 */

/**
 * @brief Prepares internal memory buffers for model execution.
 *
 * This function should be called once at the beginning of the model's predict function.
 * It aligns temporary buffers used during layer computations, avoiding memory misalignment
 * issues due to the fixed memory allocation strategy.
 */
void prepare_buffers(void);


/**
 * @brief Applies a 2D convolutional layer without padding or striding.
 *
 * @param layer    Convolutional layer configuration
 * @param input    Input data (3D tensor: height × width × channels)
 * @param output   Pointer to output data structure
 */
void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output);


/**
 * @brief Applies a 2D convolutional layer with padding support.
 *
 * @param layer    Convolutional layer configuration (with padding field)
 * @param input    Input data (3D tensor)
 * @param output   Pointer to output data structure
 */
void conv2d_padding_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output);


/**
 * @brief Applies a 2D convolutional layer with striding (no padding).
 *
 * @param layer    Convolutional layer configuration (with strides)
 * @param input    Input data (3D tensor)
 * @param output   Pointer to output data structure
 */
void conv2d_strides_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output);


/**
 * @brief Applies a separable 2D convolutional layer.
 *
 * Performs depthwise convolution followed by pointwise (1x1) convolution.
 *
 * @param layer    Separable convolution layer configuration
 * @param input    Input data (3D tensor)
 * @param output   Pointer to output data structure
 */
void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t * output);


/**
 * @brief Applies a depthwise 2D convolutional layer.
 *
 * Each input channel is filtered independently.
 *
 * @param layer    Depthwise convolution layer configuration
 * @param input    Input data (3D tensor)
 * @param output   Pointer to output data structure
 */
void depthwise_conv2d_layer(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output);


/**
 * @brief Performs forward pass of a dense (fully connected) layer.
 *
 * @param dense_layer  Pointer to dense layer configuration
 * @param input        Pointer to input data (1D vector)
 * @param output       Pointer to output data structure (1D vector)
 */
void dense_layer(dense_layer_t* dense_layer, data1d_t* input, data1d_t * output);


/**
 * @brief Applies max pooling to a 2D input.
 *
 * Currently supports square pooling windows and strides. No padding support.
 *
 * @param pool     Pooling configuration (size and stride)
 * @param input    Input data (3D tensor)
 * @param output   Pointer to output data structure
 */
void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);


/**
 * @brief Applies average pooling to a 2D input.
 *
 * @param pool     Pooling configuration (size and stride)
 * @param input    Input data (3D tensor)
 * @param output   Pointer to output data structure
 */
void average_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);


/**
 * @brief Flattens a 3D tensor into a 1D vector.
 *
 * Used to convert convolutional layer outputs into a format suitable for dense layers.
 *
 * @param input    Input data (3D tensor)
 * @param output   Pointer to output data structure (1D vector)
 */
void flatten3d_layer(data3d_t input, data1d_t * output);

/** @} */ // end of core_functions


/**
 * @defgroup activation_functions Activation Functions
 * @brief Element-wise activation functions.
 * @{
 */

/**
 * @brief Applies softmax activation to a vector.
 *
 * @param data    Pointer to input/output data (modified in-place)
 * @param length  Number of elements
 */
void softmax_activation(fixed *data, uint32_t length);


/**
 * @brief Applies ReLU activation: max(0, x).
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void relu_activation(fixed *data, uint32_t length);


/**
 * @brief Applies ReLU6 activation: min(max(0, x), 6).
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void relu6_activation(fixed *data, uint32_t length);


/**
 * @brief Applies Leaky ReLU activation: x >= 0 ? x : alpha * x.
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 * @param alpha   Slope for negative values
 */
void leakyrelu_activation(fixed *data, uint32_t length, fixed alpha);


/**
 * @brief Applies tanh activation function.
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void tanh_activation(fixed *data, uint32_t length);


/**
 * @brief Applies sigmoid activation: 1 / (1 + exp(-x)).
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void sigmoid_activation(fixed *data, uint32_t length);


/**
 * @brief Applies softsign activation: x / (1 + |x|).
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void softsign_activation(fixed *data, uint32_t length);

/** @} */ // end of activation_functions


/**
 * @defgroup normalization_functions Normalization Functions
 * @brief Functions to apply various normalization techniques.
 * @{
 */

/**
 * @brief Applies generic normalization: (x_i - sub_val[i]) * inv_div_val[i]
 *
 * Used for standard, min-max, and robust normalization.
 *
 * @param s       Normalization parameters
 * @param input   Input data (1D)
 * @param output  Pointer to output data
 */
void normalization1(normalization_layer_t s, data1d_t input, data1d_t * output);

#define standard_norm_layer(norm, input, output) normalization1(norm, input, output)

#define min_max_norm_layer(norm, input, output) normalization1(norm, input, output)

#define robust_norm_layer(norm, input, output) normalization1(norm, input, output)


/* Normalization function for:
 *  abs_max_normalization   : (x_i)/(abs_max_xi)
 */
void normalization2(normalization_layer_t s, data1d_t input, data1d_t * output);

#define max_abs_norm_layer(norm, input, output) normalization2(norm, input, output)

/**
 * @brief Applies batch normalization to a 1D data array.
 *
 * @param norm    Batch normalization layer parameters
 * @param length  Length of the data array (number of channels)
 * @param data    Pointer to data (modified in-place)
 */
void batch_normalization_layer(batch_normalization_layer_t norm, uint32_t length, fixed *data);


/**
 * @brief Applies batch normalization to a 3D tensor (channel-wise).
 *
 * @param layer  Batch normalization parameters
 * @param data   Pointer to 3D data (modified in-place)
 */
void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data);


/**
 * @brief Applies batch normalization to a 1D data structure.
 *
 * @param layer  Batch normalization parameters
 * @param data   Pointer to 1D data (modified in-place)
 */
void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data);

/** @} */ // end of normalization_functions


/**
 * @defgroup reshaping_functions Reshaping and Transformation Functions
 * @brief Utility functions for data layout manipulation.
 * @{
 */

/**
 * @brief Applies zero-padding to a 2D input data array.
 *
 * @param pad_h   Number of zero rows to add at top and bottom
 * @param pad_w   Number of zero columns to add at left and right
 * @param input   Input data (3D tensor)
 * @param output  Pointer to output data structure with padded dimensions
 */
void zero_padding2d_layer(uint8_t pad_h, uint8_t pad_w, data3d_t input, data3d_t *output);


/**
 * @brief Adapts channel ordering from (H, W, C) to (C, H, W).
 *
 * Converts TensorFlow/Keras image format to EmbedIA internal format.
 * Required before the first convolutional layer in most models.
 *
 * @param input   Input data in (H, W, C) format
 * @param output  Pointer to output data in (C, H, W) format
 */
void channel_adapt_layer(data3d_t input, data3d_t * output);

/** @} */ // end of reshaping_functions


#ifdef __cplusplus
}
#endif

#endif /* NEURAL_NET_H */