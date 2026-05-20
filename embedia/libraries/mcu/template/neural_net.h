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
 * It supports convolutional, dense, pooling, normalization, and activation layers using floating-point arithmetic.
 *
 */


#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "common.h"
//#include "normalization.h"

//{includes}

/**
 * @defgroup layer_structures Layer Structures
 * @brief Definitions of layer data structures used in neural network models.
 * @{
 */


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


// Types: layer types are defined in specific data type header (small differences like quantization parameters o types).
// @embedia-include neural_net/_types.h


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
 * issues due to the real_t memory allocation strategy.
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
 * @brief Applies a 1D convolutional layer without padding or striding.
 *
 * @param layer    Convolutional layer configuration
 * @param input    Input data (2D tensor: width × channels)
 * @param output   Pointer to output data structure
 */
void conv1d_layer(conv1d_layer_t layer, data2d_t input, data2d_t * output);

/**
 * @brief Applies a 1D convolutional layer with padding support.
 *
 * @param layer    Convolutional layer configuration (with padding field)
 * @param input    Input data (2D tensor: width × channels)
 * @param output   Pointer to output data structure
 */
void conv1d_padding_layer(conv1d_layer_t layer, data2d_t input, data2d_t * output);

/**
 * @brief Applies a 1D convolutional layer with striding (no padding).
 *
 * @param layer    Convolutional layer configuration (with strides)
 * @param input    Input data (2D tensor: width × channels)
 * @param output   Pointer to output data structure
 */
void conv1d_strides_layer(conv1d_layer_t layer, data2d_t input, data2d_t * output);

/**
 * @brief Applies a 1D convolutional layer for single-channel signals.
 *
 * @param layer    Convolutional layer configuration
 * @param input    Input data (1D tensor: single-channel signal)
 * @param output   Pointer to output data structure
 */


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
 * @defgroup local_pooling Local Pooling
 * @brief pooling functions.
 * @{
 */

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
 * @brief Applies max pooling to a 1D input.
 *
 * Currently supports any pooling window size and stride. No padding support.
 *
 * @param pool     Pooling configuration (size and stride)
 * @param input    Input data (2D tensor: channels × width)
 * @param output   Pointer to output data structure
 */
void max_pooling1d_layer(pooling1d_layer_t pool, data2d_t input, data2d_t* output);

/**
 * @brief Applies average pooling to a 1D input.
 *
 * @param pool     Pooling configuration (size and stride)
 * @param input    Input data (2D tensor: channels × width)
 * @param output   Pointer to output data structure
 */
void average_pooling1d_layer(pooling1d_layer_t pool, data2d_t input, data2d_t* output);


/** @} */ // end of local_pooling



/**
 * @defgroup global_pooling Global Pooling
 * @brief global pooling functions.
 * @{
 */

/**
 * @brief Global Max Pooling for 2D data
 * Reduces (C, H, W) to (C) by taking max over H×W dimensions
 */
void global_max_pooling2d_layer(data3d_t input, data1d_t* output);


/**
 * @brief Global Average Pooling for 2D data
 * Reduces (C, H, W) to (C) by averaging over H×W dimensions
 */
void global_average_pooling2d_layer(data3d_t input, data1d_t* output);


/**
 * @brief Global Max Pooling for 1D data
 * Takes maximum value along spatial dimensions for each channel
 */
void global_max_pooling1d_layer(data2d_t input, data1d_t* output);


/**
 * @brief Global Average Pooling for 1D data
 * Reduces spatial dimensions to single value per channel by averaging
 * No parameters needed - operates over entire input dimensions
 */
void global_average_pooling1d_layer(data2d_t input, data1d_t* output);


/** @} */ // end of global_pooling


/**
 * @brief Flattens a 3D tensor into a 1D vector.
 *
 * Used to convert convolutional layer outputs into a format suitable for dense layers.
 *
 * @param input    Input data (3D tensor)
 * @param output   Pointer to output data structure (1D vector)
 */
void flatten3d_layer(data3d_t input, data1d_t * output);


/**
 * @brief Flattens a 2D tensor into a 1D vector.
 *
 * Used to convert 2D layer outputs (like from 1D convolutions) into a format
 * suitable for dense layers. Preserves the order: channels first, then width.
 *
 * @param input    Input data (2D tensor: channels × width)
 * @param output   Pointer to output data structure (1D vector)
 */
void flatten2d_layer(data2d_t input, data1d_t * output);


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
void softmax_activation(real_t *data, uint32_t length);


/**
 * @brief Applies ReLU activation: max(0, x).
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void relu_activation(real_t *data, uint32_t length);


/**
 * @brief Applies ReLU6 activation: min(max(0, x), 6).
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void relu6_activation(real_t *data, uint32_t length);

void softmax_activation(real_t *data, uint32_t length);

/**
 * @brief Applies Leaky ReLU activation: x >= 0 ? x : alpha * x.
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 * @param alpha   Slope for negative values
 */
void leakyrelu_activation(real_t *data, uint32_t length, real_t alpha);


/**
 * @brief Applies tanh activation function.
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void tanh_activation(real_t *data, uint32_t length);


/**
 * @brief Applies sigmoid activation: 1 / (1 + exp(-x)).
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void sigmoid_activation(real_t *data, uint32_t length);


/**
 * @brief Applies softsign activation: x / (1 + |x|).
 *
 * @param data    Pointer to input/output data
 * @param length  Number of elements
 */
void softsign_activation(real_t *data, uint32_t length);

/** @} */ // end of activation_functions


/**
 * @defgroup normalization_functions Normalization Functions
 * @brief Functions to apply various normalization techniques.
 * @{
 */


/**
 * @brief Applies batch normalization to a 1D data array.
 *
 * @param norm    Batch normalization layer parameters
 * @param length  Length of the data array (number of channels)
 * @param data    Pointer to data (modified in-place)
 */
void batch_normalization_layer(batch_normalization_layer_t norm, uint32_t length, real_t *data);


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
void channel_adapt_layer_3d(data3d_t input, data3d_t * output);

/**
 * @brief Adapts channel ordering from interleaved to consecutive format for 1D data.
 *
 * Converts from time-major interleaved format [T0_C0, T0_C1, T1_C0, T1_C1, ...]
 * to channel-major consecutive format [C0_T0, C0_T1, ..., C1_T0, C1_T1, ...].
 * Required when input data format doesn't match convolution layer expectations.
 *
 * @param input   Input data in interleaved format (time, channels)
 * @param output  Pointer to output data in consecutive format (channels, time)
 */
void channel_adapt_layer_2d(data2d_t input, data2d_t * output);


/** @} */ // end of reshaping_functions


#ifdef __cplusplus
}
#endif

#endif /* NEURAL_NET_H */