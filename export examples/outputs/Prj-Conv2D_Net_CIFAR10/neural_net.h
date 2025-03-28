/*
 * EmbedIA
 * C LIBRARY FOR THE IMPLEMENTATION OF NEURAL NETWORKS ON MICROCONTROLLERS
 */
#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H

#include <stdint.h>
#include "common.h"
#include "quant8.h"

//#include <stdlib.h>


/* STRUCTURE DEFINITION */

/*
 * Structure that stores the weights of a filter.
 Specifies the number of channels (uint16_t channels), their size (uint16_t kernel_size), the weights (quant8 * weights) and the bias (float bias),
 * the weights (quant8 * weights) and the bias (quant8 bias).
 */
typedef struct{
    const quant8  * weights;
    quant8  bias;
}filter_t;


#define PAD_SAME 1
#define PAD_VALID 0

typedef struct{
    uint16_t n_filters;
    filter_t * filters;
    uint16_t channels;
    size2d_t kernel;
    uint8_t padding;
    size2d_t strides;
    qparam_t qparam;
}conv2d_layer_t;

/*
 * Structure that models a depthwise convolutional 2d layer.
 * Specifies the number of channels, kernel size, vector of weights and bias.
 */
typedef struct{
    const quant8  * weights;
    const quant8  * bias;
    uint16_t channels;
    size2d_t kernel_sz;
    uint8_t padding;
    size2d_t strides;
    qparam_t w_qparam;
    qparam_t b_qparam;
}depthwise_conv2d_layer_t;

/*
 * Structure that models a separable layer...
 * Specifies the number of filters (uint16_t n_filters,
 * a filter of the specified size (filter_t depth_filter)
 * a vector of 1x1 filters (filter_t * point_filters)
 */
typedef struct{
    uint16_t n_filters;
    filter_t * point_filters;
    uint16_t point_channels;
    size2d_t point_kernel_sz;
    filter_t depth_filter;
    uint16_t depth_channels;
    size2d_t depth_kernel_sz;
    uint8_t padding;
    size2d_t strides;
    qparam_t qparam;
}separable_conv2d_layer_t;

/*
 * Structure that models a neuron.
 * Specifies the weights of the neuron as a vector (fixed * weights) and the bias (fixed bias).
 */
typedef struct{
    const quant8  * weights;
    quant8  bias;
    qparam_t  qparam;
}neuron_t;

/*
 * Structure that models a dense layer.
 * Specifies the number of neurons (uint16_t n_neurons) and a vector of neurons (neuron_t * neurons).
 */
typedef struct{
    uint16_t n_neurons;
    neuron_t * neurons;
    qparam_t  qparam;
}dense_layer_t;


/*
 * Pooling Structure
 */

typedef struct{
    uint16_t size;
    uint16_t strides;
} pooling2d_layer_t;

/*********************************************************************************************************************************/

/* structure for normalization type (x_i-s_i)/d_i and (x_i)/d_i
 *  standard normalization  : (x_i-mean_i)/std_dev_i
 *  min_max normalization   : (x_i-min_i) / (max_i-min_i)
 *  robust normalization    : (x_i-q2_i)  / (q3_i-q1_i)
 *  abs_max_normalization   : (x_i)/(abs_max_xi)
 */

typedef struct{
    const float *sub_val;
    const float *inv_div_val;
} normalization_layer_t;


/*
 * Structure for BatchNormalization layer.
 * Contains vectors for the two parameters used for normalization.
 * The number of each of the parameters is determined by the number of channels of the previous layer.
 */
typedef struct {
    uint32_t length;
    const quant8 *moving_inv_std_dev; // = gamma / sqrt(moving_variance + epsilon)
    const quant8 *std_beta;           // = beta - moving_mean * moving_inv_std_dev
    qparam_t  mov_qparam;
    qparam_t  std_qparam;
} batch_normalization_layer_t;



/* LIBRARY FUNCTIONS PROTOTYPES */


/*
 * prepare_buffers()
 *   This function should be invoked only at the beginning of the predict function of the model file.
 *   Its purpose is to align the exchange buffers used by the different functions of the model. Due to
 *   the allocation strategy that never frees the memory, it happens that if the swap_alloc function
 *   is invoked an odd number of times in the 2nd invocation the predict reserves more memory than
 *   necessary  (something that usually happens with convolutional layers)
 */
void prepare_buffers();


/*
 * conv2d_layer()
 *   Function in charge of applying the convolution of a filter layer (conv_layer_t) without padding and strides
 *   on a given input data set.
 * Parameters:
 *  - layer => convolutional layer with loaded filters.
 *  - input => input data of type data3d_t
 *  - *output => pointer to the data3d_t structure where the result will be saved.
 */
void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output);

/* variant function with padding and strides */
void conv2d_padding_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output);

/* variant function with strides without padding*/
void conv2d_strides_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output);


/*
 * separable_conv2d_layer()
 *   Function in charge of applying the convolution of a filter layer (conv_layer_t) on a given input data set.
 *
 * Parameters:
 *  - layer => convolutional layer with loaded filters.
 *  - input => input data of type data3d_t
 *  - *output => pointer to the data3d_t structure where the result will be saved.
 */
void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t * output);


/*
 * depthwise_conv2d_layer()
 *   Function in charge of applying the depthwise of a filter layer with bias (depthwise_conv2d_layer_t) on a given input data set.
 * Parameters:
 * - layer => depthwise layer with loaded filters.
 * - input => input data of type data3d_t
 * - *output => pointer to the data3d_t structure where the result will be saved.
 */

void depthwise_conv2d_layer(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output);

/*
 * dense_layer()
 *   Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *  - dense_layer => structure with the weights of the neurons of the dense layer.
 *  - input       => structure data1d_t with the input data to process.
 *  - *output     => structure data1d_t to store the output result.
 */
void dense_layer(dense_layer_t dense_layer, data1d_t input, data1d_t * output);

/*
 * max_pooling2d_layer()
 *   Maxpooling layer, for now supports square size and stride. No support for padding
 * Parameters:
 *  - pool_size => size for pooling
 *  - stride    => stride for pooling
 *  - input     => input data
 *  - output    => output data
 */
void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);

/*
 * avg_pooling_2d()
 *   Function that applies an average pooling to an input with a window size of received
 *   by parameter (uint16_t strides)
 * Parameters:
 *  - input => input data of type data3d_t.
 *  - *output => pointer to the data3d_t structure where the result will be stored.
 */
void avg_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);


/*
 * flatten3d_layer()
 * Performs a variable shape change.
 * Converts the data format from data3d_t array format to data1d_t vector.
 * (prepares data for input into a layer of type dense_layer_t).
 * Parameters:
 *  -  input => input data of type data3d_t.
 *  -  *output => pointer to the data1d_t structure where the result will be stored.
 */
 void flatten3d_layer(data3d_t input, data1d_t * output);

/*
 * argmax()
 * Finds the index of the largest value within a vector of data (data1d_t)
 * Parameters:
 *  - data => data of type data1d_t to search for max.
 * Returns:
 *  - search result - index of the maximum value
 */
uint32_t argmax(data1d_t data);


/***************************************************************************************************************************/
/* Activation functions/layers */

void softmax_activation(float *data, uint32_t length);

void relu_activation(float *data, uint32_t length);

void leakyrelu_activation(float *data, uint32_t length, float alpha);

void tanh_activation(float *data, uint32_t length);

void sigmoid_activation(float *data, uint32_t length);

void softsign_activation(float *data, uint32_t length);



/***************************************************************************************************************************/
/* Normalization layers */

/* Normalization function for:
 *  standard normalization  : (x_i-mean_i)/std_dev_i
 *  min_max normalization   : (x_i-min_i) / (max_i-min_i)
 *  robust normalization    : (x_i-q2_i)  / (q3_i-q1_i)
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

/* Batch normalization */
void batch_normalization_layer(batch_normalization_layer_t norm, uint32_t length, float *data);

void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data);

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data);


/* Rashaping Layers */

/* void zero_padding2d_layer(uint8_t pad_h, uint8_t pad_w, data3d_t input, data3d_t *output)
 * Applies zero-padding to a 2D input data array.
 * Parameters:
 *  - pad_h: Number of zero-padding rows to add at the top and bottom.
 *  - pad_w: Number of zero-padding columns to add at the left and right.
 *  - input: 3D data structure representing the input data.
 *  - output: Pointer to a 3D data structure where the zero-padded output will be stored.
 * Description:
 *   This function performs zero-padding on a 2D input data array. It adds the specified
 *   number of zero rows at the top and bottom (pad_h) and zero columns at the left and right (pad_w).
 *   The result is stored in the output data structure.
 */
void zero_padding2d_layer(uint8_t pad_h, uint8_t pad_w, data3d_t input, data3d_t *output);


/* Tranformation Layers */

/*  Converts Tensorflow/Keras Image (Height, Width, Channel) to Embedia format (Channel, Height, Width).
   Usually required for first convolutional layer
*/
void channel_adapt_layer(data3d_t input, data3d_t * output);


#endif