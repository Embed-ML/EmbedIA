/* 
 * EmbedIA 
 * C LIBRARY FOR THE IMPLEMENTATION OF NEURAL NETWORKS ON MICROCONTROLLERS
 */
#ifndef _EMBEDIA_H
#define _EMBEDIA_H

#include <stdint.h>
#include <math.h>
#include "fixed.h"

#include <stdlib.h>


#define PRINT_RESULTS 0

/* STRUCTURE DEFINITION */

/*
 * Structure that stores an array of fixed data (fixed * data) in vector form.
 * Specifies the number of channels, the width and the height of the array.
 */

typedef struct{
    uint16_t channels;
    uint16_t width;
    uint16_t height;
    fixed  * data;
}data3d_t;

typedef struct{
    uint16_t width;
    uint16_t height;
    fixed  * data;
}data2d_t;

typedef struct{
    uint32_t length;
    fixed  * data;
}data1d_t;


/*
 * Structure that stores the weights of a filter.
 Specifies the number of channels (uint16_t channels), their size (uint16_t kernel_size), the weights (fixed * weights) and the bias (fixed bias),
 * the weights (fixed * weights) and the bias (fixed bias).
 */
typedef struct{
    uint16_t channels;
    uint16_t kernel_size;
    const fixed  * weights;
    fixed  bias; 
}filter_t;

/*
 * Structure that models a convolutional layer.
 * Specifies the number of filters (uint16_t n_filters) and a vector of filters (filter_t * filters). 
 */
typedef struct{
    uint16_t n_filters;
    filter_t * filters; 
}conv2d_layer_t;

/*
 * Structure that models a separable layer...
 * Specifies the number of filters (uint16_t n_filters,
 * a filter of the specified size (filter_t depth_filter) 
 * a vector of 1x1 filters (filter_t * point_filters) 
 */
typedef struct{
    uint16_t n_filters;
    filter_t depth_filter;
    filter_t * point_filters; 
}separable_conv2d_layer_t;

/*
 * Structure that models a neuron.
 * Specifies the weights of the neuron as a vector (fixed * weights) and the bias (fixed bias).
 */
typedef struct{
    const fixed  * weights;
    fixed  bias;
}neuron_t;

/*
 * Structure that models a dense layer.
 * Specifies the number of neurons (uint16_t n_neurons) and a vector of neurons (neuron_t * neurons). 
 */
typedef struct{
    uint16_t n_neurons;
    neuron_t * neurons;
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
    const fixed *sub_val;
    const fixed *inv_div_val;
} normalization_layer_t;


/* 
 * Structure for BatchNormalization layer.
 * Contains vectors for the two parameters used for normalization.
 * The number of each of the parameters is determined by the number of channels of the previous layer.
 */
typedef struct {
    uint32_t length;
    const fixed *moving_inv_std_dev; // = gamma / sqrt(moving_variance + epsilon)
    const fixed *std_beta;           // = beta - moving_mean * moving_inv_std_dev 
} batch_normalization_layer_t;

/*
 * spectrogram_layer_t struct
 * 
 * Defines the necessary configuration to generate a spectrogram from an 
 * audio signal. It groups all the key parameters into a single data type 
 * to facilitate passing to the processing functions.
 * 
 * - convert_to_db: indicates if the output should be converted to decibels.
 * - n_fft: FFT size.
 * - n_mels: number of mel bands.
 * - frame_length: frame length in samples. 
 * - sample_rate: sampling rate.
 * - n_blocks: number of frames.
 * - n_fft_table: 
 * - noverlap: overlap between frames.
 * - step: indicates the jump or advance in samples between each frame.
 * - len_nfft_nmels: length of the range over which the FFT values ​​are averaged to map to each mel band.
 * - spec_size: total spectrogram size in samples.
 * - ts_us: time step in microseconds between samples.
 */
typedef struct {
    uint16_t convert_to_db;
    uint16_t n_fft;
    uint16_t n_mels;
    uint16_t frame_length;
    uint16_t sample_rate;
    uint16_t n_blocks;
    uint16_t n_fft_table;
    uint16_t noverlap;
    uint16_t step;
    uint16_t len_nfft_nmels;
    uint16_t spec_size;
    uint16_t ts_us;
} spectrogram_layer_t;


/* LIBRARY FUNCTIONS PROTOTYPES */


/*
 * prepare_buffers()
 *  This function should be invoked only at the beginning of the predict function of the model file.
 * Its purpose is to align the exchange buffers used by the different functions of the model. Due to
 * the allocation strategy that never frees the memory, it happens that if the swap_alloc function
 * is invoked an odd number of times in the 2nd invocation the predict reserves more memory than
 * necessary  (something that usually happens with convolutional layers)
 */
void prepare_buffers();


/*
 * conv2d_layer()
 *  Function in charge of applying the convolution of a filter layer (conv_layer_t) on a given input data set.
 * Parameters:
 *  layer => convolutional layer with loaded filters.
 *  input => input data of type data3d_t
 *  *output => pointer to the data3d_t structure where the result will be saved.
 */
void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output);

/* 
 * separable_conv2d_layer()
 *  Function in charge of applying the convolution of a filter layer (conv_layer_t) on a given input data set.
 * 
 * Parameters:
 *   layer => convolutional layer with loaded filters.
 *   input => input data of type data3d_t
 *   *output => pointer to the data3d_t structure where the result will be saved.
 */
void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t * output);


/* 
 * dense_layer()
 * Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *   dense_layer => structure with the weights of the neurons of the dense layer.  
 *   input       => structure data1d_t with the input data to process. 
 *   *output     => structure data1d_t to store the output result.
 */
void dense_layer(dense_layer_t dense_layer, data1d_t input, data1d_t * output);

/* 
 * max_pooling2d_layer()
 * Maxpooling layer, for now supports square size and stride. No support for padding 
 * Parameters:
 *   pool_size => size for pooling
 *   stride    => stride for pooling
 *   input     => input data 
 *   output    => output data
 */
void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);

/* 
 * avg_pooling_2d()
 * Function that applies an average pooling to an input with a window size of received 
 * by parameter (uint16_t strides)
 *
 * Parameters:
 *   input => input data of type data3d_t.
 *   *output => pointer to the data3d_t structure where the result will be stored.
 */
void avg_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);


/* 
 * flatten3d_layer()
 * Performs a variable shape change. 
 * Converts the data format from data3d_t array format to data1d_t vector.
 * (prepares data for input into a layer of type dense_layer_t).
 * Parameters:
 *    input => input data of type data3d_t.
 *    *output => pointer to the data1d_t structure where the result will be stored.
 */
 void flatten3d_layer(data3d_t input, data1d_t * output);
     
/* 
 * argmax()
 * Finds the index of the largest value within a vector of data (data1d_t)
 * Parameters:
 *   data => data of type data1d_t to search for max.
 *
 * Returns:
 *  search result - index of the maximum value
 */
uint32_t argmax(data1d_t data);


/***************************************************************************************************************************/
/* Activation functions/layers */

void softmax_activation(fixed *data, uint32_t length);

void relu_activation(fixed *data, uint32_t length);

void leakyrelu_activation(fixed *data, uint32_t length, fixed alpha);

void tanh_activation(fixed *data, uint32_t length);

void sigmoid_activation(fixed *data, uint32_t length);

void softsign_activation(fixed *data, uint32_t length);



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


void batch_normalization_layer(batch_normalization_layer_t norm, uint32_t length, fixed *data);


void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data);

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data);


/* Tranformation Layers
 *
 */

/* Converts Tensorflow/Keras Image (Height, Width, Channel) to Embedia format (Channel, Height, Width).
   Usually required for first convolutional layer
*/
void image_adapt_layer(data3d_t input, data3d_t * output);

/* Signal processing */

/* 
 * void fft(float data_re[], float data_im[], const unsigned int N)
 * Performs a Fast Fourier Transform (FFT) on the complex data passed as parameters.
 * Parameters:
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data. 
 *   - N: Amount of samples to perform the FFT over.
 */
void fft(float data_re[], float data_im[],const unsigned int N);

/*
 * void rearrange(float data_re[], float data_im[], const unsigned int N)   
 * Performs the necessary reordering of the data before applying the FFT.
 * Parameters:
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data. 
 *   - N: Amount of samples to perform the FFT over.
 */
void rearrange(float data_re[],float data_im[],const unsigned int N);

/*
 * void compute(float data_re[], float data_im[], const unsigned int N)
 * Contains the FFT calculation core, applying the Fourier transforms for 
 * each recursive step.
 * Parameters: 
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data. 
 *   - N: Amount of samples to perform the FFT over.
 */
void compute(float data_re[],float data_im[],const unsigned int N);

/*
 * void create_spectrogram(spectrogram_layer_t config, data1d_t input, data3d_t *output)
 * Generates the spectrogram from the input signal by applying FFTs
 * and further processing.
 * Parameters:
 *   - config: Spectrogram layer configuration
 *   - input:  1D input signal  
 *   - output: 3D output spectrogram (W = n_mels, H = b_blocks, Ch = 1)
 */
void create_spectrogram(spectrogram_layer_t config, data1d_t input, data3d_t * output);


#endif