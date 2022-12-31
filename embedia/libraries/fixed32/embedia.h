/* 
 * EmbedIA 
 * LIBRERÍA ARDUINO QUE DEFINE FUNCIONES PARA LA IMPLEMENTACIÓN DE REDES NEURNOALES CONVOLUCIONALES
 * EN MICROCONTROLADORES Y LAS ESTRUCTURAS DE DATOS NECESARIAS PARA SU USO
 */
#ifndef _EMBEDIA_H
#define _EMBEDIA_H

#include <stdint.h>
#include <math.h>
#include "fixed.h"

{includes}



/* DEFINICIÓN DE ESTRUCTURAS */

/*
 * Estructura que almacena una matriz de datos de tipo fixed  (fixed  * data) en forma de vector
 * Especifica la cantidad de canales (uint32_t channels), el ancho (uint32_t width) y el alto (uint32_t height) de la misma
 */
typedef struct{
    uint32_t channels;
    uint32_t width;
    uint32_t height;
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
 * typedef struct filter_t
 * Estructura que almacena los pesos de un filtro.
 * Especifica la cantidad de canales (uint32_t channels), su tamaño (uint32_t kernel_size),
 * los pesos (fixed  * weights) y el bias (fixed  bias).
 */
typedef struct{
    uint32_t channels;
    uint32_t kernel_size;
    const fixed  * weights;
    fixed  bias; 
}filter_t;

/*
 * typedef struct conv_layer_t
 * Estructura que modela una capa convolucional.
 * Especifica la cantidad de filtros (uint32_t n_filters) y un vector de filtros (filter_t * filters) 
 */
typedef struct{
    uint32_t n_filters;
    filter_t * filters; 
}conv2d_layer_t;

/*
 * typedef struct conv_layer_t
 * Estructura que modela una capa separable..
 * Especifica la cantidad de filtros (uint32_t n_filters,
 * un filtro del tamaño indicado (filter_t depth_filter) 
 * un vector de filtros 1x1 (filter_t * point_filters) 
 */
typedef struct{
    uint32_t n_filters;
    filter_t depth_filter;
    filter_t * point_filters; 
}separable_conv2d_layer_t;

/*
 * typdef struct neuron_t
 * Estructura que modela una neurona.
 * Especifica los pesos de la misma en forma de vector (fixed  * weights) y el bias (fixed  bias)
 */
typedef struct{
    const fixed  * weights;
    fixed  bias;
}neuron_t;

/*
 * typdef struct dense_layer_t
 * Estructura que modela una capa densa.
 * Especifica la cantidad de neuronas (uint32_t n_neurons) y un vector de neuronas (neuron_t * neurons) 
 */
typedef struct{
    uint32_t n_neurons;
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
 *  - standard normalization  : (x_i-mean_i)/std_dev_i
 *  - min_max normalization   : (x_i-min_i) / (max_i-min_i)
 *  - robust normalization    : (x_i-q1_i)  / (q3_i-q1_i)
 *  - abs_max_normalization   : (x_i)/(abs_max_xi)
 */

typedef struct{
    const fixed *sub_val;
    const fixed *inv_div_val;
} normalization_layer_t;


/* 
 * Structure for BatchNormalization layer.
 * Contains vectors for the four parameters used for normalization.
 * The number of each of the parameters is determined by the number of channels of the previous layer.
 */
typedef struct {
    uint32_t length;
    const fixed *beta;
    // const float *gamma; //  removed due to optimization included in moving_inv_std_dev
    const fixed *moving_mean;
    const fixed *moving_inv_std_dev; // = gamma / sqrt(moving_variance + epsilon)
} batch_normalization_layer_t;



/* LIBRARY FUNCTIONS PROTOTYPES */

/* 
 * conv2d_layer()
 * Función que se encarga de aplicar la convolución de una capa de filtros (conv_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros:
 *          conv_layer_t layer  =>  capa convolucional con filtros cargados
 *         	      data3d_t input  =>  datos de entrada de tipo data3d_t
 *             data3d_t * output  =>  puntero a la estructura data3d_t donde se guardará el resultado
 */
void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output);

/* 
 * separable_conv2d_layer()
 *  Function in charge of applying the convolution of a filter layer (conv_layer_t) on a given input data set.
 * 
 * Parameters:
 *  layer => convolutional layer with loaded filters.
 *  input => input data of type data3d_t
 *  *output => pointer to the data3d_t structure where the result will be saved.
 */
void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t * output);


/* 
 * dense_layer()
 * Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *   - dense_layer => structure with the weights of the neurons of the dense layer.  
 *   - input       => structure data1d_t with the input data to process. 
 *   - *output     => structure data1d_t to store the output result.
 */
void dense_layer(dense_layer_t dense_layer, data1d_t input, data1d_t * output);

/* 
 * max_pooling2d_layer()
 * Maxpooling layer, for now supports square size and stride. No support for padding 
 * Parameters:
 *   - pool_size => size for pooling
 *   - stride    => stride for pooling
 *   - input     => input data 
 *   - output    => output data
 */
void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);

/* 
 * avg_pooling_2d()
 *  Function that applies an average pooling to an input with a window size of received 
 *  by parameter (uint16_t strides)
 *
 * Parameters:
 *  input => input data of type data3d_t.
 *  *output => pointer to the data3d_t structure where the result will be stored.
 */
void avg_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);


/* 
 * flatten3d_layer()
 * Performs a variable shape change. 
 * Converts the data format from data3d_t array format to data1d_t vector.
 * (prepares data for input into a layer of type dense_layer_t).
 * Parameters:
 *   input => input data of type data3d_t.
 *   *output => pointer to the data1d_t structure where the result will be stored.
 */
 void flatten3d_layer(data3d_t input, data1d_t * output);
     
/* 
 * argmax()
 *  Finds the index of the largest value within a vector of data (data1d_t)
 * 
 * Parameters:
 *  data => data of type data1d_t to search for max.
 *
 * Returns:
 *  search result - index of the maximum value
 */
uint32_t argmax(data1d_t data);


/***************************************************************************************************************************/
/* Activation functions/layers */

void softmax_activation(fixed *data, uint32_t length);

void relu_activation(fixed *data, uint32_t length);

void tanh_activation(fixed *data, uint32_t length);

void sigmoid_activation(fixed *data, uint32_t length);

void softsign_activation(fixed *data, uint32_t length);



/***************************************************************************************************************************/
/* Normalization layers */

/* Normalization function for:
 *  - standard normalization  : (x_i-mean_i)/std_dev_i
 *  - min_max normalization   : (x_i-min_i) / (max_i-min_i)
 *  - robust normalization    : (x_i-q1_i)  / (q3_i-q1_i)
 */
void normalization1(normalization_layer_t s, data1d_t input, data1d_t * output);

#define standard_norm_layer(norm, input, output) normalization1(norm, input, output)

#define min_max_norm_layer(norm, input, output) normalization1(norm, input, output)

#define robust_norm_layer(norm, input, output) normalization1(norm, input, output)


/* Normalization function for:
 *  - abs_max_normalization   : (x_i)/(abs_max_xi)
 */
void normalization2(normalization_layer_t s, data1d_t input, data1d_t * output);

#define max_abs_norm_layer(norm, input, output) normalization2(norm, input, output)


void batch_normalization_layer(batch_normalization_layer_t norm, uint32_t length, float *data);


void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data);

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data);


/* Tranformation Layers
 *
 */

/* Converts Tensorflow/Keras Image (Height, Width, Channel) to Embedia format (Channel, Height, Width).
   Usually required for first convolutional layer
*/
void image_adapt_layer(data3d_t input, data3d_t * output);


#endif