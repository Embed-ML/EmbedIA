/* 
 * EmbedIA 
 * LIBRERÍA ARDUINO QUE DEFINE FUNCIONES PARA LA IMPLEMENTACIÓN DE REDES NEURNOALES CONVOLUCIONALES
 * EN MICROCONTROLADORES Y LAS ESTRUCTURAS DE DATOS NECESARIAS PARA SU USO
 */
#ifndef _EMBEDIA_H
#define _EMBEDIA_H

#include <stdint.h>
#include <math.h>
#include <stdlib.h>



/* DEFINICIÓN DE ESTRUCTURAS */

/*
 * typedef struct data3d_t
 * Estructura que almacena una matriz de datos de tipo float  (float  * data) en forma de vector
 * Especifica la cantidad de canales (uint16_t channels), el ancho (uint16_t width) y el alto (uint16_t height) de la misma
 */
/*typedef struct{
    uint16_t channels;
    uint16_t width;
    uint16_t height;
    float  * data;
}data3d_t;
*/
typedef struct{
    uint16_t channels;
    uint16_t width;
    uint16_t height;
    float  * data;
}data3d_t;

typedef struct{
    uint16_t width;
    uint16_t height;
    float  * data;
}data2d_t;

typedef struct{
    uint16_t length;
    float  * data;
}data1d_t;

/*
 * typdef struct data1d_t
 * Estructura que almacena un vector de datos de tipo float  (float  * data).
 * Especifica el largo del mismo (uint16_t length).
 */
 /*
typedef struct{
    uint16_t length;
    float  * data;
}data1d_t;
*/


/*
 * typedef struct filter_t
 * Estructura que almacena los pesos de un filtro.
 * Especifica la cantidad de canales (uint16_t channels), su tamaño (uint16_t kernel_size),
 * los pesos (float  * weights) y el bias (float  bias).
 */
typedef struct{
    uint16_t channels;
    uint16_t kernel_size;
    float  * weights;
    float  bias; 
}filter_t;

/*
 * typedef struct conv_layer_t
 * Estructura que modela una capa convolucional.
 * Especifica la cantidad de filtros (uint16_t n_filters) y un vector de filtros (filter_t * filters) 
 */
typedef struct{
    uint16_t n_filters;
    filter_t * filters; 
}conv2d_layer_t;

/*
 * typedef struct conv_layer_t
 * Estructura que modela una capa separable..
 * Especifica la cantidad de filtros (uint16_t n_filters,
 * un filtro del tamaño indicado (filter_t depth_filter) 
 * un vector de filtros 1x1 (filter_t * point_filters) 
 */
typedef struct{
    uint16_t n_filters;
    filter_t depth_filter;
    filter_t * point_filters; 
}separable_layer_t;

/*
 * typdef struct neuron_t
 * Estructura que modela una neurona.
 * Especifica los pesos de la misma en forma de vector (float  * weights) y el bias (float  bias)
 */
typedef struct{
    float  * weights;
    float  bias;
}neuron_t;

/*
 * typdef struct dense_layer_t
 * Estructura que modela una capa densa.
 * Especifica la cantidad de neuronas (uint16_t n_neurons) y un vector de neuronas (neuron_t * neurons) 
 */
typedef struct{
    uint16_t n_neurons;
    neuron_t * neurons;
}dense_layer_t;


/*
 * 
 */
 
typedef struct{
    uint16_t size;
    uint16_t strides;
} pooling2d_t;

/*********************************************************************************************************************************/

/* structure for normalization type (x_i-s_i)/d_i and (x_i)/d_i
 *  - standard normalization  : (x_i-mean_i)/std_dev_i
 *  - min_max normalization   : (x_i-min_i) / (max_i-min_i)
 *  - robust normalization    : (x_i-q1_i)  / (q3_i-q1_i)
 *  - abs_max_normalization   : (x_i)/(abs_max_xi)
 */

typedef struct{
    const float *sub_val;
    const float *inv_div_val;
} normalization_t;




/* PROTOTIPOS DE FUNCIONES DE LA LIBRERÍA */

/* 
 * conv2d()
 * Función que realiza la convolución entre un filtro y un conjunto de datos.
 * Parámetros:
 *             filter_t filter  =>  estructura filtro con pesos cargados
 *                data3d_t input  =>  datos de entrada de tipo data3d_t
 *             data3d_t * output  =>  puntero a la estructura data3d_t donde se guardará el resultado
 * 				     uint16_t delta	=>  posicionamiento de feature_map dentro de output.data
 */
void conv2d(filter_t filter, data3d_t input, data3d_t * output, uint16_t delta);

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
 * Función que se encarga de aplicar la convolución de una capa de filtros (separable_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros:
 *     separable_layer_t layer  =>  capa convolucional separable con filtros cargados
 *                data3d_t input  =>  datos de entrada de tipo data3d_t
 *             data3d_t * output  =>  puntero a la estructura data3d_t donde se guardará el resultado
 */
void separable_conv2d_layer(separable_layer_t layer, data3d_t input, data3d_t * output);

/* 
 * neuron_forward()
 * Función que realiza el forward de una neurona ante determinado conjunto de datos de entrada.
 * Parámetros:
 *             neuron_t neuron  =>  neurona con sus pesos y bias cargados
 *        data1d_t input  =>  datos de entrada en forma de vector (data1d_t)
 * Retorna:
 *                      float   =>  resultado de la operación             
 */
float neuron_forward(neuron_t neuron, data1d_t input);

/* 
 * dense_layer()
 * Función que se encarga de realizar el forward de una capa densa (dense_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros
 *          dense_layer_t dense_layer  =>  capa densa con neuronas cargadas
 *               data1d_t input  =>  datos de entrada de tipo data1d_t
 *            data1d_t * output  =>  puntero a la estructura data1d_t donde se guardará el resultado
 */
void dense_layer(dense_layer_t dense_layer, data1d_t input, data1d_t * output);

/* 
 * max_pooling2d()
 * Función que se encargará de aplicar un max pooling a una entrada
 * con un tamaño de ventana de recibido por parámetro (uint16_t strides)
 * a un determinado conjunto de datos de entrada.
 * Parámetros:
 *                data3d_t input  =>  datos de entrada de tipo data3d_t
 *             data3d_t * output  =>  puntero a la estructura data3d_t donde se guardará el resultado
 */
void max_pooling2d_layer(pooling2d_t pool, data3d_t input, data3d_t* output);

/* 
 * avg_pooling_2d()
 * Función que se encargará de aplicar un average pooling a una entrada
 * con un tamaño de ventana de recibido por parámetro (uint16_t strides)
 * a un determinado conjunto de datos de entrada.
 * Parámetros:
 *                data3d_t input  =>  datos de entrada de tipo data3d_t
 *             data3d_t * output  =>  puntero a la estructura data3d_t donde se guardará el resultado
 */
void avg_pooling2d_layer(pooling2d_t pool, data3d_t input, data3d_t* output);

//void avg_pooling_2d(uint16_t pool_size, uint16_t strides, data3d_t input, data3d_t* output);



/* 
 * flatten_layer()
 * Realiza un cambio de tipo de variable. 
 * Convierte el formato de los datos en formato de matriz data3d_t en vector data1d_t.
 * (prepara los datos para ingresar en una capa de tipo dense_layer_t)
 * Parámetros:
 *                data3d_t input  =>  datos de entrada de tipo data3d_t
 *     data1d_t * output  =>  puntero a la estructura data1d_t donde se guardará el resultado
 */
void flatten3d_layer(data3d_t input, data1d_t * output);

/* 
 * argmax()
 * Busca el indice del valor mayor dentro de un vector de datos (data1d_t)
 * Parámetros:
 *         data1d_t data  =>  datos de tipo data1d_t a buscar máximo
 * Retorna
 *                         uint16_t  =>  resultado de la búsqueda - indice del valor máximo
 */
uint16_t argmax(data1d_t data);


/***************************************************************************************************************************/
/* Activation functions/layers */

void softmax_activation(float *data, uint16_t length);

void relu_activation(float *data, uint16_t length);

void tanh_activation(float *data, uint16_t length);

void sigmoid_activation(float *data, uint16_t length);

void softsign_activation(float *data, uint16_t length);



/***************************************************************************************************************************/
/* Normalization layers */

/* Normalization function for:
 *  - standard normalization  : (x_i-mean_i)/std_dev_i
 *  - min_max normalization   : (x_i-min_i) / (max_i-min_i)
 *  - robust normalization    : (x_i-q1_i)  / (q3_i-q1_i)
 */
void normalization1(normalization_t s, data1d_t input, data1d_t * output);

#define standard_norm_layer(norm, input, output) normalization1(norm, input, output)

#define min_max_norm_layer(norm, input, output) normalization1(norm, input, output)

#define robust_norm_layer(norm, input, output) normalization1(norm, input, output)


/* Normalization function for:
 *  - abs_max_normalization   : (x_i)/(abs_max_xi)
 */
void normalization2(normalization_t s, data1d_t input, data1d_t * output);

#define max_abs_norm_layer(norm, input, output) normalization2(norm, input, output)





#endif
