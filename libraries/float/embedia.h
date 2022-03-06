/* 
 * EmbedIA 
 * LIBRERÍA ARDUINO QUE DEFINE FUNCIONES PARA LA IMPLEMENTACIÓN DE REDES NEURNOALES CONVOLUCIONALES
 * EN MICROCONTROLADORES Y LAS ESTRUCTURAS DE DATOS NECESARIAS PARA SU USO
 */
#ifndef _EMBEDIA_H
#define _EMBEDIA_H

#include <stdint.h>
#include <math.h>



/* DEFINICIÓN DE ESTRUCTURAS */

/*
 * typedef struct data_t
 * Estructura que almacena una matriz de datos de tipo float  (float  * data) en forma de vector
 * Especifica la cantidad de canales (uint16_t channels), el ancho (uint16_t width) y el alto (uint16_t height) de la misma
 */
typedef struct{
    uint16_t channels;
    uint16_t width;
    uint16_t height;
    float  * data;
}data_t;

/*
 * typdef struct flatten_data_t
 * Estructura que almacena un vector de datos de tipo float  (float  * data).
 * Especifica el largo del mismo (uint16_t length).
 */
typedef struct{
    uint16_t length;
    float  * data;
}flatten_data_t;

/*
 * typedef struct filter_t
 * Estructura que almacena los pesos de un filtro.
 * Especifica la cantidad de canales (uint16_t channels), su tamaño (uint16_t kernel_size),
 * los pesos (float  * weights) y el bias (float  bias).
 */
typedef struct{
    uint16_t channels;
    uint16_t kernel_size;
    const float  * weights;
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
}conv_layer_t;

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
    const float  * weights;
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


/* PROTOTIPOS DE FUNCIONES DE LA LIBRERÍA */

/* 
 * conv2d()
 * Función que realiza la convolución entre un filtro y un conjunto de datos.
 * Parámetros:
 *             filter_t filter  =>  estructura filtro con pesos cargados
 *                data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 * 				     uint16_t delta	=>  posicionamiento de feature_map dentro de output.data
 */
void conv2d(filter_t filter, data_t input, data_t * output, uint16_t delta);

/* 
 * conv2d_layer()
 * Función que se encarga de aplicar la convolución de una capa de filtros (conv_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros:
 *          conv_layer_t layer  =>  capa convolucional con filtros cargados
 *         	      data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 */
void conv2d_layer(conv_layer_t layer, data_t input, data_t * output);

/* 
 * separable_conv2d_layer()
 * Función que se encarga de aplicar la convolución de una capa de filtros (separable_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros:
 *     separable_layer_t layer  =>  capa convolucional separable con filtros cargados
 *                data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 */
void separable_conv2d_layer(separable_layer_t layer, data_t input, data_t * output);

/* 
 * neuron_forward()
 * Función que realiza el forward de una neurona ante determinado conjunto de datos de entrada.
 * Parámetros:
 *             neuron_t neuron  =>  neurona con sus pesos y bias cargados
 *        flatten_data_t input  =>  datos de entrada en forma de vector (flatten_data_t)
 * Retorna:
 *                      float   =>  resultado de la operación             
 */
float neuron_forward(neuron_t neuron, flatten_data_t input);

/* 
 * dense_forward()
 * Función que se encarga de realizar el forward de una capa densa (dense_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros
 *          dense_layer_t dense_layer  =>  capa densa con neuronas cargadas
 *               flatten_data_t input  =>  datos de entrada de tipo flatten_data_t
 *            flatten_data_t * output  =>  puntero a la estructura flatten_data_t donde se guardará el resultado
 */
void dense_forward(dense_layer_t dense_layer, flatten_data_t input, flatten_data_t * output);

/* 
 * max_pooling2d()
 * Función que se encargará de aplicar un max pooling a una entrada
 * con un tamaño de ventana de recibido por parámetro (uint16_t strides)
 * a un determinado conjunto de datos de entrada.
 * Parámetros:
 *                data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 */
void max_pooling_2d(uint16_t pool_size, uint16_t strides, data_t input, data_t* output);

/* 
 * avg_pooling_2d()
 * Función que se encargará de aplicar un average pooling a una entrada
 * con un tamaño de ventana de recibido por parámetro (uint16_t strides)
 * a un determinado conjunto de datos de entrada.
 * Parámetros:
 *                data_t input  =>  datos de entrada de tipo data_t
 *             data_t * output  =>  puntero a la estructura data_t donde se guardará el resultado
 */
void avg_pooling_2d(uint16_t pool_size, uint16_t strides, data_t input, data_t* output);

/* 
 * softmax()
 * Función que implementa la activación softmax.
 * Convierte un vector de entrada en na distribución de probabilidades
 * (aplicado a la salida de regresión lineal para obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *         flatten_data_t data  =>  datos de entrada de tipo flatten_data_t
 */
void softmax(flatten_data_t data);

/* 
 * relu(data_t)
 * Función que implementa la activación relu, convierte un vector de entrada en 
 * una distribución de probabilidades (aplicado a la salida de regresión lineal para
 * obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *                 data_t data  =>  datos de tipo data_t a modificar
 */
void relu(data_t data);

/* 
 * relu_flatten(flatten_data_t)
 * Función que implementa la activación relu, convierte un vector de entrada en 
 * una distribución de probabilidades (aplicado a la salida de regresión lineal para
 * obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *         flatten_data_t data  =>  datos de tipo flatten_data_t a modificar
 */
void relu_flatten(flatten_data_t data);

/* 
 * tanh(data_t)
 * Función que implementa la activación tanh, convierte un vector de entrada en 
 * una distribución de probabilidades (aplicado a la salida de regresión lineal para
 * obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *                 data_t data  =>  datos de tipo data_t a modificar
 */
void tanh2d(data_t data);

/* 
 * tanh_flatten(flatten_data_t)
 * Función que implementa la activación tanh, convierte un vector de entrada en 
 * una distribución de probabilidades (aplicado a la salida de regresión lineal para
 * obtener una distribución de probabilidades para de cada clase).
 * Parámetro:
 *         flatten_data_t data  =>  datos de tipo flatten_data_t a modificar
 */
void tanh_flatten(flatten_data_t data);

/* 
 * flatten_layer()
 * Realiza un cambio de tipo de variable. 
 * Convierte el formato de los datos en formato de matriz data_t en vector flatten_data_t.
 * (prepara los datos para ingresar en una capa de tipo dense_layer_t)
 * Parámetros:
 *                data_t input  =>  datos de entrada de tipo data_t
 *     flatten_data_t * output  =>  puntero a la estructura flatten_data_t donde se guardará el resultado
 */
void flatten_layer(data_t input, flatten_data_t * output);

/* 
 * argmax()
 * Busca el indice del valor mayor dentro de un vector de datos (flatten_data_t)
 * Parámetros:
 *         flatten_data_t data  =>  datos de tipo flatten_data_t a buscar máximo
 * Retorna
 *                         uint16_t  =>  resultado de la búsqueda - indice del valor máximo
 */
uint16_t argmax(flatten_data_t data);


#endif
