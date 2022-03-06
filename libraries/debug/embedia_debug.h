#ifndef EMBEDIA_DEBUG_H_INCLUDED
#define EMBEDIA_DEBUG_H_INCLUDED

#include "embedia.h"

// exporter must define EMBEDIA_DEBUG macro
// 0 => NO DEBUG, 1 => DATA HEADER, 2 => DATA CONTENT



void print_data_t(const char *head_text, data_t data);


/*
 * print_flatten_data_t()
 * Imprime los valores presentes en un vector de datos y su largo
 * Parámetros:
 *            flatten_data_t data => vector de datos a imprimir
 */


void print_flatten_data_t(const char *head_text, flatten_data_t data);
/*
 * print_filter_t()
 * Imprime los valores de los pesos del filtro y sus dimensiones
 * Parámetros:
 *                filter_t filtro => filtro a imprimir
 */

void print_filter_t(const char *head_text, filter_t filter);


#endif // EMBEDIA_DEBUG_H_INCLUDED
