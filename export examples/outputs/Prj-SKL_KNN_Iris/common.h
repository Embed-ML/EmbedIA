#ifndef _COMMON_H
#define _COMMON_H

#include <stdlib.h>
#include <stdint.h>
#include "fixed.h"

/*
 * Structure that stores an array of float data (float * data) in vector form.
 * Specifies the number of channels, the width and the height of the array.
 */
typedef fixed compute_t;


typedef struct{
    uint16_t channels;
    uint16_t width;
    uint16_t height;
    compute_t * data;
}data3d_t;

typedef struct{
    uint16_t width;
    uint16_t height;
    compute_t * data;
}data2d_t;

typedef struct{
    uint32_t length;
    compute_t * data;
}data1d_t;

typedef struct{
    uint16_t h;
    uint16_t w;
} size2d_t;


void prepare_buffers();

void * swap_alloc(size_t s);

/*
 * argmax()
 * Finds the index of the largest value within a vector of data (data1d_t)
 * Parameters:
 *  - data => data of type data1d_t to search for max.
 * Returns:
 *  - search result - index of the maximum value
 */
uint32_t argmax(data1d_t data);


#endif