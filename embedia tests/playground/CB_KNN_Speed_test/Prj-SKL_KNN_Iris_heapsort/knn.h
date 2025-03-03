#ifndef _KNN_H
#define _KNN_H

#include "common.h"
#include <math.h>

/* STRUCTURE DEFINITION */

/*
 * Structure that models a KNN layer.
 */
typedef struct
{
    uint16_t n_neighbors;
    uint32_t n_samples;
    uint16_t n_features;
    uint16_t n_classes;
    float *neighbors_features;
    uint16_t *neighbors_id;
}k_neighbors_classifier_layer_t;

typedef k_neighbors_classifier_layer_t k_neighbors_regressor_layer_t;

/* LIBRARY FUNCTIONS PROTOTYPES */

/*
 * knn_layer()
 *   Function in charge of applying the convolution of a filter layer (conv_layer_t) without padding and strides
 *   on a given input data set.
 * Parameters:
 *  - layer => knn layer with loaded neighbors.
 *  - input => input data of type data1d_t
 *  - *output => pointer to the data1d_t structure where the result will be saved.
 */
void k_neighbors_classifier_layer(k_neighbors_classifier_layer_t layer, data1d_t input, data1d_t * output);
void k_neighbors_regressor_layer(k_neighbors_regressor_layer_t layer, data1d_t input, data1d_t * output);


// void sort_distances(uint32_t n_rows, float* distances, int* sorted_indexes);
// float get_max_label(int n_neighbors, int sorted_indexes, float neighbors_labels, int n_classes);

#endif