#ifndef _KNN_H
#define _KNN_H
/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 */


#include <math.h>
#include "common.h"

/* STRUCTURE DEFINITION */

typedef computex_t (*distance_fn_t)(compute_t*, compute_t*, int);

/**
 * @brief KNN classifier/regressor layer structure for quantized data.
 * 
 * Stores training samples in quant8 format with quantization parameters.
 * Distance calculations are performed after dequantization to fixed-point.
 * Note: Distance function returns dfixed for better precision during comparisons.
 */
typedef struct
{
    uint16_t n_neighbors;           /**< Number of neighbors (k) */
    uint32_t n_samples;             /**< Total training samples */
    uint16_t n_features;            /**< Number of features per sample */
    uint16_t n_classes;             /**< Number of classes (classifier only) */
    storage_t *neighbors_features;  /**< Training data [n_samples × n_features] in quant8 */
    uint16_t *neighbors_id;         /**< Class labels (classifier) or target values (regressor) */
    distance_fn_t distance_fn;      /**< Distance function pointer (returns computex_t) */
#if DATA_TYPE_IMPL == DT_QUANT8
    qparam_t qparam;                /**< Quantization parameters (scale, zero_point) */
#endif
}k_neighbors_classifier_layer_t;

typedef k_neighbors_classifier_layer_t k_neighbors_regressor_layer_t;

/* LIBRARY FUNCTIONS PROTOTYPES */

/**
 * @brief Performs KNN classification using heap-based k-nearest neighbor search.
 * @param layer KNN layer configuration with quantized training data.
 * @param input Input feature vector (dequantized).
 * @param output Class probabilities (length = n_classes). Memory allocated internally.
 * @note Training samples are dequantized on-the-fly during distance calculation.
 */
void k_neighbors_classifier_layer(k_neighbors_classifier_layer_t layer, data1d_t input, data1d_t * output);

/**
 * @brief Performs KNN regression using heap-based k-nearest neighbor search.
 * @param layer KNN layer configuration with quantized training data.
 * @param input Input feature vector (dequantized).
 * @param output Predicted value (length = 1). Memory allocated internally.
 * @note Training samples are dequantized on-the-fly during distance calculation.
 */
void k_neighbors_regressor_layer(k_neighbors_regressor_layer_t layer, data1d_t input, data1d_t * output);

#endif