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
 * GitHub: https://github.com/Embed-ML/EmbedIA
 */

#include "common.h"
#include "fixed.h"
#include <math.h>

/* STRUCTURE DEFINITION */

/**
 * @brief KNN classifier/regressor layer structure.
 *
 * Stores training samples and configuration for K-Nearest Neighbors algorithm.
 * Uses fixed8 arithmetic (limited precision) and function pointer for distance metric selection.
 * Note: Distance function returns dfixed for better precision during comparisons.
 */
typedef struct
{
    uint16_t n_neighbors;           /**< Number of neighbors (k) */
    uint32_t n_samples;             /**< Total training samples */
    uint16_t n_features;            /**< Number of features per sample */
    uint16_t n_classes;             /**< Number of classes (classifier only) */
    fixed *neighbors_features;      /**< Training data [n_samples × n_features] in fixed8 */
    uint16_t *neighbors_id;         /**< Class labels (classifier) or target values (regressor) */
    dfixed (*distance_fn)(fixed*, fixed*, int);  /**< Distance function pointer (returns dfixed) */
} k_neighbors_classifier_layer_t;

typedef k_neighbors_classifier_layer_t k_neighbors_regressor_layer_t;

/* LIBRARY FUNCTIONS PROTOTYPES */

/**
 * @brief Performs KNN classification using heap-based k-nearest neighbor search.
 * @param layer KNN layer configuration with training data.
 * @param input Input feature vector.
 * @param output Class probabilities (length = n_classes). Memory allocated internally.
 */
void k_neighbors_classifier_layer(k_neighbors_classifier_layer_t layer, data1d_t input, data1d_t * output);

/**
 * @brief Performs KNN regression using heap-based k-nearest neighbor search.
 * @param layer KNN layer configuration with training data.
 * @param input Input feature vector.
 * @param output Predicted value (length = 1). Memory allocated internally.
 */
void k_neighbors_regressor_layer(k_neighbors_regressor_layer_t layer, data1d_t input, data1d_t * output);

#endif