#ifndef _NORMALIZATION_H
#define _NORMALIZATION_H
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

/**
 * @file normalization.h
 * @brief EmbedIA - Data Normalization Functions
 *
 * This library provides normalization functions commonly used in machine learning
 * preprocessing, based on scikit-learn normalization techniques.
 * These functions are independent of neural network layers and can be used
 * for general data preprocessing.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "common.h"

//{includes}

/**
 * @defgroup normalization_structures Normalization Data Structures
 * @brief Data structures for normalization parameters.
 * @{
 */

/**
 * @brief Normalization layer parameters structure
 *
 * This structure contains the parameters needed for normalization operations
 * based on scikit-learn preprocessing techniques.
 */
typedef struct {
    const real_t *sub_val;      ///< Values to subtract (mean, min, median, etc.)
    const real_t *inv_div_val;   ///< Inverse values for division (1/std, 1/(max-min), etc.)
} normalization_layer_t;

/** @} */ // end of normalization_structures

/**
 * @defgroup normalization_functions Normalization Functions
 * @brief Functions to apply various normalization techniques for data preprocessing.
 * @{
 */

/**
 * @brief Applies generic normalization: (x_i - sub_val[i]) * inv_div_val[i]
 *
 * Used for standard, min-max, and robust normalization based on scikit-learn.
 *
 * @param s       Normalization parameters containing mean and scale values
 * @param input   Input data (1D)
 * @param output  Pointer to output data
 */
void normalization1(normalization_layer_t s, data1d_t input, data1d_t * output);

/**
 * @brief Applies standard normalization (z-score normalization)
 *
 * Equivalent to scikit-learn's StandardScaler: (x - mean) / std
 *
 * @param norm    Normalization parameters
 * @param input   Input data (1D)
 * @param output  Pointer to output data
 */
#define standard_norm_layer(norm, input, output) normalization1(norm, input, output)

/**
 * @brief Applies min-max normalization
 *
 * Equivalent to scikit-learn's MinMaxScaler: (x - min) / (max - min)
 *
 * @param norm    Normalization parameters
 * @param input   Input data (1D)
 * @param output  Pointer to output data
 */
#define min_max_norm_layer(norm, input, output) normalization1(norm, input, output)

/**
 * @brief Applies robust normalization
 *
 * Equivalent to scikit-learn's RobustScaler using median and IQR
 *
 * @param norm    Normalization parameters
 * @param input   Input data (1D)
 * @param output  Pointer to output data
 */
#define robust_norm_layer(norm, input, output) normalization1(norm, input, output)

/**
 * @brief Applies max absolute normalization
 *
 * Normalization function for abs_max_normalization: (x_i)/(abs_max_xi)
 * Equivalent to scikit-learn's MaxAbsScaler
 *
 * @param s       Normalization parameters
 * @param input   Input data (1D)
 * @param output  Pointer to output data
 */
void normalization2(normalization_layer_t s, data1d_t input, data1d_t * output);

/**
 * @brief Applies max absolute normalization
 *
 * Equivalent to scikit-learn's MaxAbsScaler
 *
 * @param norm    Normalization parameters
 * @param input   Input data (1D)
 * @param output  Pointer to output data
 */
#define max_abs_norm_layer(norm, input, output) normalization2(norm, input, output)

/** @} */ // end of normalization_functions

#ifdef __cplusplus
}
#endif

#endif /* NORMALIZATION_H */
