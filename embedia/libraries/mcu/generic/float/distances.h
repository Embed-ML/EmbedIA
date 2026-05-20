#ifndef _DISTANCES_H
#define _DISTANCES_H
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

#ifdef __cplusplus
extern "C" {
#endif

/*********************************** Distance Functions for vectors ******************************************/

/**
 * @brief Calculates Euclidean (L2) distance between two vectors.
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length (must be equal for both).
 * @return Euclidean distance: sqrt(sum((x[i] - y[i])^2)).
 */
float euclidean_distance(float *x, float *y, int length);

/**
 * @brief Calculates squared Euclidean distance (optimized).
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length.
 * @return sum((x[i] - y[i])^2).
 * @note Recommended for KNN: avoids sqrt, faster and preserves ordering.
 */
float euclidean_sq_distance(float *x, float *y, int length);

/**
 * @brief Fast approximate Euclidean distance.
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length.
 * @return Approximate L2 distance using magnitude approximation.
 * @note Very fast, low precision. Useful for embedded real-time heuristics.
 */
float euclidean_fast_distance(float *x, float *y, int length);

/**
 * @brief Calculates Manhattan (L1) distance between two vectors.
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length (must be equal for both).
 * @return Manhattan distance: sum(|x[i] - y[i]|).
 */
float manhattan_distance(float *x, float *y, int length);

/**
 * @brief Calculates Chebyshev (L∞) distance between two vectors.
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length (must be equal for both).
 * @return Chebyshev distance: max(|x[i] - y[i]|).
 */
float chebyshev_distance(float *x, float *y, int length);

/**
 * @brief Calculates Minkowski distance between two vectors.
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length (must be equal for both).
 * @param p Order parameter (p >= 1). Special cases: p=1 (Manhattan), p=2 (Euclidean).
 * @return Minkowski distance: (sum(|x[i] - y[i]|^p))^(1/p).
 */
float minkowski_distance(float *x, float *y, int length, float p);

/**
 * @brief Calculates cosine distance between two vectors.
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length (must be equal for both).
 * @return Cosine distance: 1 - (x·y)/(||x|| ||y||). Range [0,2]: 0=identical direction, 1=orthogonal, 2=opposite.
 */
float cosine_distance(float *x, float *y, int length);

/**
 * @brief Calculates Bray-Curtis dissimilarity between two vectors.
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length (must be equal for both).
 * @return Bray-Curtis dissimilarity: sum(|x[i]-y[i]|) / sum(|x[i]+y[i]|). Range [0,1].
 */
float braycurtis_distance(float *x, float *y, int length);

/**
 * @brief Calculates Canberra distance between two vectors.
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length (must be equal for both).
 * @return Canberra distance: sum(|x[i]-y[i]| / (|x[i]|+|y[i]|)). Sensitive to small values near zero.
 */
float canberra_distance(float *x, float *y, int length);


#ifdef __cplusplus
}
#endif

#endif
