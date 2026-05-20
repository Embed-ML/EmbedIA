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
 * @param x First vector in fixed8.
 * @param y Second vector in fixed8.
 * @param length Vector length (must be equal for both).
 * @return Euclidean distance: sqrt(sum((x[i] - y[i])^2)).
 * @note Uses dfixed accumulation internally. More expensive due to sqrt.
 */
dfixed euclidean_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates squared Euclidean distance (optimized).
 * @param x First vector in fixed8.
 * @param y Second vector in fixed8.
 * @param length Vector length.
 * @return sum((x[i] - y[i])^2) in dfixed.
 * @note Recommended for KNN: avoids sqrt, faster and preserves ordering.
 */
dfixed euclidean_sq_distance(fixed *x, fixed *y, int length);

/**
 * @brief Fast approximate Euclidean distance.
 * @param x First vector in fixed8.
 * @param y Second vector in fixed8.
 * @param length Vector length.
 * @return Approximate L2 distance using magnitude approximation.
 * @note Very fast, low precision. Useful for embedded real-time heuristics.
 */
dfixed euclidean_fast_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Manhattan (L1) distance between two vectors.
 * @return sum(|x[i] - y[i]|) in dfixed.
 */
dfixed manhattan_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Chebyshev (L∞) distance between two vectors.
 * @return max(|x[i] - y[i]|).
 */
dfixed chebyshev_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Minkowski distance between two vectors.
 * @param p Order parameter (p >= 1).
 * @return (sum(|x[i] - y[i]|^p))^(1/p).
 * @warning Computationally expensive due to pow/log/exp usage.
 */
dfixed minkowski_distance(fixed *x, fixed *y, int length, fixed p);

/**
 * @brief Calculates cosine distance between two vectors.
 * @return 1 - (x·y)/(||x|| ||y||), range [0,2].
 * @note Limited precision in fixed8; uses dfixed accumulation internally.
 */
dfixed cosine_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Bray-Curtis dissimilarity.
 * @return sum(|x[i]-y[i]|) / sum(|x[i]+y[i]|), range [0,1].
 * @note Uses dfixed accumulation and safe division.
 */
dfixed braycurtis_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Canberra distance.
 * @return sum(|x[i]-y[i]| / (|x[i]|+|y[i]|)).
 * @note Skips terms with zero denominator.
 */
dfixed canberra_distance(fixed *x, fixed *y, int length);

#ifdef __cplusplus
}
#endif

#endif