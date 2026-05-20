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
 * @defgroup distances Distance functions (fixed-point)
 *
 * @brief Collection of distance metrics for fixed-point vectors.
 *
 * @note Precision model:
 * - For fixed8 (Q4.4):
 *     All functions return results in dfixed (Q8.8) to preserve precision.
 *     This avoids early quantization and improves ranking stability (e.g. in KNN).
 *
 * - For higher precision types (fixed16, fixed32):
 *     The same API is used, but dfixed has sufficient precision to represent
 *     the result without effective loss, so values behave as native fixed.
 *
 * @note Conversion:
 * - Use DFX2FX_RND_SAT() only at the final stage (output, logging, etc).
 * - Do NOT assume returned values are in fixed scale when using fixed8.
 */

/**
 * @brief Calculates Euclidean (L2) distance between two vectors.
 * @param x First vector.
 * @param y Second vector.
 * @param length Vector length.
 * @return Euclidean distance in dfixed (Q8.8 for fixed8).
 * @note Uses dfixed_sqrt. More expensive due to sqrt.
 */
dfixed euclidean_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates squared Euclidean distance (optimized).
 * @return sum((x[i] - y[i])^2) in dfixed.
 * @note Recommended for KNN: avoids sqrt, faster and preserves ordering.
 */
dfixed euclidean_sq_distance(fixed *x, fixed *y, int length);

/**
 * @brief Fast approximate Euclidean distance.
 * @return Approximate L2 distance in dfixed.
 * @note Very fast, low precision. Uses magnitude approximation.
 */
dfixed euclidean_fast_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Manhattan (L1) distance.
 * @return sum(|x[i] - y[i]|) in dfixed.
 */
dfixed manhattan_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Chebyshev (L∞) distance.
 * @return max(|x[i] - y[i]|) in dfixed scale.
 * @note Internally computed in fixed but promoted to dfixed.
 */
dfixed chebyshev_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Minkowski distance.
 * @param p Order parameter (p >= 1).
 * @return (sum(|x[i] - y[i]|^p))^(1/p) in dfixed.
 * @warning Computationally expensive (pow/log/exp).
 */
dfixed minkowski_distance(fixed *x, fixed *y, int length, fixed p);

/**
 * @brief Calculates cosine distance.
 * @return 1 - (x·y)/(||x|| ||y||), range [0,2], in dfixed.
 * @note Fully computed in dfixed for improved precision in fixed8.
 */
dfixed cosine_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Bray-Curtis dissimilarity.
 * @return sum(|x[i]-y[i]|) / sum(|x[i]+y[i]|), range [0,1], in dfixed.
 * @note Uses dfixed accumulation and division.
 */
dfixed braycurtis_distance(fixed *x, fixed *y, int length);

/**
 * @brief Calculates Canberra distance.
 * @return sum(|x[i]-y[i]| / (|x[i]|+|y[i]|)) in dfixed.
 * @note Skips terms with zero denominator.
 */
dfixed canberra_distance(fixed *x, fixed *y, int length);

#ifdef __cplusplus
}
#endif

#endif