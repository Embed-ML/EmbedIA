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
#include "distances.h"
#include <math.h>

/**
 * @brief Calculates Euclidean distance with overflow protection.
 * @return sqrt(sum((x[i] - y[i])^2)), clamped to FIX_MAX if intermediate sum overflows.
 */
dfixed euclidean_distance(fixed *x, fixed *y, int length) {
    dfixed distance = 0, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = y[i] - x[i];
        distance += DFIXED_MUL(diff, diff);
    }
    if (distance > (FIX_MAX*FIX_MAX)){
        return FIX_MAX;
    }

    return fixed_sqrt(DFX2FX(distance));
}

/**
 * @brief Calculates Manhattan distance.
 * @return sum(|x[i] - y[i]|) accumulated in dfixed.
 */
dfixed manhattan_distance(fixed *x, fixed *y, int length) {
    dfixed distance = 0;
    for (int i = 0; i < length; i++) {
        distance += FIXED_ABS(y[i] - x[i]);
    }
    return distance;
}

/**
 * @brief Calculates Chebyshev distance.
 * @return max(|x[i] - y[i]|).
 */
dfixed chebyshev_distance(fixed *x, fixed *y, int length) {
    fixed max_diff = FIX_ZERO, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = FIXED_ABS(y[i] - x[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

/**
 * @brief Calculates Minkowski distance.
 * @return (sum(|x[i] - y[i]|^p))^(1/p).
 */
dfixed minkowski_distance(fixed *x, fixed *y, int length, fixed p) {
    dfixed distance = FIX_ZERO, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = FIXED_ABS(y[i] - x[i]);
        distance += fixed_pow(diff, p);
    }
    return fixed_pow(distance, FIXED_DIV(FIX_ONE, p));
}

/**
 * @brief Calculates cosine distance.
 * @return 1 - (x·y)/(||x|| ||y||), with clamping to handle numerical errors.
 */
dfixed cosine_distance(fixed *x, fixed *y, int length) {
    fixed dot_product = FIX_ZERO;
    fixed norm_x = FIX_ZERO;
    fixed norm_y = FIX_ZERO;
    int i;

    for (i = 0; i < length; i++) {
        dot_product += FIXED_MUL(x[i], y[i]);
        norm_x += FIXED_MUL(x[i], x[i]);
        norm_y += FIXED_MUL(y[i], y[i]);
    }

    if (norm_x == FIX_ZERO || norm_y == FIX_ZERO) {
        return FIX_ONE; // Arbitrary choice for zero vectors
    }

    // Convert from similarity (1 = identical) to distance (0 = identical)
    fixed similarity = FIXED_DIV(dot_product,FIXED_MUL(fixed_sqrt(norm_x), fixed_sqrt(norm_y) ) );

    // Clamp similarity to [-1, 1] to handle numerical errors
    if (similarity > FIX_ONE) similarity = FIX_ONE;
    if (similarity < -FIX_ONE) similarity = -FIX_ONE;

    // Convert to distance: d = 1 - similarity
    return FIX_ONE - similarity;
}

/**
 * @brief Calculates Bray-Curtis dissimilarity.
 * @return sum(|x[i]-y[i]|) / sum(|x[i]+y[i]|), or 0 if denominator is zero.
 */
dfixed braycurtis_distance(fixed *x, fixed *y, int length) {
    fixed sum_diff = 0;
    fixed sum_total = 0;
    int i;

    for (i = 0; i < length; i++) {
        sum_diff += FIXED_ABS(x[i] - y[i]); // Suma de diferencias absolutas
        sum_total += FIXED_ABS(x[i] + y[i]); // Suma de valores absolutos
    }

    // Evitar división por cero
    if (sum_total == 0) {
        return 0;
    }

    // Calcular la disimilitud: sum_diff / sum_total
    return FIXED_DIV(sum_diff, sum_total);
}

/**
 * @brief Calculates Canberra distance.
 * @return sum(|x[i]-y[i]| / (|x[i]|+|y[i]|)), skipping terms where denominator is zero.
 */
dfixed canberra_distance(fixed *x, fixed *y, int length) {
    dfixed denom, distance = 0;
    int i;

    for (i = 0; i < length; i++) {
        denom = FIXED_ABS(x[i]) + FIXED_ABS(y[i]); // Denominador
        if (denom != 0) {
            fixed diff = FIXED_ABS(x[i] - y[i]); // Diferencia absoluta
            distance += DFIXED_DIV(diff, denom);  // Sumar la fracción
        }
    }

    return RX2R_SAT(distance);
}

/**
 * @brief Squared Euclidean distance (optimized for KNN).
 * @note Avoids sqrt → much more efficient.
 */
dfixed euclidean_sq_distance(fixed *x, fixed *y, int length) {
    dfixed acc = 0;
    int i;

    for (i = 0; i < length; i++) {
        dfixed d = (dfixed)y[i] - x[i];
        acc += d * d;
    }

    return acc;
}

/**
 * @brief Fast approximate Euclidean distance using magnitude approximation.
 * @note Very fast, low precision. Useful for embedded real-time heuristics.
 */
dfixed euclidean_fast_distance(fixed *x, fixed *y, int length) {
    fixed acc = FIX_ZERO;
    int i;

    for (i = 0; i < length; i++) {
        fixed diff = FIXED_ABS(y[i] - x[i]);
        acc = fixed_magnitude(acc, diff);
    }

    return acc;
}
