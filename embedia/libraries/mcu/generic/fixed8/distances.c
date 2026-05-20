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
#include "distances.h"
#include <math.h>



static inline dfixed dfixed_sqrt(dfixed x) {
    if (x <= 0) return 0;

    dfixed guess = x;
    dfixed prev;

    // Ajuste inicial opcional (mejora convergencia)
    if (x > (DFIX_ONE << 2)) {
        guess = x >> 1;
    }

    // Iteraciones Newton
    for (int i = 0; i < 6; i++) {
        prev = guess;

        // guess = (guess + x / guess) / 2
        dfixed div = DFIXED_DDIV(x, guess);
        guess = (guess + div) >> 1;

        if (guess == prev) break;
    }

    return guess;
}

/**
 * @brief Euclidean distance (mejorada).
 * @note Mantiene compatibilidad pero evita pérdida de precisión.
 */
dfixed euclidean_distance(fixed *x, fixed *y, int length) {
    dfixed acc = 0;

    for (int i = 0; i < length; i++) {
        dfixed d = (dfixed)y[i] - x[i];
        acc += d * d;
    }

    return dfixed_sqrt(acc);
}

/**
 * @brief Euclidean distance squared (rápida, recomendada para KNN).
 * @note Evita sqrt → mucho más eficiente.
 */
dfixed euclidean_sq_distance(fixed *x, fixed *y, int length) {
    dfixed acc = 0;

    for (int i = 0; i < length; i++) {
        dfixed d = (dfixed)y[i] - x[i];
        acc += d * d;
    }

    return acc;
}

/**
 * @brief Manhattan distance (ligera mejora de consistencia).
 */
dfixed manhattan_distance(fixed *x, fixed *y, int length) {
    dfixed acc = 0;

    for (int i = 0; i < length; i++) {
        dfixed d = (dfixed)y[i] - x[i];
        acc += DFIXED_ABS(d);
    }

    return acc;
}

/**
 * @brief Chebyshev distance (sin cambios relevantes).
 */
dfixed chebyshev_distance(fixed *x, fixed *y, int length) {
    fixed max_diff = FIX_ZERO;

    for (int i = 0; i < length; i++) {
        fixed diff = FIXED_ABS(y[i] - x[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}

/**
 * @brief Minkowski distance (sin cambios estructurales).
 */
dfixed minkowski_distance(fixed *x, fixed *y, int length, fixed p) {
    dfixed acc = FIX_ZERO;

    for (int i = 0; i < length; i++) {
        fixed diff = FIXED_ABS(y[i] - x[i]);
        acc += fixed_pow(diff, p);
    }

    return fixed_pow(acc, FIXED_DIV(FIX_ONE, p));
}

/**
 * @brief Cosine distance (mejoras menores de consistencia).
 */
dfixed cosine_distance(fixed *x, fixed *y, int length) {
    dfixed dot = 0;
    dfixed norm_x = 0;
    dfixed norm_y = 0;

    for (int i = 0; i < length; i++) {
        dfixed xi = x[i];
        dfixed yi = y[i];

        dot    += xi * yi;   // Q8.8
        norm_x += xi * xi;   // Q8.8
        norm_y += yi * yi;   // Q8.8
    }

    if (norm_x == 0 || norm_y == 0) {
        return FIXED_TO_DFIXED(FIX_ONE);  // distancia = 1
    }

    dfixed sqrt_x = dfixed_sqrt(norm_x); // Q8.8 → Q8.8
    dfixed sqrt_y = dfixed_sqrt(norm_y); // Q8.8 → Q8.8

    if (sqrt_x == 0 || sqrt_y == 0) {
        return FIXED_TO_DFIXED(FIX_ONE);
    }

    // denom = sqrt_x * sqrt_y  → Q8.8 * Q8.8 = Q16.16
    int32_t denom = (int32_t)sqrt_x * (int32_t)sqrt_y;

    // dot está en Q8.8 → lo llevamos a Q16.16 para dividir correctamente
    int32_t dot_q16 = ((int32_t)dot) << FIX_FRC_SZ;

    // similitud en Q8.8
    dfixed sim = (dfixed)((dot_q16 << FIX_FRC_SZ) / denom);

    // clamp en Q8.8
    dfixed one = FIXED_TO_DFIXED(FIX_ONE);

    if (sim > one) sim = one;
    if (sim < -one) sim = -one;

    // distancia = 1 - cos
    return one - sim;
}

dfixed braycurtis_distance(fixed *x, fixed *y, int length) {
    dfixed sum_diff = 0;
    dfixed sum_total = 0;

    for (int i = 0; i < length; i++) {
        dfixed xi = x[i];
        dfixed yi = y[i];

        sum_diff  += DFIXED_ABS(xi - yi);
        sum_total += DFIXED_ABS(xi + yi);
    }

    if (sum_total == 0) return 0;

    return DFIXED_DDIV(sum_diff, sum_total);
}

/**
 * @brief Canberra distance (mejoras menores de robustez).
 */
dfixed canberra_distance(fixed *x, fixed *y, int length) {
    dfixed acc = 0;

    for (int i = 0; i < length; i++) {
        dfixed xi = x[i];
        dfixed yi = y[i];

        dfixed denom = DFIXED_ABS(xi) + DFIXED_ABS(yi);

        if (denom != 0) {
            dfixed diff = DFIXED_ABS(xi - yi);
            acc += (diff << FIX_FRC_SZ) / denom;
        }
    }

    return acc;
}

/**
 * @brief Fast Euclidean approximation using magnitude (muy rápida).
 * @note Basada en fixed_magnitude: evita sqrt y multiplicaciones costosas.
 */
dfixed euclidean_fast_distance(fixed *x, fixed *y, int length) {
    dfixed acc = 0;

    for (int i = 0; i < length; i++) {
        dfixed d = (dfixed)y[i] - x[i];
        dfixed abs_d = DFIXED_ABS(d);

        // adaptar magnitude a dfixed
        dfixed max = acc > abs_d ? acc : abs_d;
        dfixed min = acc > abs_d ? abs_d : acc;

        // approx: max + 0.375 * min
        acc = max + ((min >> 2) + (min >> 3));
    }

    return acc;
}