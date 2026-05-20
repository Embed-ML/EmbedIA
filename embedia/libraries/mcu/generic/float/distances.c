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

float euclidean_distance(float *x, float *y, int length) {
    float distance = 0, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = y[i] - x[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

/**
 * @brief Squared Euclidean distance (optimized for KNN).
 * @note Avoids sqrt → much more efficient.
 */
float euclidean_sq_distance(float *x, float *y, int length) {
    float distance = 0, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = y[i] - x[i];
        distance += diff * diff;
    }

    return distance;
}

/**
 * @brief Fast approximate Euclidean distance using magnitude approximation.
 * @note Very fast, low precision. Useful for embedded real-time heuristics.
 */
float euclidean_fast_distance(float *x, float *y, int length) {
    float acc = 0.0f;
    int i;

    for (i = 0; i < length; i++) {
        float diff = fabsf(y[i] - x[i]);
        acc = acc + diff; // Simple approximation - could be improved with better magnitude function
    }

    return acc;
}

float manhattan_distance(float *x, float *y, int length) {
    float distance = 0.0f;
    int i;

    for (i = 0; i < length; i++) {
        distance += fabsf(y[i] - x[i]);
    }
    return distance;
}

float chebyshev_distance(float *x, float *y, int length) {
    float max_diff = 0.0f, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = fabsf(y[i] - x[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

float minkowski_distance(float *x, float *y, int length, float p) {
    float distance = 0.0f, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = fabsf(y[i] - x[i]);
        distance += powf(diff, p);
    }
    return powf(distance, 1.0f / p);
}

float cosine_distance(float *x, float *y, int length) {
    float dot_product = 0.0f;
    float norm_x = 0.0f;
    float norm_y = 0.0f;
    int i;

    for (i = 0; i < length; i++) {
        dot_product += x[i] * y[i];
        norm_x += x[i] * x[i];
        norm_y += y[i] * y[i];
    }

    if (norm_x == 0.0f || norm_y == 0.0f) {
        return 1.0f;
    }

    float similarity = dot_product / (sqrtf(norm_x) * sqrtf(norm_y));

    // Clamp similarity to [-1, 1] to handle numerical errors
    if (similarity > 1.0f) similarity = 1.0f;
    if (similarity < -1.0f) similarity = -1.0f;

    return 1.0f - similarity;
}

float braycurtis_distance(float *x, float *y, int length) {
    float sum_diff = 0.0, sum_total = 0.0;
    for (int i = 0; i < length; i++) {
        sum_diff += fabsf(x[i] - y[i]);
        sum_total += fabsf(x[i] + y[i]);
    }
    return (sum_total == 0.0) ? 0.0 : (sum_diff / sum_total);
}

float canberra_distance(float *x, float *y, int length) {
    float distance = 0.0;
    for (int i = 0; i < length; i++) {
        float denom = fabsf(x[i]) + fabsf(y[i]);
        if (denom != 0.0) {
            distance += fabsf(x[i] - y[i]) / denom;
        }
    }
    return distance;
}

