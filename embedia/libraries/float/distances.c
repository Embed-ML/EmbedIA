#include "distances.h"
#include <math.h>

/*
 * euclidean_distance()
 *  Calculates the Euclidean distance between two data vectors.
 * Parameters:
 *  x      => first data vector of type float
 *  y      => second data vector of type float
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the Euclidean distance between vectors x and y
 */
float euclidean_distance(float *x, float *y, int length) {
    float distance = 0, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = y[i] - x[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

/*
 * manhattan_distance()
 *  Calculates the Manhattan (L1) distance between two data vectors.
 * Parameters:
 *  x      => first data vector of type float
 *  y      => second data vector of type float
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the Manhattan distance between vectors x and y
 */
float manhattan_distance(float *x, float *y, int length) {
    float distance = 0.0f;
    int i;

    for (i = 0; i < length; i++) {
        distance += fabsf(y[i] - x[i]);
    }
    return distance;
}

/*
 * chebyshev_distance()
 *  Calculates the Chebyshev (L∞) distance between two data vectors.
 *  This is the maximum absolute difference between any pair of elements.
 * Parameters:
 *  x      => first data vector of type float
 *  y      => second data vector of type float
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the Chebyshev distance between vectors x and y
 */
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

/*
 * minkowski_distance()
 *  Calculates the Minkowski distance between two data vectors.
 *  This is a generalization of Euclidean and Manhattan distances.
 * Parameters:
 *  x      => first data vector of type float
 *  y      => second data vector of type float
 *  length => length of the vectors (both must have the same length)
 *  p      => the order of the Minkowski distance (p >= 1)
 *             p=1: Manhattan distance
 *             p=2: Euclidean distance
 *             p→∞: Chebyshev distance
 * Returns:
 *  result - value of the Minkowski distance between vectors x and y
 */
float minkowski_distance(float *x, float *y, int length, float p) {
    float distance = 0.0f, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = fabsf(y[i] - x[i]);
        distance += powf(diff, p);
    }
    return powf(distance, 1.0f / p);
}

/*
 * cosine_distance()
 *  Calculates the cosine distance between two data vectors.
 *  This measures the angle between two vectors regardless of magnitude.
 * Parameters:
 *  x      => first data vector of type float
 *  y      => second data vector of type float
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the cosine distance between vectors x and y (range 0-2)
 *           0: identical direction, 1: orthogonal, 2: opposite direction
 */
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
        return 1.0f; // Arbitrary choice for zero vectors
    }

    // Convert from similarity (1 = identical) to distance (0 = identical)
    float similarity = dot_product / (sqrtf(norm_x) * sqrtf(norm_y));

    // Clamp similarity to [-1, 1] to handle numerical errors
    if (similarity > 1.0f) similarity = 1.0f;
    if (similarity < -1.0f) similarity = -1.0f;

    // Convert to distance: d = 1 - similarity
    return 1.0f - similarity;
}


/*
 * braycurtis_distance()
 *  Calculates the Bray-Curtis dissimilarity between two data vectors.
 *  This distance is commonly used in ecology to measure the dissimilarity
 *  between two samples based on their abundance or composition.
 * Parameters:
 *  x      => first data vector of type float
 *  y      => second data vector of type float
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the Bray-Curtis dissimilarity between vectors x and y
 */
float braycurtis_distance(float *x, float *y, int length) {
    float sum_diff = 0.0, sum_total = 0.0;
    for (int i = 0; i < length; i++) {
        sum_diff += fabsf(x[i] - y[i]);
        sum_total += fabsf(x[i] + y[i]);
    }
    return (sum_total == 0.0) ? 0.0 : (sum_diff / sum_total);
}


/*
 * canberra_distance()
 *  Calculates the Canberra distance between two data vectors.
 *  This distance is sensitive to small changes when the values are close to zero
 *  and is often used in data analysis and clustering.
 * Parameters:
 *  x      => first data vector of type float
 *  y      => second data vector of type float
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the Canberra distance between vectors x and y
 */
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

