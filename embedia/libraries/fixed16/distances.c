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
fixed euclidean_distance(fixed *x, fixed *y, int length) {
    dfixed distance = 0, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = y[i] - x[i];
        distance += DFIXED_MUL(diff, diff);
    }
    if (distance > (FIX_MAX*FIX_MAX)){
        return FIX_MAX;
    }

    return fixed_sqrt(DFIXED_TO_FIXED(distance));
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
fixed manhattan_distance(fixed *x, fixed *y, int length) {
    fixed distance = 0.0f;
    int i;

    for (i = 0; i < length; i++) {
        distance += FIXED_ABS((y[i] - x[i]));
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
fixed chebyshev_distance(fixed *x, fixed *y, int length) {
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
fixed minkowski_distance(fixed *x, fixed *y, int length, fixed p) {
    dfixed distance = FIX_ZERO, diff;
    int i;

    for (i = 0; i < length; i++) {
        diff = FIXED_ABS(y[i] - x[i]);
        distance += fixed_pow(diff, p);
    }
    return fixed_pow(distance, FIXED_DIV(FIX_ONE, p));
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
fixed cosine_distance(fixed *x, fixed *y, int length) {
    dfixed dot_product = FIX_ZERO;
    dfixed norm_x = FIX_ZERO;
    dfixed norm_y = FIX_ZERO;
    dfixed xx, yy;
    int i;

    for (i = 0; i < length; i++) {
        xx = FIXED_TO_DFIXED(x[i]);
        yy = FIXED_TO_DFIXED(y[i]);
        dot_product += DFIXED_MUL(xx, yy);
        norm_x += DFIXED_MUL(xx, xx);
        norm_y += DFIXED_MUL(yy, yy);
    }

    // Evitar división por cero
    if (norm_x == FIX_ZERO || norm_y == FIX_ZERO) {
        return FIX_ONE;
    }

    // Se convierte la norma al mismo formato y se ajusta la escala
    fixed sqrt_norm_x = fixed_sqrt(DFIXED_TO_FIXED(norm_x));
    fixed sqrt_norm_y = fixed_sqrt(DFIXED_TO_FIXED(norm_y));

    // Evitar división por cero después de la raíz cuadrada
    if (sqrt_norm_x == FIX_ZERO || sqrt_norm_y == FIX_ZERO) {
        return FIX_ONE;
    }

    // Usar DFIXED_DIV para mayor precisión en la división
    dfixed denominator = FIXED_TO_DFIXED(FIXED_MUL(sqrt_norm_x, sqrt_norm_y));

    if (denominator == FIX_ZERO) {
        return FIX_ONE;
    }

    fixed similarity = DFIXED_TO_FIXED(DFIXED_DIV(dot_product, denominator));

    // Clamping para evitar valores fuera de rango debido a errores numéricos
    if (similarity > FIX_ONE) similarity = FIX_ONE;
    if (similarity < -FIX_ONE) similarity = -FIX_ONE;

    return FIX_ONE - similarity;
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
fixed braycurtis_distance(fixed *x, fixed *y, int length) {
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
fixed canberra_distance(fixed *x, fixed *y, int length) {
    fixed denom, distance = 0;
    int i;

    for (i = 0; i < length; i++) {
        denom = FIXED_ABS(x[i]) + FIXED_ABS(y[i]); // Denominador
        if (denom != 0) {
            fixed diff = FIXED_ABS(x[i] - y[i]); // Diferencia absoluta
            distance += DFIXED_DIV(diff, denom);  // Sumar la fracción
        }
    }

    return distance;
}
