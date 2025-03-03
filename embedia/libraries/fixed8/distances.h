#ifndef _DISTANCES_H
#define _DISTANCES_H

#include "common.h"



/*********************************** Distance Functions for vectors ******************************************/

/*
 * euclidean_distance()
 *  Calculates the Euclidean distance between two data vectors.
 * Parameters:
 *  x      => first data vector of type fixed
 *  y      => second data vector of type fixed
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the Euclidean distance between vectors x and y
 */
fixed euclidean_distance(fixed *x, fixed *y, int length);

/*
 * manhattan_distance()
 *  Calculates the Manhattan (L1) distance between two data vectors.
 * Parameters:
 *  x      => first data vector of type fixed
 *  y      => second data vector of type fixed
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the Manhattan distance between vectors x and y
 */
fixed manhattan_distance(fixed *x, fixed *y, int length);

/*
 * chebyshev_distance()
 *  Calculates the Chebyshev (L∞) distance between two data vectors.
 *  This is the maximum absolute difference between any pair of elements.
 * Parameters:
 *  x      => first data vector of type fixed
 *  y      => second data vector of type fixed
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the Chebyshev distance between vectors x and y
 */
fixed chebyshev_distance(fixed *x, fixed *y, int length);

/*
 * minkowski_distance()
 *  Calculates the Minkowski distance between two data vectors.
 *  This is a generalization of Euclidean and Manhattan distances.
 * Parameters:
 *  x      => first data vector of type fixed
 *  y      => second data vector of type fixed
 *  length => length of the vectors (both must have the same length)
 *  p      => the order of the Minkowski distance (p >= 1)
 *             p=1: Manhattan distance
 *             p=2: Euclidean distance
 *             p→∞: Chebyshev distance
 * Returns:
 *  result - value of the Minkowski distance between vectors x and y
 */
fixed minkowski_distance(fixed *x, fixed *y, int length, fixed p);

/*
 * cosine_distance()
 *  Calculates the cosine distance between two data vectors.
 *  This measures the angle between two vectors regardless of magnitude.
 * Parameters:
 *  x      => first data vector of type fixed
 *  y      => second data vector of type fixed
 *  length => length of the vectors (both must have the same length)
 * Returns:
 *  result - value of the cosine distance between vectors x and y (range 0-2)
 *           0: identical direction, 1: orthogonal, 2: opposite direction
 */
fixed cosine_distance(fixed *x, fixed *y, int length);

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
fixed braycurtis_distance(fixed *x, fixed *y, int length);

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
fixed canberra_distance(fixed *x, fixed *y, int length);


#endif
