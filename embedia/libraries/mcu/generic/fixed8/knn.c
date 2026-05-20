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

#include <stdlib.h>
#include <string.h>
#include "knn.h"
#include "distances.h"

/** @brief Pair of distance and sample index for heap operations */
typedef struct {
    dfixed distance;
    uint16_t index;
} DistanceIndex;

/** @brief Swaps two elements of any type */
#define SWAP(a, b, type) do { \
    type temp = a; \
    a = b; \
    b = temp; \
} while (0)

/**
 * @brief Maintains max-heap property by sifting down.
 * @param heap Array of DistanceIndex elements.
 * @param heap_size Total size of the heap.
 * @param i Index to start heapify from.
 */
static inline void max_heapify(DistanceIndex *heap, int heap_size, int i) {
    int largest, left, right;
    while (1) {
        largest = i;
        left  = 2 * i + 1;
        right = 2 * i + 2;
        if (left  < heap_size && heap[left ].distance > heap[largest].distance) largest = left;
        if (right < heap_size && heap[right].distance > heap[largest].distance) largest = right;
        if (largest == i) break;
        SWAP(heap[i], heap[largest], DistanceIndex);
        i = largest;
    }
}

/**
 * @brief Builds a max-heap from an unordered array.
 * @param heap Array of DistanceIndex elements.
 * @param heap_size Total size of the heap.
 */
static inline void build_max_heap(DistanceIndex *heap, int heap_size) {
    int i;
    for (i = heap_size / 2 - 1; i >= 0; i--) {
        max_heapify(heap, heap_size, i);
    }
}

/**
 * @brief Performs KNN classification using max-heap for efficient k-nearest neighbor selection.
 * 
 * Algorithm:
 * 1. Build max-heap with first k samples
 * 2. For remaining samples, replace heap root if distance is smaller
 * 3. Count class votes from k nearest neighbors
 * 4. Return normalized class probabilities
 * 
 * Uses dfixed for distance comparisons and DFIXED_RECIP for efficient probability calculation.
 * Note: Fixed8 has limited precision (4 fractional bits), may affect accuracy.
 * 
 * @param layer KNN classifier configuration.
 * @param input Input feature vector.
 * @param output Class probabilities (normalized to sum = 1).
 */
void k_neighbors_classifier_layer(k_neighbors_classifier_layer_t layer,
                                   data1d_t input,
                                   data1d_t *output) {
    DistanceIndex heap[layer.n_neighbors];
    uint16_t class_count[layer.n_classes];

    output->length = layer.n_classes;
    output->data = (fixed *)swap_alloc(sizeof(fixed) * output->length);

    // Pre-calculate reciprocal for normalization
    dfixed inv_k = DFIXED_RECIP(layer.n_neighbors);

    // Phase 1: Fill heap with first k elements
    for (int i = 0; i < layer.n_neighbors; i++) {
        heap[i].distance = layer.distance_fn(layer.neighbors_features + i * layer.n_features,
                                              input.data, layer.n_features);
        heap[i].index = i;
    }
    build_max_heap(heap, layer.n_neighbors);

    // Phase 2: Process remaining samples
    for (int i = layer.n_neighbors; i < layer.n_samples; i++) {
        dfixed dist = layer.distance_fn(layer.neighbors_features + i * layer.n_features,
                                         input.data, layer.n_features);

        if (dist < heap[0].distance) {
            heap[0].distance = dist;
            heap[0].index = i;
            max_heapify(heap, layer.n_neighbors, 0);
        }
    }

    // Count votes
    memset(class_count, 0, sizeof(class_count));
    for (int i = 0; i < layer.n_neighbors; i++) {
        class_count[layer.neighbors_id[heap[i].index]]++;
    }

    // Calculate probabilities using fixed-point
    for (int i = 0; i < layer.n_classes; i++) {
        dfixed prob = (dfixed)class_count[i] * inv_k;
        output->data[i] = DFX2FX_RND_SAT(prob);
    }
}

/**
 * @brief Performs KNN regression using max-heap for efficient k-nearest neighbor selection.
 * 
 * Algorithm:
 * 1. Build max-heap with first k samples
 * 2. For remaining samples, replace heap root if distance is smaller
 * 3. Average target values from k nearest neighbors using DFIXED_AVG
 * 
 * Note: Fixed8 has limited precision (4 fractional bits), may affect accuracy.
 * 
 * @param layer KNN regressor configuration.
 * @param input Input feature vector.
 * @param output Predicted value (average of k nearest neighbors).
 */
void k_neighbors_regressor_layer(k_neighbors_regressor_layer_t layer,
                                  data1d_t input,
                                  data1d_t *output) {
    DistanceIndex heap[layer.n_neighbors];

    output->length = 1;
    output->data = (fixed *)swap_alloc(sizeof(fixed));

    // Phase 1: Fill heap with first k elements
    for (int i = 0; i < layer.n_neighbors; i++) {
        heap[i].distance = layer.distance_fn(layer.neighbors_features + i * layer.n_features,
                                              input.data, layer.n_features);
        heap[i].index = i;
    }
    build_max_heap(heap, layer.n_neighbors);

    // Phase 2: Process remaining samples
    for (int i = layer.n_neighbors; i < layer.n_samples; i++) {
        dfixed dist = layer.distance_fn(layer.neighbors_features + i * layer.n_features,
                                         input.data, layer.n_features);

        if (dist < heap[0].distance) {
            heap[0].distance = dist;
            heap[0].index = i;
            max_heapify(heap, layer.n_neighbors, 0);
        }
    }

    // Calculate average using DFIXED_AVG
    dfixed sum = 0;
    for (uint16_t i = 0; i < layer.n_neighbors; i++) {
        sum += layer.neighbors_id[heap[i].index];
    }
    dfixed avg = DFIXED_AVG(sum, layer.n_neighbors);
    *output->data = DFX2FX_RND_SAT(avg);
}
