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
#include "knn.h"
#include "distances.h"

/** @brief Pair of distance and sample index for heap operations */
typedef struct {
    float distance;
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
 * @param layer KNN classifier configuration.
 * @param input Input feature vector.
 * @param output Class probabilities (normalized to sum = 1).
 */
void k_neighbors_classifier_layer(k_neighbors_classifier_layer_t layer, data1d_t input, data1d_t *output) {
    DistanceIndex heap[layer.n_neighbors];
    float distance;
    uint16_t class_count[layer.n_classes];
    int i, class_id;

    output->length = layer.n_classes;
    output->data = (float *)swap_alloc(sizeof(float) * output->length);

    // Phase 1: Fill heap with first k elements
    for (i = 0; i < layer.n_neighbors; i++) {
        distance = layer.distance_fn(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        heap[i] = (DistanceIndex){distance, i};
    }
    build_max_heap(heap, layer.n_neighbors);

    // Phase 2: Process remaining samples
    for (i = layer.n_neighbors; i < layer.n_samples; i++) {
        distance = layer.distance_fn(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        if (distance < heap[0].distance) {
            heap[0] = (DistanceIndex){distance, i};
            max_heapify(heap, layer.n_neighbors, 0);
        }
    }

    // Count votes
    for (i = 0; i < layer.n_classes; i++) {
        class_count[i] = 0;
    }
    for (i = 0; i < layer.n_neighbors; i++) {
        class_id = layer.neighbors_id[heap[i].index];
        class_count[class_id]++;
    }

    // Calculate class probabilities
    float c = 1.0 / layer.n_neighbors;
    for (i = 0; i < layer.n_classes; i++) {
        output->data[i] = c * class_count[i];
    }
}

/**
 * @brief Performs KNN regression using max-heap for efficient k-nearest neighbor selection.
 * 
 * Algorithm:
 * 1. Build max-heap with first k samples
 * 2. For remaining samples, replace heap root if distance is smaller
 * 3. Average target values from k nearest neighbors
 * 
 * @param layer KNN regressor configuration.
 * @param input Input feature vector.
 * @param output Predicted value (average of k nearest neighbors).
 */
void k_neighbors_regressor_layer(k_neighbors_regressor_layer_t layer, data1d_t input, data1d_t *output) {
    DistanceIndex heap[layer.n_neighbors];

    output->length = 1;
    output->data = (float *)swap_alloc(sizeof(float));

    // Phase 1: Fill heap with first k elements
    for (int i = 0; i < layer.n_neighbors; i++) {
        float distance = layer.distance_fn(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        heap[i] = (DistanceIndex){distance, i};
    }
    build_max_heap(heap, layer.n_neighbors);

    // Phase 2: Process remaining samples
    for (int i = layer.n_neighbors; i < layer.n_samples; i++) {
        float distance = layer.distance_fn(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        if (distance < heap[0].distance) {
            heap[0] = (DistanceIndex){distance, i};
            max_heapify(heap, layer.n_neighbors, 0);
        }
    }

    // Calculate average of k nearest neighbors
    float prom_neighbors = 0;
    for (uint16_t i = 0; i < layer.n_neighbors; i++) {
        prom_neighbors += layer.neighbors_id[heap[i].index];
    }
    prom_neighbors /= layer.n_neighbors;

    *output->data = prom_neighbors;
}
