#include "knn.h"
#include <stdlib.h>

// Estructura para almacenar distancias e índices
typedef struct {
    fixed distance;
    uint16_t index;
} DistanceIndex;


// Macro para intercambiar dos elementos
#define SWAP(a, b, type) do { \
    type temp = a; \
    a = b; \
    b = temp; \
} while (0)

// Función para mantener la propiedad de heap máximo
static inline void max_heapify(DistanceIndex *heap, int heap_size, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < heap_size && heap[left].distance > heap[largest].distance) {
        largest = left;
    }
    if (right < heap_size && heap[right].distance > heap[largest].distance) {
        largest = right;
    }
    if (largest != i) {
        SWAP(heap[i], heap[largest], DistanceIndex); // Usando la macro SWAP
        max_heapify(heap, heap_size, largest);
    }
}

// Función para construir un heap máximo
static inline void build_max_heap(DistanceIndex *heap, int heap_size) {
    int i;
    for (i = heap_size / 2 - 1; i >= 0; i--) {
        max_heapify(heap, heap_size, i);
    }
}

// Función principal para clasificación KNN
void k_neighbors_classifier_layer(k_neighbors_classifier_layer_t layer, data1d_t input, data1d_t *output) {
    DistanceIndex heap[layer.n_neighbors];
    fixed distance;
    uint16_t class_count[layer.n_classes];
    int i, class_id;

    // Inicializar la salida
    output->length = layer.n_classes;
    output->data = (fixed *)swap_alloc(sizeof(fixed) * output->length);

    // Fase 1: Llenar el heap con los primeros k elementos
    for (i = 0; i < layer.n_neighbors; i++) {
        distance = layer.distance_fn(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        heap[i] = (DistanceIndex){distance, i};
    }
    build_max_heap(heap, layer.n_neighbors); // Construir el heap máximo

    // Fase 2: Procesar el resto de los elementos
    for (i = layer.n_neighbors; i < layer.n_samples; i++) {
        distance = layer.distance_fn(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        if (distance < heap[0].distance) {
            heap[0] = (DistanceIndex){distance, i};
            max_heapify(heap, layer.n_neighbors, 0); // Ajustar el heap
        }
    }

    // Contar las etiquetas de los k vecinos más cercanos
    for (i = 0; i < layer.n_classes; i++) {
        class_count[i] = 0;
    }
    for (i = 0; i < layer.n_neighbors; i++) {
        class_id = layer.neighbors_id[heap[i].index];
        class_count[class_id]++;
    }

    // Asignar la etiqueta predicha a la salida
    fixed c = FIXED_DIV(FIX_ONE, INT_TO_FIXED(layer.n_neighbors));
    for (i = 0; i < layer.n_classes; i++) {
        output->data[i] = FIXED_MUL(c, INT_TO_FIXED(class_count[i]));
    }
}

// Función principal para regresión KNN
void k_neighbors_regressor_layer(k_neighbors_regressor_layer_t layer, data1d_t input, data1d_t *output) {
    DistanceIndex heap[layer.n_neighbors];

    // Inicializar la salida
    output->length = 1;
    output->data = (fixed *)swap_alloc(sizeof(fixed));

    // Fase 1: Llenar el heap con los primeros k elementos
    for (int i = 0; i < layer.n_neighbors; i++) {
        fixed distance = euclidean_distance(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        heap[i] = (DistanceIndex){distance, i};
    }
    build_max_heap(heap, layer.n_neighbors); // Construir el heap máximo

    // Fase 2: Procesar el resto de los elementos
    for (int i = layer.n_neighbors; i < layer.n_samples; i++) {
        fixed distance = euclidean_distance(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        if (distance < heap[0].distance) {
            heap[0] = (DistanceIndex){distance, i};
            max_heapify(heap, layer.n_neighbors, 0); // Ajustar el heap
        }
    }

    // Calcular el promedio de los k vecinos más cercanos
    fixed prom_neighbors = 0;
    for (uint16_t i = 0; i < layer.n_neighbors; i++) {
        prom_neighbors += layer.neighbors_id[heap[i].index];
    }
    prom_neighbors = FIXED_DIV(prom_neighbors,INT_TO_FIXED(layer.n_neighbors));

    // Asignar el valor predicho a la salida
    *output->data = prom_neighbors;
}
