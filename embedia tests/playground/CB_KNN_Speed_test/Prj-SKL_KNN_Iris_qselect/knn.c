#include "knn.h"
#include <stdlib.h>

// Función para calcular la distancia euclidiana
static float euclidean_distance(float *x_train, float *x, uint16_t n_features) {
    float distance = 0;
    float diff;
    for (uint16_t i = 0; i < n_features; i++) {
        diff = x_train[i] - x[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

// Estructura para almacenar distancias e índices
typedef struct {
    float distance;
    int index;
} DistanceIndex;

// Función de comparación para quickselect
int compare_distance_index(const void *a, const void *b) {
    DistanceIndex *da = (DistanceIndex *)a;
    DistanceIndex *db = (DistanceIndex *)b;
    return (da->distance > db->distance) - (da->distance < db->distance);
}

// Función de partición para quickselect
int partition(DistanceIndex *arr, int low, int high) {
    DistanceIndex pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (compare_distance_index(&arr[j], &pivot) <= 0) {
            i++;
            DistanceIndex temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    DistanceIndex temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;
    return i + 1;
}

// Función quickselect para encontrar los k elementos más pequeños
void quickselect(DistanceIndex *arr, int low, int high, int k) {
    if (low < high) {
        int pi = partition(arr, low, high);
        if (pi == k) {
            return;
        } else if (pi < k) {
            quickselect(arr, pi + 1, high, k);
        } else {
            quickselect(arr, low, pi - 1, k);
        }
    }
}

// Función principal para clasificación KNN
void k_neighbors_classifier_layer(k_neighbors_classifier_layer_t layer, data1d_t input, data1d_t *output) {
    DistanceIndex distances_indexes[layer.n_samples];
    uint16_t class_count[layer.n_classes];
    int i, class_id;

    // Inicializar la salida
    output->length = layer.n_classes;
    output->data = (float *)swap_alloc(sizeof(float) * output->length);

    // Calcular distancias y encontrar el número de clases
    for (i = 0; i < layer.n_samples; i++) {
        distances_indexes[i].distance = euclidean_distance(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        distances_indexes[i].index = i;
    }

    // Encontrar los k vecinos más cercanos usando quickselect
    quickselect(distances_indexes, 0, layer.n_samples - 1, layer.n_neighbors - 1);

    // Contar las etiquetas de los k vecinos más cercanos
    for (i = 0; i < layer.n_classes; i++) {
        class_count[i] = 0;
    }
    for (i = 0; i < layer.n_neighbors; i++) {
        class_id = layer.neighbors_id[distances_indexes[i].index];
        class_count[class_id]++;
    }

    // Asignar la etiqueta predicha a la salida
    float c = 1.0 / layer.n_neighbors;
    for (i = 0; i < layer.n_classes; i++) {
        output->data[i] = c * class_count[i];
    }
}

// Función principal para regresión KNN
void k_neighbors_regressor_layer(k_neighbors_regressor_layer_t layer, data1d_t input, data1d_t *output) {
    DistanceIndex distances_indexes[layer.n_samples];

    // Inicializar la salida
    output->length = 1;
    output->data = (float *)swap_alloc(sizeof(float));

    // Calcular distancias
    for (int i = 0; i < layer.n_samples; i++) {
        distances_indexes[i].distance = euclidean_distance(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        distances_indexes[i].index = i;
    }

    // Encontrar los k vecinos más cercanos usando quickselect
    quickselect(distances_indexes, 0, layer.n_samples - 1, layer.n_neighbors - 1);

    // Calcular el promedio de los k vecinos más cercanos
    float prom_neighbors = 0;
    for (uint16_t i = 0; i < layer.n_neighbors; i++) {
        prom_neighbors += layer.neighbors_id[distances_indexes[i].index];
    }
    prom_neighbors /= layer.n_neighbors;

    // Asignar el valor predicho a la salida
    *output->data = prom_neighbors;
}
