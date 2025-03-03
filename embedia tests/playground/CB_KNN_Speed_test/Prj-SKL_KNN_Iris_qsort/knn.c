#include "knn.h"
#include <stdlib.h> // Para qsort

/*
 * KNN Implementation start
 */

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

// Función de comparación para qsort
int compare_distance_index(const void *a, const void *b) {
    DistanceIndex *da = (DistanceIndex *)a;
    DistanceIndex *db = (DistanceIndex *)b;
    return (da->distance > db->distance) - (da->distance < db->distance);
}

// Función principal para clasificación KNN
void k_neighbors_classifier_layer(k_neighbors_classifier_layer_t layer, data1d_t input, data1d_t *output) {
    DistanceIndex distances_indexes[layer.n_samples];
    uint16_t class_count[layer.n_classes];
    int i, class_id;

    // Inicializar la salida
    output->length = layer.n_classes;
    output->data = (float *)swap_alloc(sizeof(float)*output->length);

    // Calcular distancias y encontrar el número de clases
    for (i = 0; i < layer.n_samples; i++) {
        distances_indexes[i].distance = euclidean_distance(layer.neighbors_features + i * layer.n_features, input.data, layer.n_features);
        distances_indexes[i].index = i;
    }

    // Ordenar distancias e índices
    qsort(distances_indexes, layer.n_samples, sizeof(DistanceIndex), compare_distance_index);

    // Contar las etiquetas de los k vecinos más cercanos
    for (i = 0; i < layer.n_classes; i++) {
        class_count[i]=0;
    }
    for (i = 0; i < layer.n_neighbors; i++) {
        class_id = layer.neighbors_id[distances_indexes[i].index];
        class_count[class_id]++;
    }

    // Asignar la etiqueta predicha a la salida
    float c =1.0/layer.n_neighbors;
    for (i=0; i<layer.n_classes; i++) {
        output->data[i] = c*class_count[i];
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

    // Ordenar distancias e índices
    qsort(distances_indexes, layer.n_samples, sizeof(DistanceIndex), compare_distance_index);

    // Calcular el promedio de los k vecinos más cercanos
    float prom_neighbors = 0;
    for (uint16_t i = 0; i < layer.n_neighbors; i++) {
        prom_neighbors += layer.neighbors_id[distances_indexes[i].index];
    }
    prom_neighbors /= layer.n_neighbors;

    // Asignar el valor predicho a la salida
    *output->data = prom_neighbors;
}

/*
 * KNN Implementation end
 */