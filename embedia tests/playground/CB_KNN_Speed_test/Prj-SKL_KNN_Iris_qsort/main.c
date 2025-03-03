#include <stdio.h>
#include <windows.h>  // Necesario para QueryPerformanceCounter
#include "neural_net.h"
#include "skl_knn_iris_model.h"
#include "example_file.h"

#define NUM_RUNS 100000  // Aumentar iteraciones para mejorar precisión

data1d_t input = { INPUT_LENGTH, NULL };
data1d_t results;

int main(void) {
    LARGE_INTEGER frequency, start_time, end_time;
    double total_time, avg_time;

    // Obtener la frecuencia del contador de alto rendimiento
    QueryPerformanceFrequency(&frequency);

    input.data = sample_data;
    model_init();

    // Capturar tiempo de inicio
    QueryPerformanceCounter(&start_time);

    // Ejecutar la predicción varias veces
    for (int i = 0; i < NUM_RUNS; i++) {
        model_predict_class(input, &results);
    }

    // Capturar tiempo de finalización
    QueryPerformanceCounter(&end_time);

    // Calcular tiempo total en milisegundos
    total_time = (double)(end_time.QuadPart - start_time.QuadPart) * 1000.0 / frequency.QuadPart;
    avg_time = total_time / NUM_RUNS;

    // Imprimir resultados
    printf("KNN con QuickSort\n");
    printf("Tiempo total de prediccion (%d ejecuciones): %.5f ms\n", NUM_RUNS, total_time);
    printf("Tiempo promedio por prediccion.............: %.5f ms\n", avg_time);

    return 0;
}
