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
#include <math.h>
#include "quant8.h"

void quantize_param(float *values, int size, qparam_t *qp) {
    float min_val = values[0];
    float max_val = values[0];

    // 1. Encontrar rango
    for (int i = 1; i < size; ++i) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }

    // 2. Calcular parámetros de cuantización
    float float_scale = (max_val - min_val) / (Q_MAX_VAL - Q_MIN_VAL);

    // Prevenir división por cero
    if (float_scale < 1e-8f) float_scale = 1e-8f;

    // Convertir a punto fijo
    qp->scale = float_to_q(float_scale);

    qp->scale = float_to_q(1.0f / float_scale);
    //qp->frac_bits = Q_FRAC_BITS; // Definido globalmente (ej. 8 o 16)

    qp->zero_point = (int8_t)roundf(Q_MIN_VAL - min_val / float_scale);

    // Asegurar que zero_point esté en rango
    qp->zero_point = (qp->zero_point > Q_MAX_VAL) ? Q_MAX_VAL :
                     (qp->zero_point < Q_MIN_VAL) ? Q_MIN_VAL : qp->zero_point;
}

void quantize_vec(float values[], quant8 qvalues[], int size, qparam_t qp) {
    for (int i = 0; i < size; ++i) {
        // Convertir a punto fijo y multiplicar por scale_fixed
        int32_t input_fixed = float_to_q(values[i]);
        int32_t quantized = ((int64_t)input_fixed * qp.scale) >> Q_FRAC_BITS;
        quantized += qp.zero_point;

        // Saturar
        qvalues[i] = (quantized > Q_MAX_VAL) ? Q_MAX_VAL :
                    (quantized < Q_MIN_VAL) ? Q_MIN_VAL : (quant8)quantized;
    }
}

int32_t mul_add_vec(quant8 a[], qparam_t qa, quant8 b[], qparam_t qb, int size) {
    int32_t sum = 0;

    for (int i = 0; i < size; ++i) {
        // Descuantizar usando shifts en lugar de multiplicación
        int32_t deq_a = (a[i] - qa.zero_point) * qa.scale;
        int32_t deq_b = (b[i] - qb.zero_point) * qb.scale;

        // Multiplicación con ajuste de precisión
        sum += ((int64_t)deq_a * deq_b) >> Q_FRAC_BITS;

        // Opcional: saturación periódica para evitar overflow
        if (sum > (1 << 30)) sum = (1 << 30);
        if (sum < -(1 << 30)) sum = -(1 << 30);
    }

    return sum;
}
