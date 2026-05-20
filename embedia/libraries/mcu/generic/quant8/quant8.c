/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 */

#include <math.h>
#include "quant8.h"

/**
 * @brief Calculate quantization parameters from float array
 */
void quantize_param(float *values, int size, qparam_t *qp) {
    if (!values || !qp || size <= 0) {
        qp->scale_q = QUANT_SCALE_ONE;
        qp->zero_point = 0;
        return;
    }

    float min_val = values[0];
    float max_val = values[0];

    for (int i = 1; i < size; ++i) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }

    float range = max_val - min_val;
    float float_scale = range / (float)Q_RANGE;

    if (float_scale < 1e-8f) {
        float_scale = 1e-8f;
    }

    qp->scale_q = (uint16_t)(float_scale * (float)QUANT_SCALE_ONE + 0.5f);

    if (qp->scale_q == 0) {
        qp->scale_q = 1;
    }
    if (qp->scale_q > QUANT_SCALE_MAX) {
        qp->scale_q = QUANT_SCALE_MAX;
    }

    float zp_float = -min_val / float_scale;
    qp->zero_point = (int8_t)roundf(zp_float);

    if (qp->zero_point > Q_MAX) qp->zero_point = Q_MAX;
    if (qp->zero_point < Q_MIN) qp->zero_point = Q_MIN;
}

/**
 * @brief Quantize float array to int8 array
 */
void quantize_vec(float values[], quant8 qvalues[], int size, qparam_t qp) {
    for (int i = 0; i < size; ++i) {
        float x_scaled = values[i] * (float)QUANT_SCALE_ONE;
        float divided = x_scaled / (float)qp.scale_q;
        int32_t quantized = (int32_t)roundf(divided) + qp.zero_point;
        qvalues[i] = Q_CLAMP(quantized);
    }
}

/**
 * @brief Dequantize int8 array to fixed array
 */
void dequantize_vec(quant8 qvalues[], fixed values[], int size, qparam_t qp) {
    int i;
    for (i = 0; i < size; i++) {
        values[i] = DEQUANTIZE_FIXED(qvalues[i], qp);
    }
}
