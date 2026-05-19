#ifndef QUANT8_H
#define QUANT8_H
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

#include <stdint.h>
#include <math.h>
#include "fixed.h"

/* Cuantización de 8 bits con escala fija Qn.m.
   Se define `scale` como punto fijo de Q_FRAC_BITS bits fraccionales.
   El valor cuantizado es int8_t. */

#define Q_INT_BITS  FIX_INT_SZ
#define Q_FRAC_BITS FIX_FRC_SZ

#if (Q_INT_BITS + Q_FRAC_BITS) != 32
    #error "Qn.m debe sumar 32 bits (int32_t)"
#endif

typedef int8_t quant8;

typedef struct {
    int32_t scale;      // Escala en punto fijo Q_FRAC_BITS (int32_t)
    int8_t zero_point;  // Punto cero
} qparam_t;

#define Q_SCALE      (1 << Q_FRAC_BITS)
#define Q_MAX_VAL    ((1 << (Q_INT_BITS - 1)) - 1)
#define Q_MIN_VAL    (-(1 << (Q_INT_BITS - 1)))

#define Q_MIN (-128)
#define Q_MAX 127

// Clamp a rango int8
#define Q_CLAMP(qv) ((quant8)((qv > Q_MAX) ? Q_MAX : ((qv < Q_MIN) ? Q_MIN : (qv))))

// ===============================
// Cuantización/descuantización en punto flotante
// ===============================

// Entrada: float, Salida: int8_t
#define QUANTIZE(val, qp) \
    Q_CLAMP((int)(roundf((qp.zero_point + ((val) / ((float)(qp.scale) / Q_SCALE))))))

// Entrada: int8_t, Salida: float
#define DEQUANTIZE(qval, qp) \
    (((float)((qval) - (qp.zero_point))) * ((float)(qp.scale) / Q_SCALE))

// ===============================
// Cuantización/descuantización 100% en enteros
// ===============================

// Entrada: int32_t en Qn.m, Salida: int8_t
#define QUANTIZE_FIXED(val_qm, qp) \
    Q_CLAMP((((val_qm) + ((qp.scale) >> 1)) / (qp.scale) + (qp.zero_point)))

// Entrada: int8_t, Salida: int32_t en Qn.m
#define DEQUANTIZE_FIXED(qval, qp) \
    (((int32_t)((qval) - (qp.zero_point))) * (qp.scale))


//////////////////////////////////// Funciones ////////////////////////////////////////
// Conversiones
static inline int32_t float_to_q(float f) {
    return (int32_t)(f * Q_SCALE);
}

static inline float q_to_float(int32_t q) {
    return (float)q / Q_SCALE;
}

// Multiplicación con ajuste de formato
static inline int32_t q_mul(int32_t a, int32_t b) {
    return ((int64_t)a * b) >> Q_FRAC_BITS;
}

// Suma con saturación
static inline int32_t q_add(int32_t a, int32_t b) {
    int64_t tmp = (int64_t)a + b;
    if (tmp > Q_MAX_VAL) return Q_MAX_VAL;
    if (tmp < Q_MIN_VAL) return Q_MIN_VAL;
    return (int32_t)tmp;
}


#ifdef __cplusplus
extern "C" {
#endif

    // Calcula parámetros de cuantización
    void quantize_param(float *values, int size, qparam_t *qp);

    // Cuantiza/descuantiza vectores
    void quantize_vec(float values[], quant8 qvalues[], int size, qparam_t qp);
    void dequantize_vec(quant8 qvalues[], float values[], int size, qparam_t qp);

    // Operaciones combinadas (MAC)
    int32_t mul_add_vec(quant8 a[], qparam_t qa, quant8 b[], qparam_t qb, int size);

#ifdef __cplusplus
}
#endif

#endif