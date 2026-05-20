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
 */

#include <stdint.h>
#include <math.h>
#include "fixed.h"

/**
 * @file quant8.h
 * @brief 8-bit asymmetric quantization optimized for resource-constrained MCUs
 *
 * @details
 * Quantization format:
 * - Values: int8_t range [-128, 127]
 * - Scale: Q0.15 format (direct scale)
 * - Zero point: int8_t offset
 *
 * Key optimization: Stores DIRECT scale to convert division into multiplication
 *
 * Quantization formula:
 *   q = round(x / scale) + zero_point
 *
 * Dequantization formula:
 *   x = (q - zero_point) * scale
 */

/* ============================================================================
 * QUANTIZATION TYPE CONFIGURATION
 * ============================================================================ */

typedef int8_t quant8;

#define Q_MIN       (-128)  ///< Minimum quantized value
#define Q_MAX       127     ///< Maximum quantized value
#define Q_RANGE     (Q_MAX - Q_MIN + 1)  ///< Quantization range (256)

/* ============================================================================
 * QUANTIZATION SCALE FORMAT (Q0.15)
 * ============================================================================ */

#define QUANT_SCALE_FRAC_BITS   15
#define QUANT_SCALE_ONE         (1 << QUANT_SCALE_FRAC_BITS)  // 32768
#define QUANT_SCALE_HALF        (1 << (QUANT_SCALE_FRAC_BITS - 1))  // 16384
#define QUANT_SCALE_MAX         65535

/* ============================================================================
 * SHIFT ALIGNMENT (from fixed.h)
 * ============================================================================ */


//#define DFIX_FRC_SZ (FIX_FRC_SZ * 2)

#define FX_TO_SCALE_SHIFT   (QUANT_SCALE_FRAC_BITS - FIX_FRC_SZ)
#define SCALE_TO_FX_SHIFT   (QUANT_SCALE_FRAC_BITS - FIX_FRC_SZ)
//#define DFX_TO_SCALE_SHIFT  (QUANT_SCALE_FRAC_BITS - DFIX_FRC_SZ)
#define SCALE_TO_DFX_SHIFT  (DFIX_FRC_SZ - QUANT_SCALE_FRAC_BITS)

/* ============================================================================
 * DATA TYPES
 * ============================================================================ */

typedef struct {
    uint16_t scale_q;    ///< Direct scale in Q0.15 (0 to 65535)
    int8_t   zero_point; ///< Zero point offset [-128, 127]
} qparam_t;

/* ============================================================================
 * HELPER MACROS
 * ============================================================================ */

#define Q_CLAMP(qv) \
    ((quant8)((qv) > Q_MAX ? Q_MAX : ((qv) < Q_MIN ? Q_MIN : (qv))))

/* ============================================================================
 * FLOATING POINT QUANTIZATION (for setup/testing only)
 * ============================================================================ */

#define QUANTIZE(val, qp) \
    Q_CLAMP((int)(roundf((val) / ((float)(qp).scale_q / QUANT_SCALE_ONE) + (qp).zero_point)))

#define DEQUANTIZE(qval, qp) \
    (((float)((qval) - (qp).zero_point)) * ((float)(qp).scale_q / QUANT_SCALE_ONE))

/* ============================================================================
 * FIXED POINT QUANTIZATION - Q8.8
 * ============================================================================ */

#define DEQUANTIZE_FIXED_64(qval, qp) \
    ((fixed)( \
        (((int64_t)((qval) - (qp).zero_point) * (int64_t)(qp).scale_q) \
         + (1 << (SCALE_TO_FX_SHIFT - 1))) >> SCALE_TO_FX_SHIFT \
    ))

#define DEQUANTIZE_FIXED(qval, qp) \
    ((fixed)( \
        (((int16_t)((qval) - (qp).zero_point) * (int32_t)(qp).scale_q) \
         + (1 << (SCALE_TO_FX_SHIFT - 1))) >> SCALE_TO_FX_SHIFT \
    ))

#define DEQUANTIZE_DFIXED(qval, qp) \
    ((dfixed)( \
        (((int32_t)((qval) - (qp).zero_point) * (int32_t)(qp).scale_q) \
         << SCALE_TO_DFX_SHIFT) \
    ))

/* ============================================================================
 * DOUBLE FIXED POINT QUANTIZATION - Q16.16
 * ============================================================================ */

#define QUANTIZE_DFIXED(val_dfx, qp) \
    Q_CLAMP( \
        ((int32_t)( \
            ((((int64_t)(val_dfx) >> DFX_TO_SCALE_SHIFT) * QUANT_SCALE_ONE) \
             / (qp).scale_q + QUANT_SCALE_HALF) >> QUANT_SCALE_FRAC_BITS \
        ) + (qp).zero_point) \
    )


/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

void quantize_param(float *values, int size, qparam_t *qp);

void quantize_vec(float values[], quant8 qvalues[], int size, qparam_t qp);

void dequantize_vec(quant8 qvalues[], fixed values[], int size, qparam_t qp);


#ifdef __cplusplus
}
#endif

#endif // QUANT8_H