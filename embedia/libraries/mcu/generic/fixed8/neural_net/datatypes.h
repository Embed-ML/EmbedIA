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

/**
 * @file datatypes.h
 * @brief Data type definitions and macro operations for FIXED8 type
 *
 * Provides:
 * - real_t: fixed (8-bit, FIX_FRC_SZ=4)
 * - realx_t: int16_t (extended precision to prevent overflow)
 * - Macro operations via native_ops.h: MUL, ADD, INIT_ACC, CAST, FMA
 *
 * NOTE: fixed8 serves as the base for fixed-point arithmetic.
 *       The native_ops.h macros work for ALL fixed types (fixed8/16/32)
 *       by using real_t, realx_t, and FIX_FRC_SZ appropriately.
 */

#ifndef DATATYPES_H
#define DATATYPES_H


/* ========================================================================
 * METADATA
 * ======================================================================== */

#define DATA_TYPE_NAME  "FIXED8"     /**< Human-readable type name */


/* ========================================================================
 * FIXED8 CONFIGURATION
 * ======================================================================== */

/* Include the specific fixed-point implementation */
#include "fixed.h"


/* ========================================================================
 * TYPE DEFINITIONS
 * ======================================================================== */

/**
 * @typedef real_t
 * @brief Standard precision type for neural network computations
 *
 * Supported 8-bit fixed-point formats (choose one in fixed.h):
 *
 * | Format     | Integer bits | Fractional bits | Range (approx.)          | Precision (step)     | Typical use case                     |
 * |------------|--------------|-----------------|--------------------------|----------------------|--------------------------------------|
 * | Q3.4       | 3 + sign     | 4               | ±7.9375                  | ~0.0625              | Very limited range, low precision    |
 * | Q4.3       | 4 + sign     | 3               | ±15.875                  | ~0.125               | Wider range, lower precision         |
 * | Q2.5       | 2 + sign     | 5               | ±3.96875                 | ~0.03125             | Higher precision, narrow range       |
 * | Q5.2       | 5 + sign     | 2               | ±31.75                   | ~0.25                | Large range, very low precision      |
 *
 * Current selection: Q3.4 (FIX_FRC_SZ = 4)
 * Range: ≈ [-8.0, 7.9375]
 * Precision: ≈ 0.0625 (1/16)
 */
typedef fixed real_t;

/**
 * @typedef realx_t
 * @brief Extended precision type for accumulators and intermediate calculations
 *
 * For FIXED8: 16-bit signed integer with 8 fractional bits
 * Used to prevent overflow in multiply-accumulate operations
 */
typedef dfixed realx_t;


/* ========================================================================
 * NUMERIC LIMITS
 * ======================================================================== */

#define REAL_MAX        FIX_MAX       /**< Maximum representable value: 127/16 ≈ 7.9375 */
#define REAL_MIN        FIX_MIN       /**< Minimum representable value: -128/16 = -8.0 */
#define REALX_MAX       DFIX_MAX      /**< Extended precision max: 32767/256 ≈ 127.996 */
#define REALX_MIN       DFIX_MIN      /**< Extended precision min: -32768/256 = -128.0 */


/* ========================================================================
 * FUNDAMENTAL CONSTANTS
 * ======================================================================== */

#define REAL_ZERO       FIX_ZERO      /**< Zero value: 0 */
#define REAL_ONE        FIX_ONE       /**< Unity value: 16 (represents 1.0) */
#define REAL_TWO        FIX_TWO       /**< Two value: 32 (represents 2.0) */
#define REAL_HALF       FIX_HALF      /**< Half value: 8 (represents 0.5) */
#define REAL_INV_2      FIX_INV_2     /**< 1/2! = 0.5 */

/* ========================================================================
 * MATHEMATICAL CONSTANTS
 * ======================================================================== */

#define REAL_PI         FIX_PI        /**< π ≈ 3.14159 */
#define REAL_2PI        FIX_2PI       /**< 2π ≈ 6.28318 */
#define REAL_PI_2       FIX_PI_INV2   /**< π/2 ≈ 1.57079 */
#define REAL_PI_4       FIX_PI_INV4   /**< π/4 ≈ 0.78539 */
#define REAL_E          FIX_E         /**< e (Euler's number) ≈ 2.71828 */


/* ========================================================================
 * OPERATION MACROS (Type-agnostic interface)
 * ======================================================================== */

/* Include the generic operation layer */
#include "native_ops.h"

#endif /* DATATYPES_H */