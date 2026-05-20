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
 * @file realtype.h
 * @brief Data type definitions and macro operations for FLOAT type
 * 
 * Implements three-tier type abstraction:
 * - storage_t: Model parameter storage (weights, kernels)
 * - compute_t: Runtime calculations (activations, intermediate results)
 * - computex_t: Extended precision accumulators (prevent overflow)
 * 
 * Legacy types real_t/realx_t maintained for backward compatibility.
 * 
 * NOTE: For FLOAT, all three types are the same (native float operations).
 */

#ifndef REALTYPE_H
#define REALTYPE_H

#include <math.h>

/* ========================================================================
 * METADATA
 * ======================================================================== */

#define DATA_TYPE_NAME  "FLOAT"     /**< Human-readable type name */


/* ========================================================================
 * TYPE ABSTRACTION HIERARCHY
 * 
 * Three-tier type system for efficient neural network inference:
 * 
 * ┌─────────────┬──────────────┬─────────────┬──────────────────────────┐
 * │ Type        │ Purpose      │ Size        │ Example Usage            │
 * ├─────────────┼──────────────┼─────────────┼──────────────────────────┤
 * │ storage_t   │ Memory       │ Minimal     │ Model weights in flash   │
 * │ compute_t   │ Arithmetic   │ Balanced    │ Layer activations        │
 * │ computex_t  │ Accumulation │ Extended    │ Dot product sums         │
 * └─────────────┴──────────────┴─────────────┴──────────────────────────┘
 * 
 * Data type mappings across representations:
 * 
 * FLOAT:       storage_t = float,  compute_t = float,  computex_t = float
 * FIXED32:     storage_t = int32,  compute_t = int32,  computex_t = int64
 * FIXED16:     storage_t = int16,  compute_t = int16,  computex_t = int32
 * QUANT8:      storage_t = int8,   compute_t = int16,  computex_t = int32
 * FULL_QUANT8: storage_t = int8,   compute_t = int8,   computex_t = int32
 * 
 * Current configuration: FLOAT
 * ======================================================================== */

/**
 * @typedef storage_t
 * @brief Model parameter storage type (weights, kernels)
 * 
 * Purpose: Minimize memory/flash footprint
 * Usage: Static model data, weight arrays
 * For FLOAT: 32-bit IEEE 754 floating point
 */
typedef float storage_t;

/**
 * @typedef compute_t
 * @brief Runtime computation type (activations, intermediate results)
 * 
 * Purpose: Arithmetic operations with sufficient precision
 * Usage: Layer inputs/outputs, bias values, activations
 * For FLOAT: 32-bit IEEE 754 floating point
 */
typedef float compute_t;

/**
 * @typedef computex_t
 * @brief Extended precision accumulator type
 * 
 * Purpose: Prevent overflow in multiply-accumulate operations
 * Usage: Dot products, convolution accumulators
 * For FLOAT: 32-bit IEEE 754 (same as compute_t, sufficient range)
 */
typedef float computex_t;

/* ========================================================================
 * TYPE CONVERSIONS
 * ======================================================================== */

// Storage ↔ Compute (no conversion needed for float)
#define ST2CO(s, qp)     (s)                       /**< Load weight: float → float (no-op) */
#define CO2ST(c, qp)     (c)                       /**< Store result: float → float (no-op) */

// Compute ↔ ComputeX (no conversion needed for float)
#define CO2CX(c)         (c)                       /**< Promote: float → float (no-op) */
#define CX2CO(cx)        (cx)                      /**< Demote: float → float (no-op) */
#define CX2CO_SAT(cx)    (cx)                      /**< Demote with saturation (no-op) */

/* ========================================================================
 * BACKWARD COMPATIBILITY
 * ======================================================================== */

typedef compute_t real_t;      /**< Legacy alias (deprecated, use compute_t) */
typedef computex_t realx_t;    /**< Legacy alias (deprecated, use computex_t) */

#define R2RX(val)        CO2CX(val)      /**< Legacy (use CO2CX) */
#define RX2R(val)        CX2CO(val)      /**< Legacy (use CX2CO) */
#define RX2R_SAT(val)    CX2CO_SAT(val)  /**< Legacy (use CX2CO_SAT) */

#define SAT2R(val)       (val)           /**< Legacy saturation (no-op for float) */
#define SAT2RX(val)      (val)           /**< Legacy saturation (no-op for float) */

/* ========================================================================
 * NUMERIC LIMITS
 * ======================================================================== */

// Storage limits
#define STORAGE_MAX      INFINITY     /**< Maximum storage value (float) */
#define STORAGE_MIN     -INFINITY     /**< Minimum storage value (float) */

// Compute limits
#define COMPUTE_MAX      INFINITY     /**< Maximum compute value (float) */
#define COMPUTE_MIN     -INFINITY     /**< Minimum compute value (float) */
#define COMPUTEX_MAX     INFINITY     /**< Maximum computex value (float) */
#define COMPUTEX_MIN    -INFINITY     /**< Minimum computex value (float) */

// Legacy aliases
#define REAL_MAX         COMPUTE_MAX  /**< Legacy (use COMPUTE_MAX) */
#define REAL_MIN         COMPUTE_MIN  /**< Legacy (use COMPUTE_MIN) */
#define REALX_MAX        COMPUTEX_MAX /**< Legacy (use COMPUTEX_MAX) */
#define REALX_MIN        COMPUTEX_MIN /**< Legacy (use COMPUTEX_MIN) */


/* ========================================================================
 * FUNDAMENTAL CONSTANTS
 * ======================================================================== */

// Storage constants
#define STORAGE_ZERO     0.0f         /**< Zero value (float) */

// Compute constants
#define COMPUTE_ZERO     0.0f         /**< Zero value: 0.0 */
#define COMPUTE_ONE      1.0f         /**< Unity value: 1.0 */
#define COMPUTE_TWO      2.0f         /**< Two value: 2.0 */
#define COMPUTE_HALF     0.5f         /**< Half value: 0.5 */
#define COMPUTE_INV_2    0.5f         /**< 1/2! = 0.5 */

// ComputeX constants
#define COMPUTEX_ZERO    0.0f         /**< Zero value: 0.0 */
#define COMPUTEX_ONE     1.0f         /**< Unity value: 1.0 */
#define COMPUTEX_TWO     2.0f         /**< Two value: 2.0 */
#define COMPUTEX_HALF    0.5f         /**< Half value: 0.5 */
#define COMPUTEX_INV_2   0.5f         /**< 1/2! = 0.5 */

// Legacy aliases
#define REAL_ZERO        COMPUTE_ZERO  /**< Legacy (use COMPUTE_ZERO) */
#define REAL_ONE         COMPUTE_ONE   /**< Legacy (use COMPUTE_ONE) */
#define REAL_TWO         COMPUTE_TWO   /**< Legacy (use COMPUTE_TWO) */
#define REAL_HALF        COMPUTE_HALF  /**< Legacy (use COMPUTE_HALF) */
#define REAL_INV_2       COMPUTE_INV_2 /**< Legacy (use COMPUTE_INV_2) */

#define REALX_ZERO       COMPUTEX_ZERO   /**< Legacy (use COMPUTEX_ZERO) */
#define REALX_ONE        COMPUTEX_ONE    /**< Legacy (use COMPUTEX_ONE) */
#define REALX_TWO        COMPUTEX_TWO    /**< Legacy (use COMPUTEX_TWO) */
#define REALX_HALF       COMPUTEX_HALF   /**< Legacy (use COMPUTEX_HALF) */
#define REALX_INV_2      COMPUTEX_INV_2  /**< Legacy (use COMPUTEX_INV_2) */

/* ========================================================================
 * MATHEMATICAL CONSTANTS
 * ======================================================================== */

// Compute constants
#define COMPUTE_PI       3.14159265358979323846f    /**< π ≈ 3.14159 */
#define COMPUTE_2PI      6.28318530717958647692f    /**< 2π ≈ 6.28318 */
#define COMPUTE_PI_2     1.57079632679489661923f    /**< π/2 ≈ 1.57079 */
#define COMPUTE_PI_4     0.78539816339744830962f    /**< π/4 ≈ 0.78539 */
#define COMPUTE_E        2.71828182845904523536f    /**< e (Euler's number) ≈ 2.71828 */

// Legacy aliases
#define REAL_PI          COMPUTE_PI    /**< Legacy (use COMPUTE_PI) */
#define REAL_2PI         COMPUTE_2PI   /**< Legacy (use COMPUTE_2PI) */
#define REAL_PI_2        COMPUTE_PI_2  /**< Legacy (use COMPUTE_PI_2) */
#define REAL_PI_4        COMPUTE_PI_4  /**< Legacy (use COMPUTE_PI_4) */
#define REAL_E           COMPUTE_E     /**< Legacy (use COMPUTE_E) */



#endif /* REALTYPE_H */