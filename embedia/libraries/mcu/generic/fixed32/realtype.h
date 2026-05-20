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
 * @brief Data type definitions and macro operations for FIXED32 type
 * 
 * Implements three-tier type abstraction:
 * - storage_t: Model parameter storage (weights, kernels)
 * - compute_t: Runtime calculations (activations, intermediate results)
 * - computex_t: Extended precision accumulators (prevent overflow)
 * 
 * Legacy types real_t/realx_t maintained for backward compatibility.
 * 
 * NOTE: FIXED32 serves as the base for 32-bit fixed-point arithmetic.
 */

#ifndef REALTYPE_H
#define REALTYPE_H


/* ========================================================================
 * METADATA
 * ======================================================================== */

#define DATA_TYPE_NAME  "FIXED32"     /**< Human-readable type name */


/* ========================================================================
 * FIXED32 CONFIGURATION
 * ======================================================================== */

/* Include the specific fixed-point implementation */
#include "fixed.h"


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
 * Current configuration: FIXED32
 * ======================================================================== */

/**
 * @typedef storage_t
 * @brief Model parameter storage type (weights, kernels)
 * 
 * Purpose: Minimize memory/flash footprint
 * Usage: Static model data, weight arrays
 * For FIXED32: 32-bit fixed-point (Q15.16 or configurable)
 */
typedef fixed storage_t;

/**
 * @typedef compute_t
 * @brief Runtime computation type (activations, intermediate results)
 * 
 * Purpose: Arithmetic operations with sufficient precision
 * Usage: Layer inputs/outputs, bias values, activations
 * For FIXED32: 32-bit fixed-point (Q15.16 or configurable)
 */
typedef fixed compute_t;

/**
 * @typedef computex_t
 * @brief Extended precision accumulator type
 * 
 * Purpose: Prevent overflow in multiply-accumulate operations
 * Usage: Dot products, convolution accumulators
 * For FIXED32: 64-bit fixed-point (Q31.32 or configurable)
 */
typedef dfixed computex_t;

/* ========================================================================
 * TYPE CONVERSIONS
 * ======================================================================== */

// Storage ↔ Compute (no conversion needed, same type)
#define ST2CO(s, qp)     (s)                       /**< Load weight: fixed32 → fixed32 (no-op) */
#define CO2ST(c, qp)     (c)                       /**< Store result: fixed32 → fixed32 (no-op) */

// Compute ↔ ComputeX (precision promotion/demotion)
#define CO2CX(c)         FX2DFX(c)                 /**< Promote: fixed32 → fixed64 */
#define CX2CO(cx)        DFX2FX(cx)                /**< Demote: fixed64 → fixed32 */
#define CX2CO_SAT(cx)    DFX2FX_SAT(cx)           /**< Demote with saturation */

/* ========================================================================
 * BACKWARD COMPATIBILITY
 * ======================================================================== */

typedef compute_t real_t;      /**< Legacy alias (deprecated, use compute_t) */
typedef computex_t realx_t;    /**< Legacy alias (deprecated, use computex_t) */

#define R2RX(val)        CO2CX(val)      /**< Legacy (use CO2CX) */
#define RX2R(val)        CX2CO(val)      /**< Legacy (use CX2CO) */
#define RX2R_SAT(val)    CX2CO_SAT(val)  /**< Legacy (use CX2CO_SAT) */

/* ========================================================================
 * NUMERIC LIMITS
 * ======================================================================== */

// Storage limits
#define STORAGE_MAX      FIX_MAX       /**< Maximum storage value (fixed32) */
#define STORAGE_MIN      FIX_MIN       /**< Minimum storage value (fixed32) */

// Compute limits
#define COMPUTE_MAX      FIX_MAX       /**< Maximum compute value (fixed32) */
#define COMPUTE_MIN      FIX_MIN       /**< Minimum compute value (fixed32) */
#define COMPUTEX_MAX     DFIX_MAX      /**< Maximum computex value (fixed64) */
#define COMPUTEX_MIN     DFIX_MIN      /**< Minimum computex value (fixed64) */

// Legacy aliases
#define REAL_MAX         COMPUTE_MAX   /**< Legacy (use COMPUTE_MAX) */
#define REAL_MIN         COMPUTE_MIN   /**< Legacy (use COMPUTE_MIN) */
#define REALX_MAX        COMPUTEX_MAX  /**< Legacy (use COMPUTEX_MAX) */
#define REALX_MIN        COMPUTEX_MIN  /**< Legacy (use COMPUTEX_MIN) */


/* ========================================================================
 * FUNDAMENTAL CONSTANTS
 * ======================================================================== */

// Storage constants
#define STORAGE_ZERO     FIX_ZERO      /**< Zero value (fixed32) */

// Compute constants
#define COMPUTE_ZERO     FIX_ZERO      /**< Zero value: 0 */
#define COMPUTE_ONE      FIX_ONE       /**< Unity value (represents 1.0) */
#define COMPUTE_TWO      FIX_TWO       /**< Two value (represents 2.0) */
#define COMPUTE_HALF     FIX_HALF      /**< Half value (represents 0.5) */
#define COMPUTE_INV_2    FIX_INV_2     /**< 1/2! = 0.5 */

// ComputeX constants
#define COMPUTEX_ZERO    FX2DFX(FIX_ZERO)  /**< Zero value: 0.0 */
#define COMPUTEX_ONE     FX2DFX(FIX_ONE)   /**< Unity value: 1.0 */
#define COMPUTEX_TWO     FX2DFX(FIX_TWO)   /**< Two value: 2.0 */
#define COMPUTEX_HALF    FX2DFX(FIX_HALF)  /**< Half value: 0.5 */
#define COMPUTEX_INV_2   FX2DFX(FIX_INV_2) /**< 1/2! = 0.5 */

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
#define COMPUTE_PI       FIX_PI        /**< π ≈ 3.14159 */
#define COMPUTE_2PI      FIX_2PI       /**< 2π ≈ 6.28318 */
#define COMPUTE_PI_2     FIX_PI_INV2   /**< π/2 ≈ 1.57079 */
#define COMPUTE_PI_4     FIX_PI_INV4   /**< π/4 ≈ 0.78539 */
#define COMPUTE_E        FIX_E         /**< e (Euler's number) ≈ 2.71828 */

// Legacy aliases
#define REAL_PI          COMPUTE_PI    /**< Legacy (use COMPUTE_PI) */
#define REAL_2PI         COMPUTE_2PI   /**< Legacy (use COMPUTE_2PI) */
#define REAL_PI_2        COMPUTE_PI_2  /**< Legacy (use COMPUTE_PI_2) */
#define REAL_PI_4        COMPUTE_PI_4  /**< Legacy (use COMPUTE_PI_4) */
#define REAL_E           COMPUTE_E     /**< Legacy (use COMPUTE_E) */


/* ========================================================================
 * OPERATION ALIASES
 * ======================================================================== */

/** @defgroup operation_aliases Short aliases for fixed-point operations */
/** @{ */

/* Type Conversions */
#define FL2R(F)      FL2FX(F)           /**< float to fixed */
#define R2FL(F)      FX2FL(F)           /**< fixed to float */
#define INT2R(I)     INT2FX(I)          /**< int to fixed */
#define R2INT(F)     FIXED_TO_INT(F)    /**< fixed to int */

/* DFixed Conversions (RX prefix for DFIXED operations) */
#define FL2RX(F)     FL2DFX(F)          /**< float to dfixed */
#define RX2FL(F)     DFX2FL(F)          /**< dfixed to float */
#define INT2RX(I)    INT2DFX(I)         /**< int to dfixed */
#define RX2INT(F)    dfixed_to_int(F)   /**< dfixed to int */

/* Fixed Arithmetic */
#define RADD(A,B)    FIXED_ADD(A,B)     /**< fixed addition */
#define RSUB(A,B)    FIXED_SUB(A,B)     /**< fixed subtraction */
#define RMUL(A,B)    FIXED_MUL(A,B)     /**< fixed multiplication */
#define RDIV(A,B)    FIXED_DIV(A,B)     /**< fixed division */
#define RDIV_INT(A,B) FIXED_DIV_INT(A,B) /**< fixed division by integer */
#define RNEG(A)      FIXED_NEG(A)       /**< fixed negation */

/* DFixed Arithmetic (RX prefix for DFIXED operations) */
#define RXADD(A,B)   DFIXED_ADD(A,B)    /**< dfixed addition */
#define RXSUB(A,B)   DFIXED_SUB(A,B)    /**< dfixed subtraction */
#define RXMUL(A,B)   DFIXED_MUL(A,B)    /**< dfixed multiplication */
#define RXDIV(A,B)   DFIXED_DDIV(A,B)   /**< dfixed division */
#define RXDIV_INT(A,B) DFIXED_DDIV_INT(A,B) /**< dfixed division by integer */

/* Fixed Math Functions */
#define RABS(A)      FIXED_ABS(A)       /**< fixed absolute value */
#define RMIN(A,B)    FIXED_MIN(A,B)     /**< fixed minimum */
#define RMAX(A,B)    FIXED_MAX(A,B)     /**< fixed maximum */
#define RCEIL(A)     FIXED_CEIL(A)      /**< fixed ceiling */
#define RFLOOR(A)    FIXED_FLOOR(A)     /**< fixed floor */
#define RROUND(A)    FIXED_ROUND(A)     /**< fixed round */
#define RSQRT(A)     fixed_sqrt(A)      /**< fixed square root */
#define REXP(A)      fixed_exp(A)       /**< fixed exponential */
#define RLOG(A)      fixed_log(A)       /**< fixed logarithm */
#define RPOW(A,B)    fixed_pow(A,B)     /**< fixed power */
#define RSIN(A)      fixed_sin(A)       /**< fixed sine */
#define RCOS(A)      fixed_cos(A)       /**< fixed cosine */
#define RTANH(A)     fixed_tanh(A)      /**< fixed hyperbolic tangent */

/* DFixed Math Functions (RX prefix for DFIXED operations) */
#define RXABS(A)     DFIXED_ABS(A)      /**< dfixed absolute value */
#define RXMIN(A,B)   DFIXED_MIN(A,B)    /**< dfixed minimum */
#define RXMAX(A,B)   DFIXED_MAX(A,B)    /**< dfixed maximum */

/* Utilities */
#define RCLAMP(X,MIN,MAX)    CLAMP(X,MIN,MAX)    /**< clamp value */
#define RXSATURATE(X)        SATURATE(X)         /**< saturate dfixed to fixed */
#define RFRAC(F)             FIXED_FRAC(F)       /**< get fractional part */
#define RINT_PART(F)         FIXED_INT(F)        /**< get integer part */

/** @} */



#endif /* REALTYPE_H */