#ifndef NATIVE_OPS_H
#define NATIVE_OPS_H
/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 * GitHub: https://github.com/Embed-ML/EmbedIA
 */

/**
 * @file native_ops.h
 * @brief Generic arithmetic operations abstraction layer for EmbedIA.
 *
 * Provides type-agnostic macros for operations on real_t and realx_t.
 * Implementation is backed by the type selected in datatypes.h (currently fixed 16-bit).
 * Do not add #ifdefs here—EmbedIA generates/copies the correct version.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ========================================================================
 * BASIC ARITHMETIC OPERATIONS (real_t → real_t)
 * ======================================================================== */

/**
 * @defgroup basic_ops Basic Arithmetic Operations
 * @brief Operations on standard precision type (real_t)
 * @{
 */

/** @brief Addition: a + b */
#define ADD(a, b)       FIXED_ADD(a, b)

/** @brief Subtraction: a - b */
#define SUB(a, b)       FIXED_SUB(a, b)

/** @brief Multiplication: a * b (with proper scaling) */
#define MUL(a, b)       FIXED_MUL(a, b)

/** @brief Division: a / b (with proper scaling) */
#define DIV(a, b)       FIXED_DIV(a, b)

/** @brief Negation: -a */
#define NEG(a)          FIXED_NEG(a)

/** @} */


/* ========================================================================
 * EXTENDED PRECISION OPERATIONS (real_t → realx_t, realx_t → realx_t)
 * ======================================================================== */

/**
 * @defgroup extended_ops Extended Precision Operations
 * @brief Operations for multiply-accumulate without overflow
 * @{
 */

/**
 * @brief Extended multiplication: a * b → realx_t
 * @note Result is NOT scaled down (double precision)
 * Use for accumulation: acc += XMUL(a, b)
 */
#define XMUL(a, b)      DFIXED_MUL(a, b)

/** @brief Extended addition: a + b (both realx_t) */
#define XADD(a, b)      DFIXED_ADD(a, b)

/** @brief Extended subtraction: a - b (both realx_t) */
#define XSUB(a, b)      DFIXED_SUB(a, b)

/** @brief Extended division: a / b → realx_t */
#define XDIV(a, b)      DFIXED_DIV(a, b)

/** @} */


/* ========================================================================
 * PRECISION CONVERSION
 * ======================================================================== */

/**
 * @defgroup conversion Precision Conversion
 * @brief Convert between standard and extended precision
 * @{
 */

/** @brief Promote real_t to realx_t (increase precision) */
#define PROMOTE(a)      FIXED_TO_DFIXED(a)

/** @brief Demote realx_t to real_t (reduce precision) */
#define DEMOTE(a)       DFIXED_TO_FIXED(a)

/** @} */


/* ========================================================================
 * TYPE CONVERSIONS
 * ======================================================================== */

/**
 * @defgroup type_conv Type Conversions
 * @brief Convert between real_t and C native types
 * @{
 */

/** @brief Float to real_t */
#define FL2R(f)         FLOAT_TO_FIXED(f)

/** @brief Real_t to float */
#define R2FL(r)         FIXED_TO_FLOAT(r)

/** @brief Integer to real_t */
#define INT2R(i)        INT_TO_FIXED(i)

/** @brief Real_t to integer (truncate) */
#define R2INT(r)        FIXED_TO_INT(r)

/** @brief Real_t to double */
#define R2DB(r)         FIXED_TO_DOUBLE(r)

/** @} */


/* ========================================================================
 * UTILITY OPERATIONS
 * ======================================================================== */

/**
 * @defgroup utils Utility Operations
 * @brief Common utility functions
 * @{
 */

/** @brief Absolute value */
#define ABS(a)          FIXED_ABS(a)

/** @brief Minimum of two values */
#define MIN(a, b)       FIXED_MIN(a, b)

/** @brief Maximum of two values */
#define MAX(a, b)       FIXED_MAX(a, b)

/** @brief Round up to next integer */
#define CEIL(a)         FIXED_CEIL(a)

/** @brief Round down to integer */
#define FLOOR(a)        FIXED_FLOOR(a)

/** @brief Round to nearest integer */
#define ROUND(a)        FIXED_ROUND(a)

/** @brief Get fractional part */
#define FRAC(a)         FIXED_FRAC(a)

/** @brief Get integer part (as real_t with .0) */
#define INTPART(a)      FIXED_INT(a)

/** @} */

/* ========================================================================
 * SATURATION AND CLAMPING
 * ======================================================================== */

/**
 * @defgroup saturation Saturation and Clamping
 * @brief Macros for value clamping and saturated demotion
 * @{
 */

#define EXP_SAFE_MIN  FL2R(-10.0f)  // Rango conservador para EXP en fixed16
#define EXP_SAFE_MAX  FL2R(9.0f)    // e^9 ~8103, dentro de realx_t

/** @brief Clamp value between min and max */
#define CLAMP(val, min, max) MAX(min, MIN(val, max))

// Clamp previo (para entradas a ops riesgosas)
#define EXP_CLAMP(x) CLAMP(x, EXP_SAFE_MIN, EXP_SAFE_MAX)

/** @brief Demote realx_t to real_t with saturation */
#define SAT_DEMOTE(x) (((x) > REALX_MAX) ? REAL_MAX : (((x) < REALX_MIN) ? REAL_MIN : DEMOTE(x)))

/** @} */

/* ========================================================================
 * MATHEMATICAL FUNCTIONS
 * ======================================================================== */

/**
 * @defgroup math_funcs Mathematical Functions
 * @brief Transcendental and special functions
 * @{
 */

/** @brief Square root: √a */
#define SQRT(a)         fixed_sqrt(a)

/** @brief Exponential: e^a */
#define EXP(a)          fixed_exp(a)

/** @brief Natural logarithm: ln(a) */
#define LOG(a)          fixed_log(a)

/** @brief Logarithm base 10: log₁₀(a) */
#define LOG10(a)        FIXED_MUL(fixed_log(a), FL2FX(0.43429448190325182765f))

/** @brief Power: a^b */
#define POW(a, b)       fixed_pow(a, b)

/** @brief Sine: sin(a) [radians] */
#define SIN(a)          fixed_sin(a)

/** @brief Cosine: cos(a) [radians] */
#define COS(a)          fixed_cos(a)

/** @brief Hyperbolic tangent: tanh(a) */
#define TANH(a)         fixed_tanh(a)

/** @brief Magnitude: √(a² + b²) approximation */
#define MAG(a, b)       fixed_magnitude(a, b)

/** @} */




#ifdef __cplusplus
}
#endif

#endif /* NATIVE_OPS_H */