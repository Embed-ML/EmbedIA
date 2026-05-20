#ifndef FIXED_H
#define FIXED_H
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
 * @file fixed.h
 * @brief 32-bit fixed-point arithmetic library.
 *
 * This file defines types, constants, macros, and functions for performing
 * fixed-point arithmetic operations with controlled precision.
 * Uses 32 bits total, with configurable integer and fractional bit sizes.
 */

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef fixed
 * @brief 32-bit integer type representing a fixed-point number.
 */
typedef int32_t fixed;

/**
 * @typedef dfixed
 * @brief 32-bit integer type for extended fixed-point operations with 4 additional range bits.
 */
typedef int32_t dfixed;

/// @cond INTERNAL
// Total size of fixed type in bits
#define FIX_SIZE 32
// Number of fractional bits
#define FIX_FRC_SZ 17
// Number of integer bits
#define FIX_INT_SZ (FIX_SIZE - FIX_FRC_SZ)
// Mask for fractional part
#define FIX_FRC_MSK  (((fixed)1 << FIX_FRC_SZ) - 1)

// Fractional size for dfixed (extended precision with 4 bits more range)
#define DFIX_DIFF_BITS 2
#define FIX_DFRC_SZ (FIX_FRC_SZ - DFIX_DIFF_BITS)
// Integer size for dfixed
#define FIX_DINT_SZ (FIX_INT_SZ + DFIX_DIFF_BITS)
// Mask for dfixed fractional part
#define FIX_DFRC_MSK  (((dfixed)1 << FIX_DFRC_SZ) - 1)
/// @endcond

//////////////////////////////////// Constants ////////////////////////////////////

/** @defgroup constants Mathematical constants and fixed values */
/** @{ */

/** @brief 0.5 in fixed-point */
#define FIX_HALF (FIX_ONE >> 1)

/** @brief Zero value in fixed-point */
#define FIX_ZERO 0

/** @brief Value 1.0 in fixed-point */
#define FIX_ONE ((fixed)((fixed)1 << FIX_FRC_SZ))

/** @brief Value 2.0 in fixed-point */
#define FIX_TWO (FIX_ONE + FIX_ONE)

/** @brief Value of π in floating-point (for conversion) */
#define M_PI 3.14159265358979323846

/** @brief Safe upper limit for fixed_exp (approx 9.71) */
#define FIX_EXP_MAX FL2FX(9.7132951013616)

/** @brief Value of e (natural logarithm base) in fixed-point */
#define FIX_E  FLOAT_TO_FIXED(2.7182818284590452354)

/** @brief Value of π in fixed-point */
#define FIX_PI FLOAT_TO_FIXED(M_PI)

/** @brief Value of 2π in fixed-point */
#define FIX_2PI FL2FX(2*M_PI)   // 2π

/** @brief Value of 3π/2 in fixed-point */
#define FIX_3PI_INV2 FL2FX(3*M_PI/2) // 3π/2

/** @brief Value of 1/π in fixed-point */
#define FIX_INV_PI FL2FX(1/M_PI) // 1/π

/** @brief Value of 2/(2π) = 1/π in fixed-point (possible naming error in original) */
#define FIX_INV_2PI FL2FX(2/(2*FIX_PI)) // 2/(2π)

/** @brief Value of π/2 in fixed-point */
#define FIX_PI_INV2 FL2FX(M_PI/2)  // π/2

/** @brief Value of π/3 in fixed-point */
#define FIX_PI_INV3 FL2FX(M_PI/3) // π/3

/** @brief Value of π/4 in fixed-point */
#define FIX_PI_INV4 FL2FX(M_PI/4) // π/4

/** @brief Value of π/6 in fixed-point */
#define FIX_PI_INV6 FL2FX(M_PI/6) // π/6

/** @} */

//////////////////////////////////// Conversion Macros ////////////////////////////////////

/** @defgroup conversion_macros Macros for type conversions */
/** @{ */

/**
 * @brief Converts a fixed-point value to double.
 * @param F Fixed-point value.
 * @return Value converted to double.
 */
#define FIXED_TO_DOUBLE(F) ((double) ((F)*((double)(1)/(double)(1L << FIX_FRC_SZ))))

/**
 * @brief Converts a fixed-point value to float.
 * @param F Fixed-point value.
 * @return Value converted to float.
 */
#define FIXED_TO_FLOAT(F) ((float) ((F)*((float)(1)/(float)(1L << FIX_FRC_SZ))))

/**
 * @brief Extracts the integer part of a fixed-point value.
 * @param F Fixed-point value.
 * @return Integer part as integer.
 */
#define FIXED_TO_INT(F) ((fixed)(F) >> FIX_FRC_SZ)

/**
 * @brief Extracts the fractional part of a fixed-point value.
 * @param F Fixed-point value.
 * @return Fractional part in fixed-point format.
 */
#define FIXED_FRAC(F) ( (fixed)(F) & FIX_FRC_MSK )

/**
 * @brief Gets the integer part in fixed-point format (with .0).
 * @param F Fixed-point value.
 * @return Integer part with zero fraction.
 */
#define FIXED_INT(F) ( (fixed)(F) & ~FIX_FRC_MSK )

/**
 * @brief Converts a float value to fixed-point with rounding.
 * @param F Floating-point value.
 * @return Value converted to fixed-point.
 */
#define FLOAT_TO_FIXED(F) ((fixed)((F) * FIX_ONE + ((F) >= 0 ? 0.5 : -0.5)))

/**
 * @brief Converts an integer to fixed-point.
 * @param I Integer value.
 * @return Value in fixed-point.
 */
#define INT_TO_FIXED(I) ((fixed)(I) << FIX_FRC_SZ)

// aliases
#define INT2FX(I) INT_TO_FIXED(I)

/**
 * @brief Shortcut for FLOAT_TO_FIXED.
 * @param F Floating-point value.
 * @return Value in fixed-point.
 */
#define FL2FX(F) FLOAT_TO_FIXED(F)

/**
 * @brief Shortcut for FIXED_TO_FLOAT.
 * @param F Fixed-point value.
 * @return Floating-point value.
 */
#define FX2FL(F) FIXED_TO_FLOAT(F)

/** @} */

// Limits
/** @defgroup limits Limits of the fixed type */
/** @{ */

/** @brief Maximum representable value in fixed type */
#define FIX_MAX ((fixed)(((uint32_t)1 << (FIX_SIZE - 1)) - 1))

/** @brief Minimum representable value in fixed type */
#define FIX_MIN ((fixed)(-((int32_t)FIX_MAX) - 1))

/** @brief Maximum value in dfixed (extended precision) */
#define DFIX_MAX FX2DFX(FIX_MAX)

/** @brief Minimum value in dfixed */
#define DFIX_MIN (-DFIX_MAX-1)

/** @brief Value 1.0 in dfixed format */
#define DFIX_ONE ((dfixed)((dfixed)1 << FIX_DFRC_SZ))

/** @brief Value 0.5 in dfixed format */
#define DFIX_HALF (DFIX_ONE >> 1)

/** @} */

// Factorial inverses (for series)
/** @defgroup factorial_inv Factorial inverses (used in Taylor series) */
/** @{ */

/** @brief 1/2! = 0.5 */
#define FIX_INV_2    FLOAT_TO_FIXED(0.5)

/** @brief 1/3! = 1/6 ≈ 0.1667 */
#define FIX_INV_6    FLOAT_TO_FIXED(0.16666666666666666)

/** @brief 1/4! = 1/24 ≈ 0.04167 */
#define FIX_INV_24   FLOAT_TO_FIXED(0.041666666666666664)

/** @brief 1/5! = 1/120 ≈ 0.008333 */
#define FIX_INV_120  FLOAT_TO_FIXED(0.008333333333333333)

/** @brief 1/6! = 1/720 ≈ 0.0013889 */
#define FIX_INV_720  FLOAT_TO_FIXED(0.001388888888888889)

/** @brief 1/7! = 1/5040 ≈ 0.0001984 */
#define FIX_INV_5040 FLOAT_TO_FIXED(0.00019841269841269841)

/** @} */

//////////////////////////////////// Arithmetic Macros ////////////////////////////////////

/** @defgroup arithmetic_macros Basic arithmetic macros */
/** @{ */

/** @brief Adds two fixed-point values. */
#define FIXED_ADD(A,B) ((A) + (B))

/** @brief Subtracts two fixed-point values. */
#define FIXED_SUB(A,B) ((A) - (B))

/** @brief Negation (sign change) in fixed-point. */
#define FIXED_NEG(A) (-(A))

/**
 * @brief Multiplies two fixed-point values with rounding.
 * @param A First operand.
 * @param B Second operand.
 * @return Result in fixed-point.
 */
#define FIXED_MUL(A,B) \
    ((fixed)((((dfixed)(A) * (dfixed)(B)) + (1 << (FIX_FRC_SZ - 1))) >> FIX_FRC_SZ))

/**
 * @brief Fixed-point division.
 * @param A Dividend.
 * @param B Divisor.
 * @return Quotient in fixed-point.
 * @note Uses multiplication by reciprocal for efficiency on 32-bit
 */
#define FIXED_DIV(A,B) \
    ((fixed)((((int32_t)(A) << FIX_FRC_SZ) + ((int32_t)(B) >> 1)) / (int32_t)(B)))

/** @} */

//////////////////////////////////// Double Fixed Macros ////////////////////////////////////

/** @defgroup dfixed_macros Double precision operations (dfixed) */
/** @{ */

/** @brief Converts fixed to dfixed (increases range). */
#define FIXED_TO_DFIXED(A)      \
    ((dfixed)(A) >> DFIX_DIFF_BITS)

/** @brief Converts fixed to dfixed (increases precision). */
#define FX2DFX(A) FIXED_TO_DFIXED(A)

/** @brief Converts dfixed to fixed (reduces range). */
#define DFIXED_TO_FIXED(A)      \
    ((dfixed)(A) << DFIX_DIFF_BITS)

/** @brief Converts dfixed to fixed (reduces precision). */
#define DXF2FX(A) DFIXED_TO_FIXED(A)


/** @brief Converts integer to dfixed. */
#define INT_TO_DFIXED(A) \
    ((dfixed)(A) << FIX_DFRC_SZ)

// aliases
#define INT2DFX(A) INT_TO_DFIXED(A)

/**
 * @brief Multiplies two dfixed values with rounding.
 * @warning Both operands and result are 32-bit - risk of overflow
 */
#define DFIXED_MUL(A,B)            \
    ((dfixed)((((int32_t)(A) * (int32_t)(B)) + (1 << (FIX_DFRC_SZ - 1))) >> FIX_DFRC_SZ))

/**
 * @brief Division between fixed values with result in dfixed.
 * @param A fixed dividend.
 * @param B fixed divisor.
 * @return Quotient in dfixed.
 */
#define DFIXED_DIV(A,B) \
    ((dfixed)((((int32_t)FIXED_TO_DFIXED(A) << FIX_DFRC_SZ) + (FIXED_TO_DFIXED(B) >> 1)) / FIXED_TO_DFIXED(B)))

/**
 * @brief Division between two dfixed values, result in dfixed.
 * @param A dfixed dividend.
 * @param B dfixed divisor.
 * @return Quotient in dfixed.
 */
#define DFIXED_DDIV(A,B) \
    ((dfixed)((((int32_t)(A) << FIX_DFRC_SZ) + ((int32_t)(B) >> 1)) / (int32_t)(B)))

/**
 * @brief Division of fixed by integer (optimized).
 * @param A fixed dividend
 * @param B int32_t divisor
 * @return Quotient in fixed
 *
 * More efficient than FIXED_DIV when divisor is a pure integer.
 *
 * Performance comparison (Cortex-M4):
 * - FIXED_DIV_BY_INT: ~10-20 cycles
 * - FIXED_DIV: ~20-30 cycles
 */
#define FIXED_DIV_INT(A, B) \
    ((fixed)((((int32_t)(A) << FIX_FRC_SZ) + ((int32_t)(B) >> 1)) / (int32_t)(B)))

/**
 * @brief Division of dfixed by integer (optimized).
 * @param A dfixed dividend
 * @param B int32_t divisor
 * @return Quotient in dfixed
 *
 * Most efficient division for dfixed.
 *
 * Performance comparison (Cortex-M4):
 * - DFIXED_DDIV_INT: ~2-12 cycles (64÷32)
 * - DFIXED_DDIV: ~120-150 cycles (64÷64)
 *
 * 10-50× FASTER than DFIXED_DDIV!
 */
#define DFIXED_DDIV_INT(A, B) \
    ((dfixed)(((dfixed)(A) + ((int32_t)(B) >> 1)) / (int32_t)(B)))


/** @brief Extended addition of two dfixed values.
 * @param A dfixed operand.
 * @param B dfixed operand.
 * @return result in dfixed.
 */
#define DFIXED_ADD(A,B) ((dfixed)(A) + (dfixed)(B))

/** @brief Extended subtraction of two dfixed values.
 * @param A dfixed operand.
 * @param B dfixed operand.
 * @return result in dfixed.
 */
#define DFIXED_SUB(A,B) ((dfixed)(A) - (dfixed)(B))


/** @brief Scale used in dfixed for floating-point conversion. */
#define DFIXED_SCALE ((dfixed)(1 << FIX_DFRC_SZ))

/** @brief Converts dfixed to float. */
#define DFX2FL(x) (((float)(x)) / (float)DFIXED_SCALE)

/** @brief Converts float to dfixed. */
#define FL2DFX(x) ((dfixed)((x) * (float)DFIXED_SCALE + ((x) >= 0 ? 0.5 : -0.5)))

/** @brief Converts dfixed to double. */
#define DFX2DB(x) (((double)(x)) / (double)DFIXED_SCALE)

/** @brief Converts double to dfixed. */
#define DB2DFX(x) ((dfixed)((x) * (double)DFIXED_SCALE + ((x) >= 0 ? 0.5 : -0.5)))

/** @brief clamping macro for fixed-point values. */
#define CLAMP(X,MIN,MAX) ((X)<(MIN)?(MIN):((X)>(MAX)?(MAX):(X)))

/** @brief Clamping macro for dfixed to fixed conversion. */
#define CLAMP_DFX_TO_FX(X, MIN, MAX) ((X<FIXED_TO_DFIXED(MIN))?MIN:((X>FIXED_TO_DFIXED(MAX))?MAX:DFIXED_TO_FIXED(X)))

/** @brief Saturation macro for dfixed to fixed conversion. */
#define SATURATE(X) CLAMP_DFX_TO_FX(X, FIX_MIN, FIX_MAX)

/** @brief Absolute value for dfixed. */
#define DFIXED_ABS(A) ((A) < 0 ? -(A) : (A))

/** @brief Minimum between two dfixed values. */
#define DFIXED_MIN(a,b) ((a) < (b) ? (a) : (b))

/** @brief Maximum between two dfixed values. */
#define DFIXED_MAX(a,b) ((a) > (b) ? (a) : (b))

/** @} */

//////////////////////////////////// Additional Macros ////////////////////////////////////

/** @defgroup extra_macros Additional useful macros */
/** @{ */

/** @brief Absolute value in fixed-point. */
#define FIXED_ABS(A) ((A) < 0 ? -(A) : (A))

/** @brief Round up to next integer in fixed-point. */
#define FIXED_CEIL(A) ( FIXED_INT(A) +  (FIXED_FRAC(A) ? FIX_ONE : 0) )

/** @brief Truncate down to integer in fixed-point. */
#define FIXED_FLOOR(A) ( FIXED_INT(A) )

/** @brief Round to nearest integer in fixed-point. */
#define FIXED_ROUND(A) (FIXED_INT(A) + ((FIXED_FRAC(A) >= FIX_HALF) ? FIX_ONE : 0))

/** @brief Minimum between two fixed-point values. */
#define FIXED_MIN(a,b) ((a) < (b) ? (a) : (b))

/** @brief Maximum between two fixed-point values. */
#define FIXED_MAX(a,b) ((a) > (b) ? (a) : (b))

/** @} */

//////////////////////////////////// Type Conversion Functions ////////////////////////////////////

/** @addtogroup conversion_functions
 * @{
 */

/**
 * @brief Converts a float number to fixed-point.
 * @param f Floating-point value.
 * @return Fixed-point value.
 */
fixed float_to_fixed(float f);

/**
 * @brief Converts an integer to fixed-point.
 * @param i Integer value.
 * @return Fixed-point value.
 */
fixed int_to_fixed(int32_t i);

/**
 * @brief Converts a fixed-point number to double.
 * @param f Fixed-point value.
 * @return Double-precision value.
 */
double fixed_to_double(fixed f);

/**
 * @brief Converts a fixed-point number to float.
 * @param f Fixed-point value.
 * @return Floating-point value.
 */
float fixed_to_float(fixed f);

/**
 * @brief Converts a fixed-point number to integer (truncating).
 * @param f Fixed-point value.
 * @return Integer part as integer.
 */
int32_t fixed_to_int(fixed f);

/** @} */

//////////////////////////////////// DFIXED Type Conversion Functions ////////////////////////////////////

/** @addtogroup dfixed_conversion_functions
 * @{
 */

/**
 * @brief Converts a float number to dfixed.
 * @param f Floating-point value.
 * @return dfixed value.
 */
dfixed float_to_dfixed(float f);

/**
 * @brief Converts an integer to dfixed.
 * @param i Integer value.
 * @return dfixed value.
 */
dfixed int_to_dfixed(int32_t i);

/**
 * @brief Converts a dfixed number to double.
 * @param f dfixed value.
 * @return Double-precision value.
 */
double dfixed_to_double(dfixed f);

/**
 * @brief Converts a dfixed number to float.
 * @param f dfixed value.
 * @return Floating-point value.
 */
float dfixed_to_float(dfixed f);

/**
 * @brief Converts a dfixed number to integer (truncating).
 * @param f dfixed value.
 * @return Integer part as integer.
 */
int32_t dfixed_to_int(dfixed f);

/** @} */

//////////////////////////////////// Arithmetic Functions ////////////////////////////////////

/** @addtogroup arithmetic_functions
 * @{
 */

/**
 * @brief Adds two fixed-point values.
 * @param a First operand.
 * @param b Second operand.
 * @return a + b in fixed-point.
 */
fixed fixed_add(fixed a, fixed b);

/**
 * @brief Subtracts two fixed-point values.
 * @param a Minuend.
 * @param b Subtrahend.
 * @return a - b in fixed-point.
 */
fixed fixed_sub(fixed a, fixed b);

/**
 * @brief Multiplies two fixed-point values.
 * @param a First operand.
 * @param b Second operand.
 * @return a * b in fixed-point.
 */
fixed fixed_mul(fixed a, fixed b);

/**
 * @brief Divides two fixed-point values.
 * @param a Dividend.
 * @param b Divisor.
 * @return a / b in fixed-point.
 */
fixed fixed_div(fixed a, fixed b);

/** @} */

//////////////////////////////////// DFIXED Arithmetic Functions ////////////////////////////////////

/** @addtogroup dfixed_arithmetic_functions
 * @{
 */

/**
 * @brief Adds two dfixed values.
 * @param a First operand.
 * @param b Second operand.
 * @return a + b in dfixed.
 */
dfixed dfixed_add(dfixed a, dfixed b);

/**
 * @brief Subtracts two dfixed values.
 * @param a Minuend.
 * @param b Subtrahend.
 * @return a - b in dfixed.
 */
dfixed dfixed_sub(dfixed a, dfixed b);

/**
 * @brief Multiplies two dfixed values.
 * @param a First operand.
 * @param b Second operand.
 * @return a * b in dfixed.
 */
dfixed dfixed_mul(dfixed a, dfixed b);

/**
 * @brief Divides two dfixed values.
 * @param a Dividend.
 * @param b Divisor.
 * @return a / b in dfixed.
 */
dfixed dfixed_div(dfixed a, dfixed b);

/**
 * @brief Absolute value of a dfixed number.
 * @param a dfixed value.
 * @return |a| in dfixed.
 */
dfixed dfixed_abs(dfixed a);

/** @} */

//////////////////////////////////// Special Functions ////////////////////////////////////

/** @addtogroup special_functions
 * @{
 */

/**
 * @brief Calculates the square root of a fixed-point number.
 * @param a Non-negative fixed-point value.
 * @return sqrt(a) or -1 if a < 0.
 */
fixed fixed_sqrt(fixed a);

/**
 * @brief Calculates the exponential: exp(a).
 * @param a Fixed-point value.
 * @return e^a (limited to safe range).
 */
fixed fixed_exp(fixed a);

/**
 * @brief Calculates x * 2^exp (equivalent to binary scaling).
 * @param x Base value.
 * @param exp Integer exponent.
 * @return Result in fixed-point.
 */
fixed fixed_ldexp(fixed x, int exp);

/**
 * @brief Calculates the natural logarithm of x.
 * @param x Positive fixed-point value.
 * @return ln(x).
 */
fixed fixed_log(fixed x);

/**
 * @brief Calculates the logarithm of x in base b.
 * @param x Positive value.
 * @param b Logarithm base (positive and ≠ 1).
 * @return log_b(x).
 */
fixed fixed_logn(fixed x, fixed b);

/**
 * @brief Calculates n raised to exp in fixed-point.
 * @param n Base.
 * @param exp Exponent.
 * @return n^exp.
 */
fixed fixed_pow(fixed n, fixed exp);

/**
 * @brief Approximates sqrt(a² + b²) without overflow.
 * @param a First value.
 * @param b Second value.
 * @return Magnitude approximation.
 */
fixed fixed_magnitude(fixed a, fixed b);

/** @} */

/////////////////////////////////// Trigonometric Functions ///////////////////////////////////

/** @addtogroup trigonometric_functions
 * @{
 */

/**
 * @brief Calculates the hyperbolic tangent: tanh(x).
 * @param x Fixed-point value.
 * @return tanh(x).
 */
fixed fixed_tanh(fixed x);

/**
 * @brief Calculates the sine of x (x in radians).
 * @param x Angle in radians (fixed-point).
 * @return sin(x).
 */
fixed fixed_sin(fixed x);

/**
 * @brief Calculates the cosine of x (x in radians).
 * @param x Angle in radians (fixed-point).
 * @return cos(x).
 */
fixed fixed_cos(fixed x);

/** @} */

//////////////////////////////////// Additional Functions ////////////////////////////////////

/** @addtogroup extra_functions
 * @{
 */

/**
 * @brief Absolute value of a fixed-point number.
 * @param a Fixed-point value.
 * @return |a|.
 */
fixed fixed_abs(fixed a);

/**
 * @brief Round up (ceil).
 * @param a Fixed-point value.
 * @return Smallest integer ≥ a.
 */
fixed fixed_ceil(fixed a);

/**
 * @brief Round down (floor).
 * @param a Fixed-point value.
 * @return Largest integer ≤ a.
 */
fixed fixed_floor(fixed a);

/** @} */

#ifdef __cplusplus
}
#endif

#endif // FIXED_H