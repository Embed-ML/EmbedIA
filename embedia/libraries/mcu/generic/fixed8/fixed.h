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
 * @brief 8-bit fixed-point arithmetic library.
 *
 * This file defines types, constants, macros, and functions for performing
 * fixed-point arithmetic operations with controlled precision.
 * Uses 8 bits total, with configurable integer and fractional bit sizes.
 */

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef fixed
 * @brief 8-bit integer type representing a fixed-point number.
 */
typedef int8_t fixed;

/**
 * @typedef dfixed
 * @brief 16-bit integer type for double-precision fixed-point operations.
 */
typedef int16_t dfixed;

/// @cond INTERNAL
/** @brief Total size of fixed type in bits */
#define FIX_SIZE 8
/** @brief Number of fractional bits */
#define FIX_FRC_SZ 4
/** @brief Number of integer bits */
#define FIX_INT_SZ (FIX_SIZE - FIX_FRC_SZ)
/** @brief Mask for fractional part */
#define FIX_FRC_MSK  (((fixed)1 << FIX_FRC_SZ) - 1)
/** @brief Fractional size for dfixed (double precision) */
#define FIX_DFRC_SZ (FIX_FRC_SZ*2)
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

/** @brief Safe upper limit for fixed_exp (approx 4.15) */
#define FIX_EXP_MAX FL2FX_CONST(2.77258872224)

/** @brief Value of e (natural logarithm base) in fixed-point */
#define FIX_E  FL2FX_CONST(2.7182818284590452354)

/** @brief Value of π in fixed-point */
#define FIX_PI FL2FX_CONST(M_PI)

/** @brief Value of 2π in fixed-point */
#define FIX_2PI FL2FX_CONST(2*M_PI)   // 2π

/** @brief Value of 3π/2 in fixed-point */
#define FIX_3PI_INV2 FL2FX_CONST(3*M_PI/2) // 3π/2

/** @brief Value of 1/π in fixed-point */
#define FIX_INV_PI FL2FX_CONST(1/M_PI) // 1/π

/** @brief Value of 2/(2π) = 1/π in fixed-point (possible naming error in original) */
#define FIX_INV_2PI FL2FX_CONST(2/(2*M_PI)) // 2/(2π)

/** @brief Value of π/2 in fixed-point */
#define FIX_PI_INV2 FL2FX_CONST(M_PI/2)  // π/2

/** @brief Value of π/3 in fixed-point */
#define FIX_PI_INV3 FL2FX_CONST(M_PI/3) // π/3

/** @brief Value of π/4 in fixed-point */
#define FIX_PI_INV4 FL2FX_CONST(M_PI/4) // π/4

/** @brief Value of π/6 in fixed-point */
#define FIX_PI_INV6 FL2FX_CONST(M_PI/6) // π/6

/**
 * @brief Reciprocal precision for average operations.
 * sum(dfixed/int32, Q8) * recip(Q8) = Q16, fits in int32
 * Supports pool up to 181x181 with values in [-128, 128)
 */
#define FIX_AVG_PREC  (9)

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

/** @brief Alias for INT_TO_FIXED */
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

/**
 * @brief Converts a floating-point literal to fixed-point at compile time.
 *
 * This macro is intended EXCLUSIVELY for constant expressions and table
 * initializers evaluated at compile time (e.g. lookup tables, #define
 * constants). The intermediate arithmetic uses int64_t to avoid overflow
 * for values greater than 1.0 in any Q format.
 *
 * Unlike FL2FX, this macro does NOT generate floating-point instructions
 * at runtime — the result is a fixed integer literal embedded directly
 * in .rodata or as an immediate value.
 *
 * @param F Floating-point literal or constant expression.
 *          DO NOT use with runtime variables — use FL2FX instead.
 * @return Fixed-point representation of F, scaled by 2^FIX_FRC_SZ.
 *
 * @note Requires FIX_FRC_SZ to be defined before use.
 * @note Safe for values up to (2^(32-FIX_FRC_SZ) - 1) without overflow.
 *
 * Correct usage:
 * @code
 *   static const fixed lut[3] = {
 *       FL2FX_CONST(0.0),
 *       FL2FX_CONST(0.70710678),  // sqrt(2)/2
 *       FL2FX_CONST(1.0),
 *   };
 *   #define FIX_PI  FL2FX_CONST(3.14159265358979323846)
 *   #define FIX_E   FL2FX_CONST(2.71828182845904523536)
 * @endcode
 *
 * Incorrect usage:
 * @code
 *   float val = 0.5f;
 *   fixed fx = FL2FX_CONST(val);  // WRONG — use FL2FX(val) instead
 * @endcode
 */
#define FL2FX_CONST(F) \
    ((fixed)((int64_t)((F) * ((int64_t)1 << FIX_FRC_SZ) + 0.5)))

/** @} */

/** @defgroup limits Limits of the fixed type */
/** @{ */

/** @brief Maximum representable value in fixed type */
#define FIX_MAX (fixed)(((dfixed)1 << (FIX_SIZE-1)) - 1)

/** @brief Minimum representable value in fixed type */
#define FIX_MIN (-FIX_MAX)

/** @brief Maximum value in dfixed (double precision) */
#define DFIX_MAX ((dfixed)FIX_MAX << FIX_FRC_SZ)

/** @brief Minimum value in dfixed */
#define DFIX_MIN (-DFIX_MAX)

/** @brief Value 1.0 in dfixed format */
#define DFIX_ONE ((dfixed)FIX_ONE << FIX_FRC_SZ)

/** @brief Value 0.5 in dfixed format */
#define DFIX_HALF (DFIX_ONE >> 1)

/** @} */

/** @defgroup factorial_inv Factorial inverses (used in Taylor series) */
/** @{ */

/** @brief 1/2! = 0.5 */
#define FIX_INV_2    FL2FX_CONST(0.5)

/** @brief 1/3! = 1/6 ≈ 0.1667 */
#define FIX_INV_6    FL2FX_CONST(0.16666666666666666)

/** @brief 1/4! = 1/24 ≈ 0.04167 */
#define FIX_INV_24   FL2FX_CONST(0.041666666666666664)

/** @brief 1/5! = 1/120 ≈ 0.008333 */
#define FIX_INV_120  FL2FX_CONST(0.008333333333333333)

/** @brief 1/6! = 1/720 ≈ 0.0013889 */
#define FIX_INV_720  FL2FX_CONST(0.001388888888888889)

/** @brief 1/7! = 1/5040 ≈ 0.0001984 */
#define FIX_INV_5040 FL2FX_CONST(0.00019841269841269841)

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
 */
#define FIXED_DIV(A,B) \
    ((fixed)((((dfixed)(A) << FIX_FRC_SZ) + ((dfixed)(B) >> 1)) / (dfixed)(B)))

/** @} */

//////////////////////////////////// Double Fixed Macros ////////////////////////////////////

/** @defgroup dfixed_macros Double precision operations (dfixed) */
/** @{ */

/** @brief Converts fixed to dfixed (increases precision). */
#define FIXED_TO_DFIXED(A)      \
    ((dfixed)(A) << FIX_FRC_SZ)

/** @brief Shortcut for FIXED_TO_DFIXED. */
#define FX2DFX(A) FIXED_TO_DFIXED(A)

/** @brief Converts dfixed to fixed (reduces precision). */
#define DFIXED_TO_FIXED(A)      \
    ((dfixed)(A) >> FIX_FRC_SZ)

/** @brief Shortcut for DFIXED_TO_FIXED. */
#define DFX2FX(A) DFIXED_TO_FIXED(A)

/** @brief Converts integer to dfixed. */
#define INT_TO_DFIXED(A) \
    ((dfixed)(A) << 2*FIX_FRC_SZ)

/** @brief Alias for INT_TO_DFIXED */
#define INT2DFX(A) INT_TO_DFIXED(A)

/** @brief Multiplies two fixed values promoted to dfixed. */
#define DFIXED_MUL(A,B)            \
    ((dfixed)(((dfixed)(A) * (dfixed)(B)) ))

/**
 * @brief Division with result in dfixed (higher precision).
 * @param A fixed dividend.
 * @param B fixed divisor.
 * @return Quotient in dfixed.
 */
#define DFIXED_DIV(A,B) \
    ((((dfixed)(A) << FIX_DFRC_SZ) + ((dfixed)(B) >> 1)) / (dfixed)(B))

/**
 * @brief Division between two dfixed values, result in dfixed.
 * @param A dfixed dividend.
 * @param B dfixed divisor.
 * @return Quotient in dfixed.
 */
#define DFIXED_DDIV(A,B) \
    ((((dfixed)(A) << FIX_FRC_SZ) + ((dfixed)(B) >> 1)) / (dfixed)(B))

/**
 * @brief Division of fixed by integer with rounding.
 * @param A Dividend (fixed).
 * @param B Divisor (integer).
 * @return Quotient with rounding.
 */
#define FIXED_DIV_INT(A, B) ((A) >= 0 ? ((A) + ((B) >> 1)) / (B) : ((A) - ((B) >> 1)) / (B))

/**
 * @brief Division of dfixed by integer with rounding.
 * @param A Dividend (dfixed).
 * @param B Divisor (integer).
 * @return Quotient with rounding.
 */
#define DFIXED_DDIV_INT(A, B) FIXED_DIV_INT(A, B)

/**
 * @brief Addition of two dfixed values.
 * @param A First operand (dfixed).
 * @param B Second operand (dfixed).
 * @return Sum in dfixed.
 */
#define DFIXED_ADD(A,B) ((dfixed)(A) + (dfixed)(B))

/**
 * @brief Subtraction of two dfixed values.
 * @param A Minuend (dfixed).
 * @param B Subtrahend (dfixed).
 * @return Difference in dfixed.
 */
#define DFIXED_SUB(A,B) ((dfixed)(A) - (dfixed)(B))


/**
 * @brief Multiply-Accumulate for fixed-point (acc += w * x).
 * @param acc Accumulator (dfixed).
 * @param w First operand (fixed).
 * @param x Second operand (fixed).
 *
 * Multiplies two fixed values and accumulates in dfixed precision.
 * Avoids rescaling overhead of FIXED_MUL, preserving accuracy.
 * Ideal for CNN/DSP operations.
 */
#define DFIXED_MAC(acc,w,x) \
    ((acc) += ((dfixed)(w) * (dfixed)(x)))

/**
 * @brief Multiply-Accumulate with extended accumulator (acc += w * x).
 * @param acc Accumulator (int32_t).
 * @param w First operand (fixed).
 * @param x Second operand (fixed).
 *
 * Uses 32-bit accumulator to prevent overflow in long accumulations
 * (e.g., deep convolutions). Maintains same fractional scale as DFIXED_MAC.
 */
#define DDFIXED_MAC(acc,w,x) \
    ((acc) += ((int32_t)(w) * (int32_t)(x)))

/** @brief Scale factor for dfixed floating-point conversion. */
#define DFIXED_SCALE ((dfixed)(1 << FIX_DFRC_SZ))

/** @brief Converts dfixed to float. */
#define DFX2FL(x) (((float)x) / (float)DFIXED_SCALE)

/** @brief Converts float to dfixed. */
#define FL2DFX(x) ((dfixed)(x * (float)DFIXED_SCALE))

/** @brief Converts dfixed to double. */
#define DFX2DB(x) (((double)x) / (double)DFIXED_SCALE)

/** @brief Converts double to dfixed. */
#define DB2DFX(x) ((dfixed)(x * (double)DFIXED_SCALE))

/**
 * @brief Converts dfixed to fixed with rounding.
 * @param x Value in dfixed.
 * @return Rounded fixed value.
 */
#define DFX2FX_RND(x) \
    ((fixed)(((x) + ((dfixed)1 << (FIX_FRC_SZ - 1))) >> FIX_FRC_SZ))

/**
 * @brief Converts dfixed to fixed with saturation.
 * @param x Value in dfixed.
 * @return Saturated fixed value.
 */
#define DFX2FX_SAT(x) \
    ((fixed)( \
        (x) < DFIX_MIN ? FIX_MIN : \
        (x) > DFIX_MAX ? FIX_MAX : \
        DFX2FX(x) \
    ))

/**
 * @brief Converts dfixed to fixed with rounding and saturation.
 * @param x Value in dfixed.
 * @return Rounded and saturated fixed value.
 */
#define DFX2FX_RND_SAT(x) \
    ((fixed)( \
        (x) < DFIX_MIN ? FIX_MIN : \
        (x) > DFIX_MAX ? FIX_MAX : \
        DFX2FX_RND(x) \
    ))

/**
 * @brief Clamps value between min and max.
 * @param X Value to clamp.
 * @param MIN Minimum bound.
 * @param MAX Maximum bound.
 * @return Clamped value.
 */
#define CLAMP(X,MIN,MAX) ((X)<(MIN)?(MIN):((X)>(MAX)?(MAX):(X)))

/**
 * @brief Clamps dfixed to fixed range during conversion.
 * @param X Value in dfixed.
 * @param MIN Minimum in fixed.
 * @param MAX Maximum in fixed.
 * @return Clamped fixed value.
 */
#define CLAMP_DFX_TO_FX(X, MIN, MAX) ((X<FX2DFX(MIN))?MIN:((X>FX2DFX(MAX))?MAX:DFX2FX(X)))

/**
 * @brief Saturates dfixed to valid fixed range.
 * @param X Value in dfixed.
 * @return Saturated fixed value.
 */
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

/**
 * @defgroup reciprocal_ops Reciprocal-based division optimization
 *
 * These macros optimize division by using pre-calculated reciprocals,
 * which is much faster on MCUs without hardware division.
 *
 * The precision parameter (PREC) determines accuracy vs. range trade-off:
 * - Higher PREC = more accurate but smaller maximum dividend
 * - FIX_DFRC_SZ (8) is optimal for most neural network operations
 *
 * @{
 */

/**
 * @brief Calculate reciprocal with given precision
 * @param DIVISOR Integer divisor (>0)
 * @param PREC Fractional bits for reciprocal (0-16)
 * @return Reciprocal as uint32_t with PREC fractional bits
 */
#define FIXED_RECIP(DIVISOR, PREC) (((uint32_t)1 << (PREC)) / (DIVISOR))

/**
 * @brief Divide using pre-calculated reciprocal
 * @param VALUE Value to divide (any integer type)
 * @param RECIP Reciprocal from FIXED_RECIP
 * @param PREC Precision used for reciprocal
 * @return VALUE / divisor (implied) with rounding
 */
#define FIXED_DIV_RECIP(VALUE, RECIP, PREC) \
    (((VALUE) * (RECIP) + (1 << ((PREC) - 1))) >> (PREC))

/** @} */

/**
 * @defgroup dfixed_ops dfixed-optimized operations (recommended for neural nets)
 *
 * These macros use FIX_DFRC_SZ precision automatically and are the preferred
 * choice for most neural network operations.
 *
 * @{
 */

/**
 * @brief Reciprocal using standard dfixed precision
 * @param DIVISOR Integer divisor
 * @return Reciprocal as dfixed
 */
#define DFIXED_RECIP(DIVISOR) ((dfixed)(((uint32_t)1 << FIX_DFRC_SZ) / (DIVISOR)))

/**
 * @brief Average using dfixed accumulation
 * @param SUM Accumulated sum (dfixed)
 * @param COUNT Number of elements
 * @return Average in dfixed format
 */
#define DFIXED_AVG(SUM, COUNT) \
    ((dfixed)(((SUM) * DFIXED_RECIP(COUNT) + ((dfixed)1 << (FIX_DFRC_SZ - 1))) >> FIX_DFRC_SZ))


/**
 * @brief Average with direct fixed output (most common use case)
 * @param SUM Accumulated sum (dfixed)
 * @param COUNT Number of elements
 * @return Average as fixed with saturation
 */
#define FIXED_AVG(SUM, COUNT) \
    DFX2FX_RND_SAT(DFIXED_AVG(SUM, COUNT))

/** @} */

/**
 * @brief Precompute reciprocal for average pooling operations.
 * Returns dfixed to safely store larger reciprocal values across all fixed variants.
 * @param COUNT Number of elements to average (pool_cells)
 * @return Reciprocal as dfixed with FIX_AVG_PREC fractional bits
 */
#define FIXED_AVG_RECIP(COUNT) \
    ((dfixed)(((uint32_t)1 << FIX_AVG_PREC) / (COUNT)))

#define FIXED_AVG_APPLY(SUM, RECIP)    FIXED_DIV_RECIP(SUM, RECIP, FIX_AVG_PREC)



/**
 * @defgroup safe_ops Safe arithmetic operations
 * @{
 */

/**
 * @brief Safe fixed division with saturation
 * @param NUM Numerator (fixed)
 * @param DEN Denominator (fixed, non-zero)
 * @return NUM/DEN as fixed with saturation
 */
#define FIXED_DIV_SAFE(NUM, DEN) \
    DFX2FX_RND_SAT(DFIXED_DIV(NUM, DEN))

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