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
 * @file fixed.c
 * @brief Implementation of 32-bit fixed-point arithmetic
 * 
 * @details
 * - Uses 32-bit dfixed operations with extended range (4 additional bits)
 * - Taylor series approximations for transcendental functions
 * - Lookup tables for optimized sqrt() and trigonometric functions
 * 
 * @note Default configuration uses FIX_FRC_SZ = 17, DFIXED uses FIX_DFRC_SZ = 34
 * @warning fixed_div() doesn't handle division by zero
 */

#include "fixed.h"

/////////////////////////////////// Type Conversion Functions ///////////////////////////////////

/**
 * @brief Converts float to fixed-point.
 * @param f Float value.
 * @return Fixed-point value.
 */
fixed float_to_fixed(float f){
    return FLOAT_TO_FIXED(f);
}

/**
 * @brief Converts integer to fixed-point.
 * @param i Integer value.
 * @return Fixed-point value.
 */
fixed int_to_fixed(int32_t i){
    return INT_TO_FIXED(i);
}

/**
 * @brief Converts fixed-point to double.
 * @param f Fixed-point value.
 * @return Double value.
 */
double fixed_to_double(fixed f){
    return FIXED_TO_DOUBLE(f);
}

/**
 * @brief Converts fixed-point to float.
 * @param f Fixed-point value.
 * @return Float value.
 */
float fixed_to_float(fixed f){
    return FIXED_TO_FLOAT(f);
}

/**
 * @brief Extracts integer part from fixed-point.
 * @param f Fixed-point value.
 * @return Integer part.
 */
int32_t fixed_to_int(fixed f){
    return FIXED_TO_INT(f);
}

/////////////////////////////////// Arithmetic Functions ///////////////////////////////////

/**
 * @brief Adds two fixed-point values.
 * @param a First operand.
 * @param b Second operand.
 * @return Sum in fixed-point.
 */
fixed fixed_add(fixed a, fixed b){
    return FIXED_ADD(a, b);
}

/**
 * @brief Subtracts two fixed-point values.
 * @param a Minuend.
 * @param b Subtrahend.
 * @return Difference in fixed-point.
 */
fixed fixed_sub(fixed a, fixed b){
    return FIXED_SUB(a, b);
}

/**
 * @brief Multiplies two fixed-point values.
 * @param a First operand.
 * @param b Second operand.
 * @return Product in fixed-point.
 */
fixed fixed_mul(fixed a, fixed b){
    return FIXED_MUL(a, b);
}

/**
 * @brief Divides two fixed-point values.
 * @param a Dividend.
 * @param b Divisor (must be non-zero).
 * @return Quotient in fixed-point.
 */
fixed fixed_div(fixed a, fixed b){
    return FIXED_DIV(a,b);
}

/////////////////////////////////// Special Functions ///////////////////////////////////

/**
 * @brief Calculates square root using table lookup and Newton-Raphson.
 * @param x Non-negative fixed-point value.
 * @return sqrt(x) or 0 if x <= 0.
 */
fixed fixed_sqrt(fixed x) {
    // Table with 17 points for range [1, 2)
    static const fixed sqrt_table[17] = {
        FL2FX_CONST(1.00000000), FL2FX_CONST(1.03077641), FL2FX_CONST(1.06066017),
        FL2FX_CONST(1.08972474), FL2FX_CONST(1.11803399), FL2FX_CONST(1.14564392),
        FL2FX_CONST(1.17260394), FL2FX_CONST(1.19895788), FL2FX_CONST(1.22474487),
        FL2FX_CONST(1.25000000), FL2FX_CONST(1.27475488), FL2FX_CONST(1.29903811),
        FL2FX_CONST(1.32287566), FL2FX_CONST(1.34629120), FL2FX_CONST(1.36930639),
        FL2FX_CONST(1.39194109), FL2FX_CONST(1.41421356)
    };

    if (x <= 0) return 0;
    if (x == FIX_ONE) return FIX_ONE;

    // Normalization to [1, 4) range
    int n = 0;
    while (x >= FL2FX_CONST(4.0) && n < 10) { x >>= 2; n++; }
    while (x < FL2FX_CONST(1.0) && n > -10) { x <<= 2; n--; }

    // Table lookup with interpolation
    fixed position = FIXED_MUL(FIXED_SUB(x, FL2FX_CONST(1.0)), FL2FX_CONST(16.0));
    unsigned int idx = FIXED_MIN(FIXED_TO_INT(position), 15);
    fixed frac = FIXED_SUB(position, INT_TO_FIXED(idx));

    // Quadratic interpolation
    fixed y0 = sqrt_table[idx];
    fixed y1 = sqrt_table[idx+1];
    fixed ym = FIXED_ADD(y0, FIXED_DIV(FIXED_SUB(y1, y0), FL2FX_CONST(2.0)));

    fixed est = FIXED_ADD(y0, FIXED_MUL(frac, FIXED_ADD(FIXED_SUB(y1, y0),
                          FIXED_MUL(frac, FIXED_SUB(FIXED_MUL(FL2FX_CONST(2.0), ym),
                          FIXED_ADD(y0, y1))))));

    // Newton-Raphson refinement (2 iterations)
    if (est != 0) {
        fixed ratio = FIXED_DIV(x, est);
        est = FIXED_ADD(est, ratio) >> 1;
        ratio = FIXED_DIV(x, est);
        est = FIXED_ADD(est, ratio) >> 1;
    }

    // Denormalization
    if (n > 0) return FIXED_MIN(FIXED_MUL(est, INT_TO_FIXED(1 << n)), FIX_MAX);
    return FIXED_DIV(est, INT_TO_FIXED(1 << (-n)));
}

/**
 * @brief Calculates exponential function exp(x).
 * @param fp Exponent in fixed-point.
 * @return e^x, saturated to valid range.
 */
fixed fixed_exp(fixed fp){
    const fixed AUX[9] = {FL2FX_CONST(1.0/2), FL2FX_CONST(1.0/3), FL2FX_CONST(1.0/4),
                         FL2FX_CONST(1.0/5), FL2FX_CONST(1.0/6), FL2FX_CONST(1.0/7),
                         FL2FX_CONST(1.0/8), FL2FX_CONST(1.0/9), FL2FX_CONST(1.0/10)};

    #define MAX_EXP_IT 8

    if(fp == FIX_ZERO) return FIX_ONE;
    if(fp == FIX_ONE) return FIX_E;
    if(fp >= FIX_EXP_MAX) return FIX_MAX;
    if(fp <= -FIX_EXP_MAX) return FIX_ZERO;

    // Range reduction: exp(x) = exp(x/2)²
    fp = fp >> 1;

    uint8_t i;
    uint8_t neg = (fp < FIX_ZERO);
    if (neg) fp = -fp;

    // Taylor series expansion
    fixed result = fp + FIX_ONE;
    fixed term = fp;
    for (i = 0; i <= MAX_EXP_IT; i++){
        term = FIXED_MUL(term, FIXED_MUL(fp, AUX[i]));
        result += term;
        if (term < 100)
            break;
    }

    if (neg) result = FIXED_DIV(FIX_ONE, result);

    // Recover from range reduction: exp(x) = exp(x/2)²
    return FIXED_MUL(result, result);
}

/**
 * @brief Multiplies x by 2^exp.
 * @param x Base value.
 * @param exp Integer exponent.
 * @return x * 2^exp.
 */
fixed fixed_ldexp(fixed x, int exp){
    return FIXED_MUL(x, fixed_pow(FIX_TWO, exp));
}

/**
 * @brief Calculates natural logarithm ln(x) using lookup table.
 * @param x Positive fixed-point value.
 * @return ln(x) or FIX_MIN if x <= 0.
 */

fixed fixed_log(fixed x) {
    if (x <= 0) return FIX_MIN;

    /* Normalizar a [1, 2): x = m * 2^n, m ∈ [1, 2)
     * ln(x) = ln(m) + n*ln(2)
     * Usando shifts — igual que implementación actual        */
    int n = 0;
    while (x >= FIX_TWO) { x >>= 1; n++; }
    while (x <  FIX_ONE) { x <<= 1; n--; }

    /* Tabla de ln(m) en [1, 2), paso 0.125 = 2^(-3)
     * 9 intervalos + 2 sentinelas para Horner cuadrático
     * 11 entries × 4 bytes = 44 bytes                        */
    static const fixed log_table[11] = {
        FL2FX_CONST(0.00000000), /* [0]  ln(1.000) */
        FL2FX_CONST(0.11778304), /* [1]  ln(1.125) */
        FL2FX_CONST(0.22314355), /* [2]  ln(1.250) */
        FL2FX_CONST(0.31845373), /* [3]  ln(1.375) */
        FL2FX_CONST(0.40546511), /* [4]  ln(1.500) */
        FL2FX_CONST(0.48550782), /* [5]  ln(1.625) */
        FL2FX_CONST(0.55961579), /* [6]  ln(1.750) */
        FL2FX_CONST(0.62860866), /* [7]  ln(1.875) */
        FL2FX_CONST(0.69314718), /* [8]  ln(2.000) — fin rango + sentinela */
        FL2FX_CONST(0.75377180), /* [9]  ln(2.125) — sentinela cuadrático  */
        FL2FX_CONST(0.81093022), /* [10] ln(2.250) — sentinela cuadrático  */
    };

    /* ln(2) precalculado para corrección de escala */
    static const fixed LN2 = FL2FX_CONST(0.69314718);

    /* x está en [1, 2) en Q16.16
     * Restar FIX_ONE para trabajar en [0, 1): f = x - 1.0  */
    fixed f = x - FIX_ONE;  /* f ∈ [0, FIX_ONE) */

    /* idx = floor(f / 0.125) = f >> (FIX_FRC_SZ - 3) */
    unsigned int idx = (unsigned int)(f >> (FIX_FRC_SZ - 3));
    if (idx > 8) idx = 8;

    fixed base = (fixed)idx << (FIX_FRC_SZ - 3);
    fixed t    = (f - base) << 3;  /* t ∈ [0, FIX_ONE) */

    fixed a = log_table[idx];
    fixed b = log_table[idx + 1];
    fixed c = log_table[idx + 2];

    /* Horner cuadrático: 2 FIXED_MUL */
    fixed d1    = b - a;
    fixed d2    = c - (b << 1) + a;
    fixed inner = FIXED_MUL(t - FIX_ONE, d2) >> 1;
    fixed log_m = a + FIXED_MUL(t, inner + d1);

    /* ln(x) = ln(m) + n*ln(2)
     * n*ln(2): shift + add en vez de FIXED_MUL
     * ln(2) ≈ 0.693, para n entero: n*LN2 con shifts y sumas */
    fixed n_correction;
    if (n == 0)       n_correction = 0;
    else if (n > 0)   n_correction =  (fixed) n * LN2;  /* mul entero, no FIXED_MUL */
    else              n_correction = -(fixed)(-n) * LN2;

    return log_m + n_correction;
}

/**
 * @brief Calculates logarithm in base b.
 * @param x Positive value.
 * @param base Logarithm base (positive, ≠ 1).
 * @return log_base(x).
 */
fixed fixed_logn(fixed x, fixed base){
    return (FIXED_DIV(fixed_log(x), fixed_log(base)));
}

/**
 * @brief Calculates power n^exp.
 * @param n Base (must be non-negative).
 * @param exp Exponent.
 * @return n^exp.
 */
fixed fixed_pow(fixed n, fixed exp){
    if (exp == 0)
        return (FIX_ONE);
    if (n < 0)
        return 0;
    return (fixed_exp(FIXED_MUL(fixed_log(n), exp)));
}

/**
 * @brief Fast approximation of sqrt(a² + b²).
 * @param a First value.
 * @param b Second value.
 * @return Magnitude approximation without overflow.
 */
fixed fixed_magnitude(fixed a, fixed b){
    fixed abs_a = FIXED_ABS(a);
    fixed abs_b = FIXED_ABS(b);

    fixed max_val = FIXED_MAX(abs_a, abs_b);
    fixed min_val = FIXED_MIN(abs_a,abs_b);

    // Approximation: max + 0.375*min
    fixed delta = (min_val >> 2) + (min_val >> 3);
    return max_val + delta;
}

/////////////////////////////////// Trigonometric Functions ///////////////////////////////////


fixed fixed_tanh(fixed x) {
    /* Paso 0.25 = 2^(-2), 13 intervalos + 1 sentinela
     * Error máx teórico con Horner grado 3: ~0.00016 << bound 0.00075
     * 14 entries × 4 bytes = 56 bytes ROM                              */
    static const fixed tanh_table[14] = {
        FL2FX_CONST(0.00000000), /* [0]  tanh(0.00) */
        FL2FX_CONST(0.24491866), /* [1]  tanh(0.25) */
        FL2FX_CONST(0.46211716), /* [2]  tanh(0.50) */
        FL2FX_CONST(0.63514895), /* [3]  tanh(0.75) */
        FL2FX_CONST(0.76159416), /* [4]  tanh(1.00) */
        FL2FX_CONST(0.84828364), /* [5]  tanh(1.25) */
        FL2FX_CONST(0.90514825), /* [6]  tanh(1.50) */
        FL2FX_CONST(0.94137513), /* [7]  tanh(1.75) */
        FL2FX_CONST(0.96402758), /* [8]  tanh(2.00) */
        FL2FX_CONST(0.97802611), /* [9]  tanh(2.25) */
        FL2FX_CONST(0.98661430), /* [10] tanh(2.50) */
        FL2FX_CONST(0.99130655), /* [11] tanh(2.75) */
        FL2FX_CONST(0.99505475), /* [12] tanh(3.00) */
        FL2FX_CONST(0.99777080), /* [13] tanh(3.25) — sentinela */
    };

    fixed abs_x = FIXED_ABS(x);

    if (abs_x >= FL2FX_CONST(3.0))
        return (x >= 0) ? FIX_ONE : -FIX_ONE;

    /* paso = 0.25 = 2^(-2)
     * idx = floor(abs_x / 0.25) = abs_x >> (FIX_FRC_SZ - 2)          */
    unsigned int idx = (unsigned int)(abs_x >> (FIX_FRC_SZ - 2));
    if (idx > 11) idx = 11;

    fixed base = (fixed)idx << (FIX_FRC_SZ - 2);
    fixed t    = (abs_x - base) << 2;  /* t ∈ [0, FIX_ONE) */

    fixed a = tanh_table[idx];
    fixed b = tanh_table[idx + 1];
    fixed c = tanh_table[idx + 2];
    fixed d = tanh_table[idx + 3];

    /* Terceras diferencias finitas forward */
    fixed d1 = b - a;
    fixed d2 = c - (b << 1) + a;
    fixed d3 = d - c - c - c + (b << 1) + b - a;  /* d - 3c + 3b - a */

    /* Horner grado 3: f(t) = a + t*(d1 + (t-1)*(d2/2 + (t-2)*d3/6))
     * 3 FIXED_MUL en total, mismo costo que cuadrático actual.
     *
     * d3/6: multiplicar por 1/6 en Q16.16 = 10922
     * Se aplica antes de FIXED_MUL para mantener rango               */


    fixed inner = (FIXED_MUL(t - (FIX_ONE << 1), FIXED_MUL(d3, FIX_INV_6)) >> 0)
                + (d2 >> 1);
    fixed mid   = FIXED_MUL(t - FIX_ONE, inner) + d1;
    fixed result = a + FIXED_MUL(t, mid);

    return (x >= 0) ? result : -result;
}


/* Average time: 12.391us. Avg abs error: 0.0000040. Max error: 0.0000111 */
fixed fixed_sin(fixed x) {
    // Step 1: Reduce to [0, 2π)
    /* posible mejora para evitar division?
    dfixed tmp = (dfixed)x * (dfixed)FIX_INV_2PI;
    int32_t k = (int32_t)(tmp >> (FIX_FRC_SZ * 2));
    */
    fixed k = x / FIX_2PI;
    x -= k * FIX_2PI;

    // Handle negative values
    if (x < 0) x += FIX_2PI;

    // Step 2: Determine sign and reduce to [0, π]
    fixed sign = FIX_ONE;
    if (x > FIX_PI) {
        x = FIX_2PI - x;
        sign = FIXED_NEG(FIX_ONE);
    }

    // Step 3: Reduce to [0, π/2]
    if (x > FIX_PI_INV2) {
        x = FIX_PI - x;
    }

    // Step 4: Use cosine identity for x > π/4
    if (x > FIX_PI_INV4) {
        fixed y = FIX_PI_INV2 - x;
        fixed y2 = FIXED_MUL(y, y);

        // Cosine polynomial
        fixed cos_y = FIX_ONE - FIXED_MUL(y2,
                           FIX_INV_2 - FIXED_MUL(y2,
                               FIX_INV_24 - FIXED_MUL(y2, FIX_INV_720)));

        return sign == FIX_ONE ? cos_y : FIXED_NEG(cos_y);
    }

    // Step 5: Sine polynomial evaluation
    fixed x2 = FIXED_MUL(x, x);
    fixed sin_x = x - FIXED_MUL(FIXED_MUL(x, x2),
                           FIX_INV_6 - FIXED_MUL(x2,
                               FIX_INV_120 - FIXED_MUL(x2, FIX_INV_5040)));

    return sign == FIX_ONE ? sin_x : FIXED_NEG(sin_x);
}

/**
 * @brief Calculates cosine using sine identity.
 * @param x Angle in radians (fixed-point).
 * @return cos(x).
 */
fixed fixed_cos(fixed x) {

    fixed k = x / FIX_2PI;
    x -= k * FIX_2PI;

    if (x < 0) x += FIX_2PI;

    int sign = 1;

    if (x > FIX_PI) {
        x = FIX_2PI - x;
    }

    if (x > FIX_PI_INV2) {
        x = FIX_PI - x;
        sign = -1;
    }

    fixed x2 = FIXED_MUL(x, x);

    fixed cos_x = FIX_ONE - FIXED_MUL(x2,
                      FIX_INV_2 - FIXED_MUL(x2,
                      FIX_INV_24 - FIXED_MUL(x2, FIX_INV_720)));

    return (sign > 0) ? cos_x : -cos_x;
}

/////////////////////////////////// Additional Functions ///////////////////////////////////

/**
 * @brief Returns absolute value.
 * @param a Fixed-point value.
 * @return |a|.
 */
fixed fixed_abs(fixed a){
    return FIXED_ABS(a);
}

/**
 * @brief Rounds up to next integer.
 * @param a Fixed-point value.
 * @return Ceiling of a.
 */
fixed fixed_ceil(fixed a){
    return FIXED_CEIL(a);
}

/**
 * @brief Rounds down to integer.
 * @param a Fixed-point value.
 * @return Floor of a.
 */
fixed fixed_floor(fixed a){
    return FIXED_FLOOR(a);
}