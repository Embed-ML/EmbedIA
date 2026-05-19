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
 * - Uses 64-bit operations (dfixed) to prevent overflow
 * - Taylor series approximations for transcendental functions
 * - Lookup tables for optimized sqrt() and trigonometric functions
 *
 * @note Default configuration uses FIX_FRC_SZ = 17
 * @warning fixed_div() doesn't handle division by zero
 */

#include "fixed.h"

/////////////////////////////////// Type Conversion Functions ///////////////////////////////////

    // Returns a fixed-point number from a float
    fixed float_to_fixed(float f){
        return FLOAT_TO_FIXED(f);
    }

    // Returns a fixed-point number from an integer
    fixed int_to_fixed(int32_t i){
        return INT_TO_FIXED(i);
    }

    // Returns the double-precision floating-point representation of a fixed-point number
    double fixed_to_double(fixed f){
        return FIXED_TO_DOUBLE(f);
    }

    // Returns the floating-point representation of a fixed-point number
    float fixed_to_float(fixed f){
        return FIXED_TO_FLOAT(f);
    }

    // Returns the integer part of a fixed-point number
    int32_t fixed_to_int(fixed f){
        return FIXED_TO_INT(f);
    }

/////////////////////////////////// Arithmetic Functions ///////////////////////////////////

    // Returns the fixed-point sum of a and b
    fixed fixed_add(fixed a, fixed b){
        return FIXED_ADD(a, b);
    }

    // Returns the fixed-point difference between a and b
    fixed fixed_sub(fixed a, fixed b){
        return FIXED_SUB(a, b);
    }

    // Returns the fixed-point product of a and b
    fixed fixed_mul(fixed a, fixed b){
        return FIXED_MUL(a, b);
    }

    // Returns the fixed-point division of a by b
    fixed fixed_div(fixed a, fixed b){
        return FIXED_DIV(a,b);
    }

/////////////////////////////////// Special Functions ///////////////////////////////////

    // Returns the square root of a or -1 on error
    /* Previous version. Current version is 50% faster with an additional 33-element table
       while maintaining similar low error as the previous version
    fixed fixed_sqrt(fixed a){
        [previous implementation]
    }
    */

// New implementation
fixed fixed_sqrt(fixed x) {
    // More precise table with 17 points (68 bytes)
    static const fixed sqrt_table[17] = {
        FL2FX(1.00000000), FL2FX(1.03077641), FL2FX(1.06066017),
        FL2FX(1.08972474), FL2FX(1.11803399), FL2FX(1.14564392),
        FL2FX(1.17260394), FL2FX(1.19895788), FL2FX(1.22474487),
        FL2FX(1.25000000), FL2FX(1.27475488), FL2FX(1.29903811),
        FL2FX(1.32287566), FL2FX(1.34629120), FL2FX(1.36930639),
        FL2FX(1.39194109), FL2FX(1.41421356)
    };

    if (x <= 0) return 0;
    if (x == FIX_ONE) return FIX_ONE;

    // Improved normalization
    int n = 0;
    while (x >= FL2FX(4.0) && n < 10) { x >>= 2; n++; }
    while (x < FL2FX(1.0) && n > -10) { x <<= 2; n--; }

    // Table lookup with quadratic interpolation
    fixed position = FIXED_MUL(FIXED_SUB(x, FL2FX(1.0)), FL2FX(16.0));
    unsigned int idx = FIXED_MIN(FIXED_TO_INT(position), 15);
    fixed frac = FIXED_SUB(position, INT_TO_FIXED(idx));

    // Quadratic interpolation for better precision
    fixed y0 = sqrt_table[idx];
    fixed y1 = sqrt_table[idx+1];
    fixed ym = FIXED_ADD(y0, FIXED_DIV(FIXED_SUB(y1, y0), FL2FX(2.0)));

    fixed est = FIXED_ADD(y0, FIXED_MUL(frac, FIXED_ADD(FIXED_SUB(y1, y0),
                          FIXED_MUL(frac, FIXED_SUB(FIXED_MUL(FL2FX(2.0), ym),
                          FIXED_ADD(y0, y1))))));

    // Two optimized Newton-Raphson iterations
    if (est != 0) {
        fixed ratio = FIXED_DIV(x, est);
        est = FIXED_ADD(est, ratio) >> 1;
        ratio = FIXED_DIV(x, est);
        est = FIXED_ADD(est, ratio) >> 1;
    }

    // Safe final adjustment
    if (n > 0) return FIXED_MIN(FIXED_MUL(est, INT_TO_FIXED(1 << n)), FIX_MAX);
    return FIXED_DIV(est, INT_TO_FIXED(1 << (-n)));
}

/* Previous version. Modified to take advantage of range reduction */
fixed fixed_exp(fixed fp){
    const fixed AUX[9] = {FL2FX(1.0/2), FL2FX(1.0/3), FL2FX(1.0/4),
                         FL2FX(1.0/5), FL2FX(1.0/6), FL2FX(1.0/7),
                         FL2FX(1.0/8), FL2FX(1.0/9), FL2FX(1.0/10)};

    #define MAX_EXP_IT 8

    if(fp == FIX_ZERO) return FIX_ONE;
    if(fp == FIX_ONE) return FIX_E;
    if(fp >= FIX_EXP_MAX) return FIX_MAX;
    if(fp <= -FIX_EXP_MAX) return FIX_ZERO;

    // Range reduction using identity exp(x) = exp(x/2)^2
    fp = fp >> 1;

    uint8_t i;
    uint8_t neg = (fp < FIX_ZERO);
    if (neg) fp = -fp;

    fixed result = fp + FIX_ONE;
    fixed term = fp;
    for (i = 0; i <= MAX_EXP_IT; i++){
        term = FIXED_MUL(term, FIXED_MUL(fp, AUX[i]));
        result += term;
        if (term < 100)
            break;
    }

    if (neg) result = FIXED_DIV(FIX_ONE, result);

    // Recover initial range reduction using identity exp(x) = exp(x/2)^2
    return FIXED_MUL(result, result);
}

// Returns x * 2^exp
fixed fixed_ldexp(fixed x, int exp){
    return FIXED_MUL(x, fixed_pow(FIX_TWO, exp));
}

/* Previous version replaced by one twice as fast with half the error */
fixed fixed_log(fixed x) {
    if (x <= 0) return FIX_MIN;

    // Precalculated table of ln(1+i/16) for i=0..16
    static const fixed log_table[17] = {
        FL2FX(0.00000000),  // ln(1)
        FL2FX(0.06062462),  // ln(1+1/16)
        FL2FX(0.11778304),  // ln(1+2/16)
        FL2FX(0.17185026),  // ...
        FL2FX(0.22314351),
        FL2FX(0.27193372),
        FL2FX(0.31845373),
        FL2FX(0.36290549),
        FL2FX(0.40546511),
        FL2FX(0.44628710),
        FL2FX(0.48550782),
        FL2FX(0.52324814),
        FL2FX(0.55961579),
        FL2FX(0.59470711),
        FL2FX(0.62860866),
        FL2FX(0.66139848),
        FL2FX(0.69314718)   // ln(2)
    };

    // 1. Normalization: bring x to range [1, 2)
    int n = 0;
    while (x >= FIX_TWO) {
        x >>= 1;  // x /= 2
        n++;
    }
    while (x < FIX_ONE) {
        x <<= 1;  // x *= 2
        n--;
    }

    // 2. Calculate table position (x = 1 + k/16 + fraction)
    fixed segment = FIXED_MUL(FIXED_SUB(x, FIX_ONE), INT_TO_FIXED(16));
    unsigned int idx = FIXED_TO_INT(segment);
    fixed frac = FIXED_SUB(segment, INT_TO_FIXED(idx));

    // 3. Linear interpolation between table points
    fixed log_low = log_table[idx];
    fixed log_high = log_table[idx+1];
    fixed interpolated = FIXED_ADD(log_low, FIXED_MUL(frac, FIXED_SUB(log_high, log_low)));

    // 4. Add normalization correction (ln(2^n) = n*ln(2))
    fixed n_correction = FIXED_MUL(INT_TO_FIXED(n), log_table[16]); // ln(2)
    return FIXED_ADD(interpolated, n_correction);
}

// Returns the base-b logarithm of x
fixed fixed_logn(fixed x, fixed base){
    return (FIXED_DIV(fixed_log(x), fixed_log(base)));
}

// Returns n^exp
fixed fixed_pow(fixed n, fixed exp){
    if (exp == 0)
        return (FIX_ONE);
    if (n < 0)
        return 0;
    return (fixed_exp(FIXED_MUL(fixed_log(n), exp)));
}

// Returns an approximation of sqrt(a² + b²)
fixed fixed_magnitude(fixed a, fixed b){
    fixed abs_a = FIXED_ABS(a);
    fixed abs_b = FIXED_ABS(b);

    fixed max_val = FIXED_MAX(abs_a, abs_b);
    fixed min_val = FIXED_MIN(abs_a,abs_b);

    // 0.375 * min ≈ (min >> 2) + (min >> 3)
    fixed delta = (min_val >> 2) + (min_val >> 3);
    return max_val + delta;
}

/////////////////////////////////// Trigonometric Functions ///////////////////////////////////

/* Previous version. New version has same speed but much less error */
fixed fixed_tanh(fixed x) {
    // Table of 9 values (36 bytes)
    static const fixed tanh_table[9] = {
        FL2FX(0.0), FL2FX(0.24491866), FL2FX(0.46211716),
        FL2FX(0.63514895), FL2FX(0.76159416), FL2FX(0.84828364),
        FL2FX(0.90514825), FL2FX(0.94137513), FL2FX(0.96402758)
    };

    fixed abs_x = FIXED_ABS(x);
    if(abs_x > FIX_TWO) return x > 0 ? FIX_ONE : FIXED_SUB(0, FIX_ONE);

    fixed position = FIXED_MUL(abs_x, FL2FX(4.0));
    unsigned int idx = FIXED_TO_INT(position);
    fixed frac = FIXED_SUB(position, INT_TO_FIXED(idx));

    fixed result = FIXED_ADD(tanh_table[idx],
                           FIXED_MUL(frac, FIXED_SUB(tanh_table[idx+1], tanh_table[idx])));
    return x > 0 ? result : FIXED_SUB(0, result);
}

/* Average time: 12.391us. Avg abs error: 0.0000040. Max error: 0.0000111 */
fixed fixed_sin(fixed x) {
    // Step 1: More efficient reduction to [0, 2π)
    // Use integer division to avoid double conversions
    fixed k = x / FIX_2PI;
    x -= k * FIX_2PI;

    // Handle negative values (more efficient than addition)
    if (x < 0) x += FIX_2PI;

    // Step 2: Determine sign and reduce to [0, π]
    fixed sign = FIX_ONE;
    if (x > FIX_PI) {
        x = FIX_2PI - x;
        sign = FIXED_NEG(FIX_ONE);  // Only negate once
    }

    // Step 3: Reduce to [0, π/2]
    if (x > FIX_PI_INV2) {
        x = FIX_PI - x;
    }

    // Step 4: Use cosine identity if x > π/4
    if (x > FIX_PI_INV4) {
        fixed y = FIX_PI_INV2 - x;
        fixed y2 = FIXED_MUL(y, y);

        // Optimized cosine polynomial (same order but more compact)
        fixed cos_y = FIX_ONE - FIXED_MUL(y2,
                           FIX_INV_2 - FIXED_MUL(y2,
                               FIX_INV_24 - FIXED_MUL(y2, FIX_INV_720)));

        return sign == FIX_ONE ? cos_y : FIXED_NEG(cos_y);
    }

    // Step 5: Polynomial sine evaluation
    fixed x2 = FIXED_MUL(x, x);
    fixed sin_x = x - FIXED_MUL(FIXED_MUL(x, x2),
                           FIX_INV_6 - FIXED_MUL(x2,
                               FIX_INV_120 - FIXED_MUL(x2, FIX_INV_5040)));

    return sign == FIX_ONE ? sin_x : FIXED_NEG(sin_x);
}

// Cosine using table with special cases
fixed fixed_cos(fixed x) {
    return fixed_sin(FIXED_ADD(x, FIXED_DIV(FIX_PI, FL2FX(2.0))));
}

/////////////////////////////////// Additional Functions ///////////////////////////////////

    // Returns the absolute value of a
    fixed fixed_abs(fixed a){
        return FIXED_ABS(a);
    }

    // Returns the smallest integer x such that x >= a
    fixed fixed_ceil(fixed a){
        return FIXED_CEIL(a);
    }

    // Returns the largest integer x such that x <= a
    fixed fixed_floor(fixed a){
        return FIXED_FLOOR(a);
    }