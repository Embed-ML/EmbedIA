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
 * @brief Implementation of 8-bit fixed-point arithmetic.
 *
 * Uses 16-bit dfixed for intermediate operations to prevent overflow.
 * Implements Taylor series for transcendental functions.
 *
 * @note Default configuration: FIX_FRC_SZ = 4
 * @warning Division functions don't handle division by zero.
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
 * @brief Calculates square root using Newton's method.
 * @param a Non-negative fixed-point value.
 * @return sqrt(a) or -1 if a < 0.
 */
fixed fixed_sqrt(fixed a){
    int invert = 0;
    int iter = FIX_FRC_SZ + 2;  // Más iteraciones para mejor precisión
    int l, i;

    if (a < 0)
        return (-1);
    if (a == 0 || a == FIX_ONE)
        return (a);
    if (a < FIX_ONE && a > 6) {
        invert = 1;
        a = FIXED_DIV(FIX_ONE, a);
    }
    if (a > FIX_ONE) {
        int s = a;

        iter = 0;
        while (s > 0) {
            s >>= 2;
            iter++;
        }
        iter += 2;  // Iteraciones adicionales
    }

    // Newton's iterations con mejor estimación inicial
    l = (a >> 1) + 1;
    for (i = 0; i < iter; i++) {
        // Usar dfixed para mantener precisión en la división
        dfixed a_d = FIXED_TO_DFIXED(a);
        dfixed l_d = FIXED_TO_DFIXED(l);
        dfixed div_d = DFIXED_DDIV(a_d, l_d);
        fixed div = DFIXED_TO_FIXED(div_d);
        l = (l + div) >> 1;
    }
    
    if (invert)
        return (FIXED_DIV(FIX_ONE, l));
    return (l);
}

/**
 * @brief Calculates exponential function exp(x).
 * @param x Exponent in fixed-point.
 * @return e^x, saturated to valid range.
 */
fixed fixed_exp(fixed x) {
    if (x == FIX_ZERO) return FIX_ONE;
    if (x == FIX_ONE)  return FIX_E;
    if (x >= FIX_EXP_MAX) return FIX_MAX;
    if (x <= -FIX_EXP_MAX) return FIX_ZERO;

    uint8_t neg = (x < 0);
    if (neg) x = -x;

    x >>= 1;  // exp(x) = exp(x/2)²

    fixed x2 = FIXED_MUL(x, x);

    fixed r = FIX_ONE + x;
    r += FIXED_MUL(x2, FL2FX_CONST(0.5));
    r += FIXED_MUL(FIXED_MUL(x2, x), FL2FX_CONST(1.0/6));

    if (neg) r = FIXED_DIV(FIX_ONE, r);

    return FIXED_MUL(r, r);
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
 * @brief Calculates natural logarithm ln(x).
 * @param x Positive fixed-point value.
 * @return ln(x) or special value if x <= 0.
 */
 /*
fixed fixed_log(fixed x){
    fixed log2, xi;
    fixed f, s, z, w, R;
    const fixed LN2 = FLOAT_TO_FIXED(0.69314718055994530942);
    const fixed LG[7] = {
        FLOAT_TO_FIXED(6.666666666666735130e-01),
        FLOAT_TO_FIXED(3.999999999940941908e-01),
        FLOAT_TO_FIXED(2.857142874366239149e-01),
        FLOAT_TO_FIXED(2.222219843214978396e-01),
        FLOAT_TO_FIXED(1.818357216161805012e-01),
        FLOAT_TO_FIXED(1.531383769920937332e-01),
        FLOAT_TO_FIXED(1.479819860511658591e-01)
    };

    if (x < 0)
        return (0);
    if (x == 0)
        return  -FIX_ONE;

    log2 = 0;
    xi = x;
    while (xi > FIX_TWO) {
        xi >>= 1;
        log2++;
    }
    f = xi - FIX_ONE;
    s = FIXED_DIV(f, FIX_TWO + f);
    z = FIXED_MUL(s, s);
    w = FIXED_MUL(z, z);
    R = FIXED_MUL(w, LG[1] + FIXED_MUL(w, LG[3]
        + FIXED_MUL(w, LG[5]))) + FIXED_MUL(z, LG[0]
        + FIXED_MUL(w, LG[2] + FIXED_MUL(w, LG[4]
        + FIXED_MUL(w, LG[6]))));
    return (FIXED_MUL(LN2, (log2 << FIX_FRC_SZ)) + f - FIXED_MUL(s, f - R));
}
*/
fixed fixed_log(fixed x) {

    if (x <= 0) return -FIX_ONE;

    /* Tabla de ln(a) para a ∈ [1, 2), paso 0.125 = 2^(-3)
     * 9 entries + 1 sentinela = 10 entries = 40 bytes
     * Misma estructura que fixed_exp — consistente         */
    static const fixed log_table[10] = {
        FL2FX_CONST(0.00000000), /* [0] ln(1.000) */
        FL2FX_CONST(0.11778304), /* [1] ln(1.125) */
        FL2FX_CONST(0.22314355), /* [2] ln(1.250) */
        FL2FX_CONST(0.31845373), /* [3] ln(1.375) */
        FL2FX_CONST(0.40546511), /* [4] ln(1.500) */
        FL2FX_CONST(0.48550782), /* [5] ln(1.625) */
        FL2FX_CONST(0.55961579), /* [6] ln(1.750) */
        FL2FX_CONST(0.62860866), /* [7] ln(1.875) */
        FL2FX_CONST(0.69314718), /* [8] ln(2.000) — sentinela    */
        FL2FX_CONST(0.75377180), /* [9] ln(2.125) — sentinela cúbico */
    };

    /* ln(2) para corrección de parte entera */
    static const fixed LN2 = FL2FX_CONST(0.69314718);

    /* 1. Normalizar x a [1, 2): x = m * 2^n
     * ln(x) = ln(m) + n*ln(2)
     * shift aritmético — sin FIXED_MUL                    */
    int n = 0;
    fixed m = x;

    while (m >= FIX_TWO) { m >>= 1; n++;  }
    while (m <  FIX_ONE) { m <<= 1; n--; }

    /* 2. idx = floor((m - 1) / 0.125)
     *       = floor((m - FIX_ONE) >> (FIX_FRC_SZ - 3))   */
    fixed f   = m - FIX_ONE;                          /* f ∈ [0, FIX_ONE) */
    unsigned int idx = (unsigned int)(f >> (FIX_FRC_SZ - 3));
    if (idx > 7) idx = 7;

    fixed base = (fixed)idx << (FIX_FRC_SZ - 3);
    fixed t    = (f - base) << 3;                     /* t ∈ [0, FIX_ONE) */

    fixed a = log_table[idx];
    fixed b = log_table[idx + 1];
    fixed c = log_table[idx + 2];

    /* Horner cuadrático: 2 FIXED_MUL
     * f(t) = a + t*(d1 + (t-1)*d2/2)                     */
    fixed d1    = b - a;
    fixed d2    = c - (b << 1) + a;
    fixed inner = FIXED_MUL(t - FIX_ONE, d2) >> 1;
    fixed log_m = a + FIXED_MUL(t, inner + d1);

    /* 3. ln(x) = ln(m) + n*ln(2)
     * n*ln(2): FIXED_MUL solo si n != 0, frecuentemente n es pequeño */
    fixed result = log_m;
    if (n != 0)
        result += FIXED_MUL(INT_TO_FIXED(n), LN2);   /* 1 FIXED_MUL */

    return result;
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

/**
 * @brief Calculates hyperbolic tangent tanh(x).
 * @param x Fixed-point value.
 * @return tanh(x) using continued fraction approximation.
 */
fixed fixed_tanh(fixed x){
    int i;
    int sum_const_size = 5;
    fixed sum_const[] = {
        FLOAT_TO_FIXED(1.0),
        FLOAT_TO_FIXED(3.0),
        FLOAT_TO_FIXED(5.0),
        FLOAT_TO_FIXED(7.0),
        FLOAT_TO_FIXED(9.0),
    };

    fixed x2 = FIXED_MUL(x,x);
    fixed temp = x2;

    if(x>=FIX_TWO){
        temp = FIX_ONE;
    }else{
        if(x<=-FIX_TWO){
            temp = -FIX_ONE;
        }else{
            for(i=sum_const_size-1; i>0; i--){
                temp = FIXED_DIV(x2 ,(sum_const[i] + temp));
            }
            temp = FIXED_DIV(x  ,(sum_const[0] + temp));
        }
    }

    return temp;
}



/**
 * @brief Calculates sine using Taylor series.
 * @param x Angle in radians (fixed-point).
 * @return sin(x).
 */
fixed fixed_sin(fixed x) {
    // Reducir a [0, 2π)
    if (x >= FIX_2PI || x <= -FIX_2PI) {
        // Usar división mejorada para reducción de rango
        dfixed x_d = FIXED_TO_DFIXED(x);
        dfixed pi2_d = FIXED_TO_DFIXED(FIX_2PI);
        dfixed k_d = DFIXED_DDIV(x_d, pi2_d);
        fixed k = DFIXED_TO_FIXED(k_d);
        x -= FIXED_MUL(k, FIX_2PI);
    }

    if (x < 0) x += FIX_2PI;

    // Determinar signo y reducir a [0, π]
    fixed sign = FIX_ONE;
    if (x > FIX_PI) {
        x = FIX_2PI - x;
        sign = FIXED_NEG(FIX_ONE);
    }

    // Reducir a [0, π/2]
    if (x > FIX_PI_INV2) {
        x = FIX_PI - x;
    }

    // Usar identidad de coseno para x > π/4
    if (x > FIX_PI_INV4) {
        fixed y = FIX_PI_INV2 - x;
        fixed y2 = FIXED_MUL(y, y);

        // Polinomio de coseno con mejor precisión
        fixed term1 = FIXED_MUL(y2, FIX_INV_2);
        fixed term2 = FIXED_MUL(y2, FIXED_MUL(y2, FIX_INV_24));
        fixed term3 = FIXED_MUL(y2, FIXED_MUL(y2, FIXED_MUL(y2, FIX_INV_720)));
        
        fixed cos_y = FIX_ONE - term1 + term2 - term3;

        return sign == FIX_ONE ? cos_y : FIXED_NEG(cos_y);
    }

    // Polinomio de seno con mejor precisión
    fixed x2 = FIXED_MUL(x, x);
    fixed x3 = FIXED_MUL(x, x2);
    
    fixed term1 = FIXED_MUL(x3, FIX_INV_6);
    fixed term2 = FIXED_MUL(x2, FIXED_MUL(x3, FIX_INV_120));
    fixed term3 = FIXED_MUL(x2, FIXED_MUL(x2, FIXED_MUL(x3, FIX_INV_5040)));
    
    fixed sin_x = x - term1 + term2 - term3;

    return sign == FIX_ONE ? sin_x : FIXED_NEG(sin_x);
}

/**
 * @brief Calculates cosine using sine identity.
 * @param x Angle in radians (fixed-point).
 * @return cos(x).
 */
fixed fixed_cos(fixed x) {
    return fixed_sin(FIXED_ADD(x, FIXED_DIV(FIX_PI, FL2FX_CONST(2.0))));
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