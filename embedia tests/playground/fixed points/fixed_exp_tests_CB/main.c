#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "../embedia/fixed32/embedia.h"
#include "../embedia/fixed32/fixed.h"


// FIX_EXP_MAX = ln(FIX_MAX), recalculate if FIX_FRC_SZ change
#define FIX_EXP_MAX FL2FX(9.7132951013616)



            float measure_error(data2d_t o_real, data2d_t o_pred, float err){
                int i, match;
                for (match=0,i=0; i<o_real.width*o_real.height; i++){
                    //printf("%f   %f\n", FX2FL(o_real.data[i]), FX2FL(o_pred.data[i]) );
                    if (fabs(FX2FL(o_real.data[i])-FX2FL(o_pred.data[i])) <= err){
                        match++;
                    }
                }
                return 100.0*match/(o_real.width*o_real.height);
            }

//data2d_t input = { 1, 10, (fixed[]){ 26146, 35678, 45425, 29331, 95099, 99711, 83365, 54175, 63708, 92940 } };
data2d_t input = { 1, 10, (fixed[]){ 500000, 1250000, 45425, 29331, 95099, 99711, 83365, 54175, -63708, -92940 } };

data2d_t real_output = { 1, 10, (fixed[]){ 72051, 74401, 76780, 72838, 88320, 89327, 85702, -78890, -81157, 0 } };


float expf_custom(float x) {
  float result = 1.0f;
  float term = 1.0f;
  int i;

  for (i = 1; i <= 10; i++) {
    term *= x / i;
    result += term;
  }

  return result;
}

  // Devuelve el valor de exp(a)
    fixed fixed_exp1(fixed fp){
        fixed xabs, k, z, R, xp;
        const fixed LN2 = FLOAT_TO_FIXED(0.69314718055994530942);
        const fixed LN2_INV = FLOAT_TO_FIXED(1.4426950408889634074);
        const fixed EXP_P[5] = {
            FLOAT_TO_FIXED(1.66666666666666019037e-01),
            FLOAT_TO_FIXED(-2.77777777770155933842e-03),
            FLOAT_TO_FIXED(6.61375632143793436117e-05),
            FLOAT_TO_FIXED(-1.65339022054652515390e-06),
            FLOAT_TO_FIXED(4.13813679705723846039e-08),
        };

        if (fp == 0)
            return (FIX_ONE);
        xabs = FIXED_ABS(fp);
        k = FIXED_MUL(xabs, LN2_INV);
        k += FIX_HALF;
        k &= ~FIX_FRC_MSK;
        if (fp < 0)
            k = -k;
        if (k >= FIX_EXP_MAX)
            return FIX_MAX;
        if (k <= -FIX_EXP_MAX)
            return -FIX_MAX;
        fp -= FIXED_MUL(k, LN2);
        z = FIXED_MUL(fp, fp);
        /* Taylor */
        R = FIX_TWO +
            FIXED_MUL(z, EXP_P[0] + FIXED_MUL(z, EXP_P[1] +
            FIXED_MUL(z, EXP_P[2] + FIXED_MUL(z, EXP_P[3] +
            FIXED_MUL(z, EXP_P[4])))));
        xp = FIX_ONE + FIXED_DIV(FIXED_MUL(fp, FIX_TWO), R - fp);
        if (k < 0)
            k = -k;

        k = FIX_ONE << (k >> FIX_FRC_SZ);

        return (FIXED_MUL(k, xp));
    }

fixed fixed_exp_new(fixed x){
  const fixed AUX[11] = {0,FL2FX(1.0/1),FL2FX(1.0/2), FL2FX(1.0/3), FL2FX(1.0/4), FL2FX(1.0/5), FL2FX(1.0/6), FL2FX(1.0/7), FL2FX(1.0/8), FL2FX(1.0/9), FL2FX(1.0/10)};
  fixed result = FIX_ONE;
  fixed term = FIX_ONE;
  int i;

  for (i = 1; i <= 10; i++) {
    term = FIXED_MUL(term, FIXED_MUL(x, AUX[i]));
    result += term;
  }

  return result;
}


 fixed fixed_exp3(fixed fp){
        const fixed AUX[9] = {FL2FX(1.0/2), FL2FX(1.0/3), FL2FX(1.0/4), FL2FX(1.0/5), FL2FX(1.0/6), FL2FX(1.0/7), FL2FX(1.0/8), FL2FX(1.0/9), FL2FX(1.0/10)};

        #define MAX_EXP_IT 8

        if(fp == FIX_ZERO) return FIX_ONE;
        if(fp == FIX_ONE) return FIX_E;
        if(fp >= FIX_EXP_MAX) return FIX_MAX;
        if(fp <= -FIX_EXP_MAX) return FIX_ZERO;

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

        return result;
    }

fixed fixed_exp3_(fixed fp){
        const fixed AUX[11] = {0,0,FL2FX(1.0/2), FL2FX(1.0/3), FL2FX(1.0/4), FL2FX(1.0/5), FL2FX(1.0/6), FL2FX(1.0/7), FL2FX(1.0/8), FL2FX(1.0/9), FL2FX(1.0/10)};

        #define MAX_EXP_IT 10

        if(fp == FIX_ZERO) return FIX_ONE;
        if(fp == FIX_ONE) return FIX_E;
        if(fp >= FIX_EXP_MAX) return FIX_MAX;
        if(fp <= -FIX_EXP_MAX) return FIX_ZERO;

        uint8_t i;
        uint8_t neg = (fp < FIX_ZERO);
        if (neg) fp = -fp;

        fixed result = fp + FIX_ONE;
        fixed term = fp;
        for (i = 2; i <= MAX_EXP_IT; i++){
            //term = FIXED_MUL(FIXED_MUL(term, fp), AUX[i]);
            term = FIXED_MUL(term, FIXED_MUL(fp, AUX[i]));
            result += term;
            if (term < 100)
                break;
        }

        if (neg) result = FIXED_DIV(FIX_ONE, result);

        return result;
    }


double measure_exec_time(int repetitions, int arr_size, fixed arr[], fixed (*func)(fixed)) {
    // Guarda el tiempo de inicio
    clock_t start_time = clock();
    int i,j;

    for (i = 0; i < repetitions; ++i) {
        for (j = 0; j < arr_size; ++j) {
            fixed result = func(arr[j]);
            // Puedes hacer algo con 'result' si es necesario
        }
    }

    // Guarda el tiempo de finalización
    clock_t end_time = clock();

    // Calcula el tiempo transcurrido en segundos
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    return elapsed_time;
}


data2d_t output;

# define ERROR_BOUND 0.0005

#define TIMES 1000000

int main(){

    float a, b, c, d, e,f;
    int i;

    printf("     exp      expf_custom      fixed_exp1      fixed_exp2     fixed_exp3     fixed_exp_new\n");
    for (i=0; i<10; i++){
        a = exp(FX2FL(input.data[i]));
        b = expf_custom(FX2FL(input.data[i]));
        c = FX2FL(fixed_exp1(input.data[i]));
        d = FX2FL(fixed_exp2(input.data[i]));
        e = FX2FL(fixed_exp3(input.data[i]));
        f = FX2FL(fixed_exp_new(input.data[i]));
        printf("%11f   %11f    %11f    %11f    %11f    %11f\n", a, b, c, d, e, f);
    }

    printf("fixed_exp    : %f\n", measure_exec_time(TIMES, 10, input.data, fixed_exp1));
    printf("fixed_exp2   : %f\n", measure_exec_time(TIMES, 10, input.data, fixed_exp2));
    printf("fixed_exp3   : %f\n", measure_exec_time(TIMES, 10, input.data, fixed_exp3));
    printf("fixed_exp_new: %f\n", measure_exec_time(TIMES, 10, input.data, fixed_exp_new));


    //************************ LAYER  0 ***********************//
    output = input;
    sigmoid_activation(output.data, 10);

    printf("Test result: %6.3f %%\n", measure_error(real_output, output, ERROR_BOUND));

    return 0;
}
