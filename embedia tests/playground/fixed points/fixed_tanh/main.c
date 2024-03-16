#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "fixed.h"


float tanh_fl(float v)
{
    const float c1 = 0.03138777F;
    const float c2 = 0.276281267F;
    const float c_log2f = 1.442695022F;
    v *= c_log2f;
    int intPart = (int)v;
    float x = (v - intPart);
    float xx = x * x;
    float v1 = c_log2f + c2 * xx;
    float v2 = x + xx * c1 * x;
    float v3 = (v2 + v1);
    *((int*)&v3) += intPart << 24;
    float v4 = v2 - v1;
    return (v3 + v4) / (v3 - v4);
}

// https://stackoverflow.com/questions/73770905/best-non-trigonometric-floating-point-approximation-of-tanhx-in-10-instruction

/* min(max(0.1073*x*(1+25.125/(x*x+3.125),-1,1)

           */
#define MAX(a,b) (a>b) ? a : b
#define MIN(a,b) (a<b) ? a : b

float tanh_fx(fixed x)
{
    fixed v = FIXED_MUL(FL2FX(0.1073), x);
    fixed w = FIXED_MUL(x,x)+FL2FX(3.125);
    w = 1+FIXED_DIV(FL2FX(25.125), w);
    v = FIXED_MUL(v,w);
    return MIN(MAX(v,-FIX_ONE), FIX_ONE);
}

int main(void)
{
    float i;

    for (i=-3; i<=3; i+=0.5){
        float fl_orig = tanh(i);
        float fl_1 = tanh_fl(i);
        float fx_1 = FX2FL(fixed_tanh(FL2FX(i)));
        float fx_2 = FX2FL(tanh_fx(FL2FX(i)));

        printf("%2.8f\n%2.8f\n%2.8f\n%2.8f\n----------\n", fl_orig, fl_1, fx_1, fx_2);
    }

    return 0;
}

