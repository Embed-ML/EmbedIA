#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "fixed.h"



/*
Implements the 5-order polynomial approximation to sin(x).
@param i   angle (with 2^15 units/circle)
@return    16 bit fixed point Sine value (4.12) (ie: +4096 = +1 & -4096 = -1)

The result is accurate to within +- 1 count. ie: +/-2.44e-4.
*/
int16_t fpsin(int16_t i)
{
    /* Convert (signed) input to a value between 0 and 8192. (8192 is pi/2, which is the region of the curve fit). */
    /* ------------------------------------------------------------------- */
    i <<= 1;
    uint8_t c = i<0; //set carry for output pos/neg

    if(i == (i|0x4000)) // flip input value to corresponding value in range [0..8192)
        i = (1<<15) - i;
    i = (i & 0x7FFF) >> 1;
    /* ------------------------------------------------------------------- */

    /* The following section implements the formula:
     = y * 2^-n * ( A1 - 2^(q-p)* y * 2^-n * y * 2^-n * [B1 - 2^-r * y * 2^-n * C1 * y]) * 2^(a-q)
    Where the constants are defined as follows:
    */
    enum {A1=3370945099UL, B1=2746362156UL, C1=292421UL};
    enum {n=13, p=32, q=31, r=3, a=12};

    uint32_t y = (C1*((uint32_t)i))>>n;
    y = B1 - (((uint32_t)i*y)>>r);
    y = (uint32_t)i * (y>>n);
    y = (uint32_t)i * (y>>n);
    y = A1 - (y>>(p-q));
    y = (uint32_t)i * (y>>n);
    y = (y+(1UL<<(q-a-1)))>>(q-a); // Rounding

    return c ? -y : y;
}

//Cos(x) = sin(x + pi/2)
#define fpcos(i) fpsin((int16_t)(((uint16_t)(i)) + 8192U))


fixed fix_sin(int rad){
  //  rad = rad % FIX_PI + FL2FX(180);
 //   rad = FIXED_DIV(rad, FIX_PI)*13;
    return fpsin(rad);
}


#define PI 3.1415926535
//#define FIX_RAD_MAX = FL2FX(PI)
fixed fixed_sin(fixed rad){
    #define FIX_IN_BITS 15
    #define FIX_OUT_BITS 12

    // ensure -PI <= rad <= PI
    if (rad<-FIX_PI || rad > FIX_PI)
        rad = rad % FIX_PI;
    // force 0 <= rad <= 2*PI
    rad = rad + FIX_PI;
    // 0 <= rad <= 1
    rad = FIXED_DIV(rad, 2*FIX_PI);

    //rad = FIXED_MUL(rad, 32768);
    #if FIX_IN_BITS > FIX_FRC_SZ
    rad = rad << (FIX_IN_BITS - FIX_FRC_SZ);
    #else
    rad = rad >> (FIX_FRC_SZ-FIX_IN_BITS);
    #endif // FIX_SIN_BITS

    rad = fpsin(rad);

    #if FIX_OUT_BITS > FIX_FRC_SZ
    return rad >> (FIX_OUT_BITS-FIX_FRC_SZ);
    #else
    return rad << (FIX_FRC_SZ-FIX_OUT_BITS);
    #endif // FIX_OUT_BITS

}


int rad2val(float rad){
    return 32768*(rad+PI)/(2*PI);
}

int main(void)
{
    int32_t max = 0, min = 0;
    float fl = PI;
    int i;

    printf("Ejemplo original adaptado:\n");
    for (i=-180; i<=180; i+=45){
        float rad = PI*i/180; // radians
        printf("sin(%d) = %f\n", i, (fix_sin(rad2val(rad))/4096.0));

    }

    printf("\nEjemplo para fixed 32:\n");
    for (i=180; i>=-180; i-=45){
        float rad = PI*i/180; // radians
        fixed fx_rad  = FL2FX(rad);
        printf("sin(%d) => %f   %f\n", i, FX2FL(fixed_sin(fx_rad)), sin(rad));

    }

    printf("sin(PI) = %f\n", FX2FL(fixed_sin(FIX_PI/2)));
    printf("sin(3*PI) = %f\n", FX2FL(fixed_sin(3*FIX_PI/2)));
    printf("sin(-3*PI) = %f\n", FX2FL(fixed_sin(-3*FIX_PI/2)));


 /*   for(uint16_t i = 0; i <= 32768; ++i)
    {
        int32_t s = lround(4096*sin(2*M_PI * i / 32768));
        int16_t s5d = fpsin(i);
        int32_t err = s - s5d;
        if(err > max)
            max = err;
        if(err < min)
            min = err;
        printf("The value of %i is %i - compare %i, diff : %i\n", i, s5d, s, err);
    }

    printf("min: %i max: %i\n", min, max);
    */


    return 0;
}

