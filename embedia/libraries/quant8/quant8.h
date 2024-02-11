#ifndef QUANT8_H
#define QUANT8_H

#include <stdint.h>

typedef uint8_t quant8;

typedef struct{
    float sc;
    uint8_t zp;
} qparam_t;


#define Q_MAX 255


//////////////////////////////////// Constantes ////////////////////////////////////

//#define FIX_HALF (FIX_ONE >> 1)
//#define FIX_ZERO 0
//#define FIX_ONE ((fixed)((fixed)1 << FIX_FRC_SZ))
//#define FIX_TWO (FIX_ONE + FIX_ONE)
//#define FIX_E  FLOAT_TO_FIXED(2.7182818284590452354)
//// FIX_EXP_MAX = ln(FIX_MAX), recalculate if FIX_FRC_SZ change
//#define FIX_EXP_MAX FLOAT_TO_FIXED(2.77258872224)
//#define FIX_PI FLOAT_TO_FIXED(3.14159265358979323846)
//#define FIX_MAX (fixed)(((dfixed)1 << (FIX_SIZE-1)) -1)
//#define FIX_MIN (-FIX_MAX)
//#define DFIX_MAX ((dfixed)FIX_MAX << FIX_FRC_SZ)
//#define DFIX_MIN (-DFIX_MAX)

//////////////////////////////////// Macros de conversion ////////////////////////////////////

//#define FIXED_TO_DOUBLE(F) ((double) ((F)*((double)(1)/(double)(1L << FIX_FRC_SZ))))
//#define FIXED_TO_FLOAT(F) ((float) ((F)*((float)(1)/(float)(1L << FIX_FRC_SZ))))
//#define FIXED_TO_INT(F) ((fixed)(F) >> FIX_FRC_SZ)
//#define FIXED_FRAC(F) ( (fixed)(F) & FIX_FRC_MSK )
//#define FIXED_INT(F) ( (fixed)(F) & ~FIX_FRC_MSK )
//#define FLOAT_TO_FIXED(F) ((fixed)((F) * FIX_ONE + ((F) >= 0 ? 0.5 : -0.5)))
//#define INT_TO_FIXED(I) ((fixed)(I) << FIX_FRC_SZ)




void quantize_param(float* values, int size, qparam_t* qp);

void quantize_vec(float values[], quant8 qvalues[], int size, qparam_t qp);

void dequantize_vec(quant8 qvalues[], float values[], int size, qparam_t qp);


float mul_add_vec(quant8 a[], qparam_t qa, quant8 b[], qparam_t qb, int size);


#define Q_CUT(qv) ( (qv > Q_MAX)? Q_MAX : ( (qv < 0) ? 0 : (quant8)qv ) )

#define QUANTIZE(value, qp) Q_CUT( (int)((qp.zp + value) / qp.sc) )
#define DEQUANTIZE(qvalue, qp) ( (float)( qp.sc * (qvalue - qp.zp) ) )


//////////////////////////////////// Macros aritmeticas ////////////////////////////////////

#define _ADD(A, QA, B, QB) (QA.sc*(A-QA.zc) + QB.sc*(B-QB.zc))




#endif
