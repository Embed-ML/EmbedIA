#ifndef _LOGISTIC_REGRESSION_H
#define _LOGISTIC_REGRESSION_H

#include "common.h" // Para data1d_t
#include <stdint.h> // Para uint16_t

typedef struct 
{
  uint16_t n_features;
  uint16_t n_classes;
  float *weights; //puntero a los pesos
  float *bias; //puntero a las bias
  float *classes;
} logistic_regression_layer_t;

void logistic_regression_layer(logistic_regression_layer_t lr, data1d_t  input, data1d_t * output);

#endif

