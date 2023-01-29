/* EmbedIA model */
#ifndef RAIN_PREDICTOR_MODEL_H
#define RAIN_PREDICTOR_MODEL_H

#include "embedia.h"

#define INPUT_LENGTH 11

#define INPUT_SIZE 11


void model_init();

void model_predict(data1d_t input, data1d_t * output);

int model_predict_class(data1d_t input, data1d_t * results);

#endif
