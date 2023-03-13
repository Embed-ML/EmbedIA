/* EmbedIA model */
#ifndef SEPARABLE_MODEL_H
#define SEPARABLE_MODEL_H

#include "embedia.h"

#define INPUT_CHANNELS 3
#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32

#define INPUT_SIZE 3072


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
