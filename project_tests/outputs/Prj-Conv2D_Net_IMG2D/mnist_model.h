/* EmbedIA model */
#ifndef MNIST_MODEL_H
#define MNIST_MODEL_H

#include "embedia.h"

#define INPUT_CHANNELS 2
#define INPUT_WIDTH 14
#define INPUT_HEIGHT 14

#define INPUT_SIZE 392


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
