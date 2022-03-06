
/* EmbedIA model - Autogenerado */
#ifndef MNIST_DIGITS_MODEL_H
#define MNIST_DIGITS_MODEL_H

#include "embedia.h"
#define MNIST_DIGITS_MODEL_CHANNELS 1
#define MNIST_DIGITS_MODEL_HEIGHT 8
#define MNIST_DIGITS_MODEL_WIDTH 8
void model_init();

int model_predict(data_t input, flatten_data_t * results);

#endif
