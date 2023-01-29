/* EmbedIA model */
#ifndef PERSON_DETECTION_MODEL_H
#define PERSON_DETECTION_MODEL_H

#include "embedia.h"

#define INPUT_CHANNELS 1
#define INPUT_WIDTH 96
#define INPUT_HEIGHT 96

#define INPUT_SIZE 9216


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
