
/* EmbedIA model - Autogenerado */
#ifndef EMBEDIA_MODEL_H
#define EMBEDIA_MODEL_H

#include "embedia.h"
#define EMBEDIA_MODEL_CHANNELS 1
#define EMBEDIA_MODEL_HEIGHT 8
#define EMBEDIA_MODEL_WIDTH 8
void model_init();

int model_predict(data_t input, flatten_data_t * results);

#endif
