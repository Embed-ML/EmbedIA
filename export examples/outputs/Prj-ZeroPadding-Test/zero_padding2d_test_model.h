/* EmbedIA model definition file*/
#ifndef ZERO_PADDING2D_TEST_MODEL_H
#define ZERO_PADDING2D_TEST_MODEL_H

/*

+-------------------+----------------+------------+-----------+------+------------+
| Layer(activation) | Name           | #Param(NT) |   Shape   | MACs | Size (KiB) |
+-------------------+----------------+------------+-----------+------+------------+
| NoneType          | NoneType       |          0 | (3, 3, 3) |    0 |     0.000  |
| ZeroPadding2D     | zero_padding2d |          0 | (7, 5, 3) |    0 |     0.000  |
+-------------------+----------------+------------+-----------+------+------------+
Total params (NT)....: 0
Total size in KiB....: 0.000
Total MACs operations: 0

*/

#include "embedia.h"

#define INPUT_CHANNELS 3
#define INPUT_WIDTH 3
#define INPUT_HEIGHT 3

#define INPUT_SIZE 27


void model_init();

void model_predict(data3d_t input, data3d_t * output);

int model_predict_class(data3d_t input, data3d_t * results);

#endif
