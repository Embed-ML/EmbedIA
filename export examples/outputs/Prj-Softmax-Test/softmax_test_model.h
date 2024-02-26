/* EmbedIA model definition file*/
#ifndef SOFTMAX_TEST_MODEL_H
#define SOFTMAX_TEST_MODEL_H

/*

+-------------------+---------+------------+----------+------+------------+
| Layer(activation) | Name    | #Param(NT) |  Shape   | MACs | Size (KiB) |
+-------------------+---------+------------+----------+------+------------+
| Dense(linear)     | dense   |         20 | (10, 10) |   10 |     0.107  |
| Softmax           | softmax |          0 | (10, 10) |    0 |     0.000  |
+-------------------+---------+------------+----------+------+------------+
Total params (NT)....: 20
Total size in KiB....: 0.107
Total MACs operations: 10

*/

#include "embedia.h"

#define INPUT_WIDTH 1
#define INPUT_HEIGHT 10

#define INPUT_SIZE 10


void model_init();

void model_predict(data1d_t input, data2d_t * output);

int model_predict_class(data1d_t input, data2d_t * results);

#endif
