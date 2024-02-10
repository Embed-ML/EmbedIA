/* EmbedIA model definition file*/
#ifndef EMBEDIA_MODEL_H
#define EMBEDIA_MODEL_H

/*

+--------------------+---------------+------------+------------+------+------------+
| Layer(activation)  | Name          | #Param(NT) |   Shape    | MACs | Size (KiB) |
+--------------------+---------------+------------+------------+------+------------+
| LayerInfo(relu)    | conv2d        |         80 | (6, 6, 8)  | 2592 |     0.375  |
| LayerInfo          | max_pooling2d |          0 | (3, 3, 8)  |    0 |     0.000  |
| LayerInfo(relu)    | conv2d_1      |        528 | (2, 2, 16) | 2048 |     2.188  |
| LayerInfo          | flatten       |          0 |   (64,)    |    0 |     0.000  |
| LayerInfo(relu)    | dense         |       1040 |   (16,)    | 1024 |     4.125  |
| LayerInfo(softmax) | dense_1       |        170 |   (10,)    |  160 |     0.703  |
+--------------------+---------------+------------+------------+------+------------+
Total params (NT)....: 1818
Total size in KiB....: 7.391
Total MACs operations: 5824

*/

#include "embedia.h"

#define INPUT_CHANNELS 1
#define INPUT_WIDTH 8
#define INPUT_HEIGHT 8

#define INPUT_SIZE 64


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
