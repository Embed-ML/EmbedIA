/* EmbedIA model definition file*/
#ifndef CIFAR10_MODEL_DPW_MODEL_H
#define CIFAR10_MODEL_DPW_MODEL_H

/*

+-----------------------+--------------------+------------+--------------+--------+------------+
| Layer(activation)     | Name               | #Param(NT) |    Shape     |   MACs | Size (KiB) |
+-----------------------+--------------------+------------+--------------+--------+------------+
| Conv2D(relu)          | conv2d             |        448 | (14, 14, 16) |  84672 |     1.875  |
| DepthwiseConv2D(relu) | depthwise_conv2d   |        160 | (12, 12, 16) |  20736 |     0.812  |
| Conv2D(relu)          | conv2d_1           |       4640 | (10, 10, 32) | 460800 |    18.375  |
| DepthwiseConv2D(relu) | depthwise_conv2d_1 |        320 |  (8, 8, 32)  |  18432 |     1.625  |
| Flatten               | flatten            |          0 |   (2048,)    |      0 |     0.000  |
| Dense(relu)           | dense              |     131136 |    (64,)     | 131072 |   512.500  |
| Dense(softmax)        | dense_1            |        650 |    (10,)     |    640 |     2.578  |
+-----------------------+--------------------+------------+--------------+--------+------------+
Total params (NT)....: 137354
Total size in KiB....: 537.766
Total MACs operations: 716352

*/

#include "embedia.h"

#define INPUT_CHANNELS 3
#define INPUT_WIDTH 16
#define INPUT_HEIGHT 16

#define INPUT_SIZE 768


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
