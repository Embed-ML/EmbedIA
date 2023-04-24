/* EmbedIA model definition file*/
#ifndef MNIST_MODEL_DPW_MODEL_H
#define MNIST_MODEL_DPW_MODEL_H

/*

+-----------------------+--------------------+------------+--------------+--------+------------+
| Layer(activation)     | Name               | #Param(NT) |    Shape     |   MACs | Size (KiB) |
+-----------------------+--------------------+------------+--------------+--------+------------+
| DepthwiseConv2D(relu) | depthwise_conv2d   |         10 | (12, 12, 1)  |   1296 |     0.051  |
| Conv2D(relu)          | conv2d             |         64 | (12, 12, 32) |   4608 |     0.500  |
| DepthwiseConv2D(relu) | depthwise_conv2d_1 |        320 | (10, 10, 32) |  28800 |     1.625  |
| Conv2D(relu)          | conv2d_1           |       2112 | (10, 10, 64) | 204800 |     8.750  |
| MaxPooling2D          | max_pooling2d      |          0 |  (5, 5, 64)  |      0 |     0.000  |
| Flatten               | flatten            |          0 |   (1600,)    |      0 |     0.000  |
| Dense(relu)           | dense              |     204928 |    (128,)    | 204800 |   801.000  |
| Dense(softmax)        | dense_1            |       1290 |    (10,)     |   1280 |     5.078  |
+-----------------------+--------------------+------------+--------------+--------+------------+
Total params (NT)....: 208724
Total size in KiB....: 817.004
Total MACs operations: 445584

*/

#include "embedia.h"

#define INPUT_CHANNELS 1
#define INPUT_WIDTH 14
#define INPUT_HEIGHT 14

#define INPUT_SIZE 196


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
