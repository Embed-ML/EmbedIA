/* EmbedIA model definition file*/
#ifndef MNIST_MODEL_DPW_MODEL_H
#define MNIST_MODEL_DPW_MODEL_H

/*

+-----------------------+--------------------+------------+--------------+--------+------------+
| Layer(activation)     | Name               | #Param(NT) |    Shape     |   MACs | Size (KiB) |
+-----------------------+--------------------+------------+--------------+--------+------------+
| DepthwiseConv2D(relu) | depthwise_conv2d   |         10 | (12, 12, 1)  |   1296 |     0.019  |
| Conv2D(relu)          | conv2d             |         64 | (12, 12, 32) |   4608 |     0.312  |
| DepthwiseConv2D(relu) | depthwise_conv2d_1 |        320 | (10, 10, 32) |  28800 |     0.594  |
| Conv2D(relu)          | conv2d_1           |       2112 | (10, 10, 64) | 204800 |     2.562  |
| MaxPooling2D          | max_pooling2d      |          0 |  (5, 5, 64)  |      0 |     0.000  |
| Flatten               | flatten            |          0 |   (1600,)    |      0 |     0.000  |
| Dense(relu)           | dense              |     204928 |    (128,)    | 204800 |   201.250  |
| Dense(softmax)        | dense_1            |       1290 |    (10,)     |   1280 |     1.348  |
+-----------------------+--------------------+------------+--------------+--------+------------+
Total params (NT)....: 208724
Total size in KiB....: 206.085
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
