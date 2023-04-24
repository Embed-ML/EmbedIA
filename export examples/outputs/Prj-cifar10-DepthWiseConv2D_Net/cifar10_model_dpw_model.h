/* EmbedIA model definition file*/
#ifndef CIFAR10_MODEL_DPW_MODEL_H
#define CIFAR10_MODEL_DPW_MODEL_H

/*

+-----------------------+---------------------+------------+--------------+--------+------------+
| Layer(activation)     | Name                | #Param(NT) |    Shape     |   MACs | Size (KiB) |
+-----------------------+---------------------+------------+--------------+--------+------------+
| DepthwiseConv2D(relu) | depthwise_conv2d_13 |         30 | (16, 16, 3)  |   6912 |     0.152  |
| Conv2D(relu)          | conv2d_12           |        128 | (16, 16, 32) |  24576 |     0.750  |
| MaxPooling2D          | max_pooling2d_12    |          0 |  (8, 8, 32)  |      0 |     0.000  |
| DepthwiseConv2D(relu) | depthwise_conv2d_14 |        320 |  (8, 8, 32)  |  18432 |     1.625  |
| Conv2D(relu)          | conv2d_13           |       2112 |  (8, 8, 64)  | 131072 |     8.750  |
| MaxPooling2D          | max_pooling2d_13    |          0 |  (4, 4, 64)  |      0 |     0.000  |
| Flatten               | flatten_7           |          0 |   (1024,)    |      0 |     0.000  |
| Dense(relu)           | dense_14            |     131200 |    (128,)    | 131072 |   513.000  |
| Dense(softmax)        | dense_15            |       1290 |    (10,)     |   1280 |     5.078  |
+-----------------------+---------------------+------------+--------------+--------+------------+
Total params (NT)....: 135080
Total size in KiB....: 529.355
Total MACs operations: 313344

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
