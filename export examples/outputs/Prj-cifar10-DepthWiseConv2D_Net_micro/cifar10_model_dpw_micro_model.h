/* EmbedIA model definition file*/
#ifndef CIFAR10_MODEL_DPW_MICRO_MODEL_H
#define CIFAR10_MODEL_DPW_MICRO_MODEL_H

/*

+-----------------------+------------------+------------+-----------+------+------------+
| Layer(activation)     | Name             | #Param(NT) |   Shape   | MACs | Size (KiB) |
+-----------------------+------------------+------------+-----------+------+------------+
| NoneType              | NoneType         |          0 | (4, 4, 3) |    0 |     0.000  |
| DepthwiseConv2D(relu) | depthwise_conv2d |         15 | (3, 3, 3) |  108 |     0.094  |
| Conv2D(relu)          | conv2d           |         13 | (2, 2, 1) |   48 |     0.059  |
| Flatten               | flatten          |          0 |    (4,)   |    0 |     0.000  |
| Dense(softmax)        | dense            |         50 |   (10,)   |   40 |     0.234  |
+-----------------------+------------------+------------+-----------+------+------------+
Total params (NT)....: 78
Total size in KiB....: 0.387
Total MACs operations: 196

*/

#include "embedia.h"

#define INPUT_CHANNELS 3
#define INPUT_WIDTH 4
#define INPUT_HEIGHT 4

#define INPUT_SIZE 48


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
