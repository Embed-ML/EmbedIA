/* EmbedIA model definition file*/
#ifndef MNIST_MODEL_H
#define MNIST_MODEL_H

/*

+--------------------+-----------------------+------------+-------------+-------+------------+
| EmbedIA Layer      | Name                  | #Param(NT) |    Shape    |  MACs | Size (KiB) |
+--------------------+-----------------------+------------+-------------+-------+------------+
| Conv2D             | conv2d_4              |          0 | (13, 13, 8) |  5408 |     0.070  |
| Activation         | conv2d_41             |          0 | (13, 13, 8) |     0 |     0.000  |
| Conv2D             | conv2d_5              |          0 | (12, 12, 8) | 36864 |     0.289  |
| Activation         | conv2d_51             |          0 | (12, 12, 8) |     0 |     0.000  |
| BatchNormalization | batch_normalization   |          0 | (12, 12, 8) |     0 |     0.027  |
| Pooling            | max_pooling2d         |          0 |  (3, 3, 8)  |     0 |     0.000  |
| Flatten            | flatten               |          0 |    (72,)    |     0 |     0.000  |
| BatchNormalization | batch_normalization_1 |          0 |    (72,)    |     0 |     0.152  |
| Dense              | dense                 |          0 |    (16,)    |  1152 |     1.203  |
| Activation         | dense1                |          0 |    (16,)    |     0 |     0.000  |
| Dense              | dense_1               |          0 |    (10,)    |   160 |     0.205  |
| Activation         | dense_11              |          0 |    (10,)    |     0 |     0.000  |
+--------------------+-----------------------+------------+-------------+-------+------------+
Total params (NT)....: 0
Total size in KiB....: 1.947
Total MACs operations: 43584

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
