/* EmbedIA model definition file*/
#ifndef MNIST_MODEL_H
#define MNIST_MODEL_H

/*

+-------------------+---------------+------------+-------------+-------+------------+
| Layer(activation) | Name          | #Param(NT) |    Shape    |  MACs | Size (KiB) |
+-------------------+---------------+------------+-------------+-------+------------+
| Conv2D(relu)      | conv2d        |         40 | (13, 13, 8) |  5408 |     0.219  |
| Conv2D(relu)      | conv2d_1      |        264 | (12, 12, 8) | 36864 |     1.094  |
| MaxPooling2D      | max_pooling2d |          0 |  (3, 3, 8)  |     0 |     0.000  |
| Flatten           | flatten       |          0 |    (72,)    |     0 |     0.000  |
| Dense(relu)       | dense         |       1168 |    (16,)    |  1152 |     4.625  |
| Dense(softmax)    | dense_1       |        170 |    (10,)    |   160 |     0.703  |
+-------------------+---------------+------------+-------------+-------+------------+
Total params (NT)....: 1642
Total size in KiB....: 6.641
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
