/* EmbedIA model definition file*/
#ifndef MNIST_MODEL_H
#define MNIST_MODEL_H

/*

+-------------------+-------------------+------------+-------------+-------+------------+
| Layer(activation) | Name              | #Param(NT) |    Shape    |  MACs | Size (KiB) |
+-------------------+-------------------+------------+-------------+-------+------------+
| Conv2D(linear)    | conv_2d_1         |        208 | (10, 10, 8) | 20000 |     0.875  |
| LeakyReLU         | leaky_relu_1      |          0 | (10, 10, 8) |     0 |     0.000  |
| MaxPooling2D      | max_pool_2d_1     |          0 |  (5, 5, 8)  |     0 |     0.000  |
| Conv2D(linear)    | conv_2d_2         |       1168 |  (3, 3, 16) | 10368 |     4.688  |
| LeakyReLU         | leaky_relu_2      |          0 |  (3, 3, 16) |     0 |     0.000  |
| MaxPooling2D      | max_pool_2d_2     |          0 |  (1, 1, 16) |     0 |     0.000  |
| Flatten           | reshape_1         |          0 |    (16,)    |     0 |     0.000  |
| Dense(linear)     | fully_connected_1 |        408 |    (24,)    |   384 |     1.688  |
| LeakyReLU         | leaky_relu_3      |          0 |    (24,)    |     0 |     0.000  |
| Dense(linear)     | fully_connected_2 |        250 |    (10,)    |   240 |     1.016  |
+-------------------+-------------------+------------+-------------+-------+------------+
Total params (NT)....: 2034
Total size in KiB....: 8.266
Total MACs operations: 30992

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
