/* EmbedIA model definition file*/
#ifndef LAYERSTEST01_MODEL_H
#define LAYERSTEST01_MODEL_H

/*

+-----------------+----------------------+------------+-------------+-------+------------+
| EmbedIA Layer   | Name                 | #Param(NT) |    Shape    |  MACs | Size (KiB) |
+-----------------+----------------------+------------+-------------+-------+------------+
| Conv2D          | conv2_d_1            |          0 | (27, 27, 8) | 23328 |     0.070  |
| Activation      | conv2_d_11           |          0 | (27, 27, 8) |     0 |     0.000  |
| DepthwiseConv2D | depthwise_conv2_d_1  |          0 | (10, 10, 8) | 51200 |     0.539  |
| Activation      | depthwise_conv2_d_11 |          0 | (10, 10, 8) |     0 |     0.000  |
| Pooling         | max_pool_1           |          0 |  (5, 5, 8)  |     0 |     0.000  |
| Conv2D          | conv2_d_2            |          0 |  (4, 4, 12) |  6144 |     0.457  |
| Activation      | conv2_d_21           |          0 |  (4, 4, 12) |     0 |     0.000  |
| Pooling         | avg_pool_2           |          0 |  (2, 2, 12) |     0 |     0.000  |
| Activation      | re_l_u_2             |          0 |  (2, 2, 12) |     0 |     0.000  |
| SeparableConv2D | sep_conv2_d_3        |          0 |  (1, 1, 15) |   285 |     0.281  |
| Activation      | sep_conv2_d_31       |          0 |  (1, 1, 15) |     0 |     0.000  |
| Flatten         | flatten_4            |          0 |    (15,)    |     0 |     0.000  |
| Dense           | dense_4              |          0 |    (32,)    |   480 |     0.594  |
| Activation      | dense_41             |          0 |    (32,)    |     0 |     0.000  |
| Dense           | dense_5              |          0 |    (10,)    |   320 |     0.352  |
| Activation      | dense_51             |          0 |    (10,)    |     0 |     0.000  |
| Activation      | soft_max_6           |          0 |    (10,)    |     0 |     0.000  |
+-----------------+----------------------+------------+-------------+-------+------------+
Total params (NT)....: 0
Total size in KiB....: 2.293
Total MACs operations: 81757

*/

#include "embedia.h"

#define INPUT_CHANNELS 1
#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28

#define INPUT_SIZE 784


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
