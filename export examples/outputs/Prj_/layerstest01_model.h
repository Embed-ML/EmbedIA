/* EmbedIA model definition file*/
#ifndef _LAYERSTEST01_MODEL_H_H
#define _LAYERSTEST01_MODEL_H_H

/*

+--------------------+----------------+------------+--------------+--------+------------+
| EmbedIA Layer      | Name           | #Param(NT) |    Shape     |   MACs | Size (KiB) |
+--------------------+----------------+------------+--------------+--------+------------+
| DummyLayer         | random_flip    |          0 | (28, 28, 1)  |      0 |     0.000  |
| DummyLayer         | random_flip_1  |          0 | (28, 28, 1)  |      0 |     0.000  |
| Conv2D             | conv2_d_1      |          0 | (26, 26, 8)  |  48672 |     0.352  |
| Activation         | conv2_d_11     |          0 | (26, 26, 8)  |      0 |     0.000  |
| Pooling            | max_pool_1     |          0 | (13, 13, 8)  |      0 |     0.000  |
| Activation         | leaky_re_l_u_1 |          0 | (13, 13, 8)  |      0 |     0.000  |
| BatchNormalization | batch_1        |          0 | (13, 13, 8)  |      0 |     0.074  |
| Conv2D             | conv2_d_2      |          0 | (11, 11, 16) | 139392 |     4.848  |
| Activation         | conv2_d_21     |          0 | (11, 11, 16) |      0 |     0.000  |
| Pooling            | avg_pool_2     |          0 |  (5, 5, 16)  |      0 |     0.000  |
| Activation         | re_l_u_2       |          0 |  (5, 5, 16)  |      0 |     0.000  |
| SeparableConv2D    | sep_conv2_d_3  |          0 |  (3, 3, 16)  |   3600 |     1.688  |
| Activation         | sep_conv2_d_31 |          0 |  (3, 3, 16)  |      0 |     0.000  |
| Pooling            | avg_pool_3     |          0 |  (1, 1, 16)  |      0 |     0.000  |
| Flatten            | flatten_4      |          0 |    (16,)     |      0 |     0.000  |
| Dense              | dense_4        |          0 |    (32,)     |    512 |     2.125  |
| Activation         | dense_41       |          0 |    (32,)     |      0 |     0.000  |
| Dense              | dense_5        |          0 |     (8,)     |    256 |     1.031  |
| Activation         | dense_51       |          0 |     (8,)     |      0 |     0.000  |
| Activation         | soft_max_6     |          0 |     (8,)     |      0 |     0.000  |
+--------------------+----------------+------------+--------------+--------+------------+
Total params (NT)....: 0
Total size in KiB....: 10.117
Total MACs operations: 192432

*/

#include "common.h"

#define INPUT_CHANNELS 1
#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28

#define INPUT_SIZE 784


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
