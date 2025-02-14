/* EmbedIA model definition file*/
#ifndef _CIFAR10_MODEL_H_H
#define _CIFAR10_MODEL_H_H

/*

+--------------------+-----------------------+------------+--------------+---------+------------+
| EmbedIA Layer      | Name                  | #Param(NT) |    Shape     |    MACs | Size (KiB) |
+--------------------+-----------------------+------------+--------------+---------+------------+
| ChannelsAdapter    | channels_adapter      |          0 | (32, 32, 3)  |       0 |     0.000  |
| Conv2D             | conv2d_4              |          0 | (30, 30, 16) |  388800 |     0.515  |
| Activation         | conv2d_41             |          0 | (30, 30, 16) |       0 |     0.000  |
| Conv2D             | conv2d_5              |          0 | (28, 28, 16) | 1806336 |     2.457  |
| Activation         | conv2d_51             |          0 | (28, 28, 16) |       0 |     0.000  |
| BatchNormalization | batch_normalization_4 |          0 | (28, 28, 16) |       0 |     0.043  |
| Pooling            | max_pooling2d_3       |          0 | (14, 14, 16) |       0 |     0.000  |
| Conv2D             | conv2d_6              |          0 | (12, 12, 32) |  663552 |     4.770  |
| Activation         | conv2d_61             |          0 | (12, 12, 32) |       0 |     0.000  |
| BatchNormalization | batch_normalization_5 |          0 | (12, 12, 32) |       0 |     0.074  |
| Pooling            | max_pooling2d_4       |          0 |  (6, 6, 32)  |       0 |     0.000  |
| Conv2D             | conv2d_7              |          0 |  (4, 4, 64)  |  294912 |    18.535  |
| Activation         | conv2d_71             |          0 |  (4, 4, 64)  |       0 |     0.000  |
| BatchNormalization | batch_normalization_6 |          0 |  (4, 4, 64)  |       0 |     0.137  |
| Pooling            | max_pooling2d_5       |          0 |  (2, 2, 64)  |       0 |     0.000  |
| Flatten            | flatten_1             |          0 |    (256,)    |       0 |     0.000  |
| Dense              | dense_2               |          0 |    (100,)    |   25600 |    25.391  |
| Activation         | dense_21              |          0 |    (100,)    |       0 |     0.000  |
| BatchNormalization | batch_normalization_7 |          0 |    (100,)    |       0 |     0.207  |
| Dense              | dense_3               |          0 |    (10,)     |    1000 |     1.016  |
| Activation         | dense_31              |          0 |    (10,)     |       0 |     0.000  |
+--------------------+-----------------------+------------+--------------+---------+------------+
Total params (NT)....: 0
Total size in KiB....: 53.144
Total MACs operations: 3180200

*/

#include "common.h"

#define INPUT_CHANNELS 3
#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32

#define INPUT_SIZE 3072


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
