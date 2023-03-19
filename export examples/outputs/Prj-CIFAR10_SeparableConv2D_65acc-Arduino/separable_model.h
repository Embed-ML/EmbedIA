/* EmbedIA model definition file*/
#ifndef SEPARABLE_MODEL_H
#define SEPARABLE_MODEL_H

/*

+-------------------------+-------------------------+------------+--------------+---------+------------+
| Layer(activation)       | Name                    | #Param(NT) |    Shape     |    MACs | Size (KiB) |
+-------------------------+-------------------------+------------+--------------+---------+------------+
| Conv2D(linear)          | conv2d_5                |        448 | (30, 30, 16) |  388800 |     1.875  |
| MaxPooling2D            | max_pooling2d_132       |          0 | (15, 15, 16) |       0 |     0.000  |
| BatchNormalization      | batch_normalization_398 |     64(32) | (15, 15, 16) |       0 |     0.137  |
| Activation(tanh)        | activation_406          |          0 | (15, 15, 16) |       0 |     0.000  |
| SeparableConv2D(linear) | separable_conv2d_9      |       1232 | (13, 13, 64) |  789568 |     4.812  |
| BatchNormalization      | batch_normalization_399 |   256(128) | (13, 13, 64) |       0 |     0.512  |
| Activation(tanh)        | activation_407          |          0 | (13, 13, 64) |       0 |     0.000  |
| SeparableConv2D(linear) | separable_conv2d_10     |       6816 | (11, 11, 96) | 1219680 |    27.250  |
| MaxPooling2D            | max_pooling2d_133       |          0 |  (5, 5, 96)  |       0 |     0.000  |
| BatchNormalization      | batch_normalization_400 |   384(192) |  (5, 5, 96)  |       0 |     0.762  |
| Activation(tanh)        | activation_408          |          0 |  (5, 5, 96)  |       0 |     0.000  |
| SeparableConv2D(linear) | separable_conv2d_11     |      50528 | (3, 3, 512)  | 2400768 |   196.875  |
| AveragePooling2D        | average_pooling2d_20    |          0 | (1, 1, 512)  |       0 |     0.000  |
| BatchNormalization      | batch_normalization_401 | 2048(1024) | (1, 1, 512)  |       0 |     4.012  |
| Activation(tanh)        | activation_409          |          0 | (1, 1, 512)  |       0 |     0.000  |
| Flatten                 | flatten_82              |          0 |    (512,)    |       0 |     0.000  |
| Dropout                 | dropout_82              |          0 |    (512,)    |       0 |     0.000  |
| Dense(linear)           | dense_81                |       5130 |    (10,)     |    5120 |    20.078  |
| Activation(softmax)     | activation_410          |          0 |    (10,)     |       0 |     0.000  |
+-------------------------+-------------------------+------------+--------------+---------+------------+
Total params (NT)....: 66906(1376)
Total size in KiB....: 256.312
Total MACs operations: 4803936

*/

#include "embedia.h"

#define INPUT_CHANNELS 3
#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32

#define INPUT_SIZE 3072


void model_init();

void model_predict(data3d_t input, data1d_t * output);

int model_predict_class(data3d_t input, data1d_t * results);

#endif
