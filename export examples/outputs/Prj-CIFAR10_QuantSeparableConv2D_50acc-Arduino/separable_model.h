/* EmbedIA model definition file*/
#ifndef SEPARABLE_MODEL_H
#define SEPARABLE_MODEL_H

/*

+------------------------------+----------------------------+------------+--------------+---------+------------+
| Layer(activation)            | Name                       | #Param(NT) |    Shape     |    MACs | Size (KiB) |
+------------------------------+----------------------------+------------+--------------+---------+------------+
| Conv2D(linear)               | conv2d_4                   |        448 | (30, 30, 16) |  388800 |     1.875  |
| MaxPooling2D                 | max_pooling2d_130          |          0 | (15, 15, 16) |       0 |     0.000  |
| BatchNormalization           | batch_normalization_394    |     64(32) | (15, 15, 16) |       0 |     0.137  |
| Activation(tanh)             | activation_401             |          0 | (15, 15, 16) |       0 |     0.000  |
| QuantSeparableConv2D(linear) | quant_separable_conv2d_300 |       1232 | (13, 13, 64) |  789568 |     2.219  |
| BatchNormalization           | batch_normalization_395    |   256(128) | (13, 13, 64) |       0 |     0.512  |
| Activation(tanh)             | activation_402             |          0 | (13, 13, 64) |       0 |     0.000  |
| QuantSeparableConv2D(linear) | quant_separable_conv2d_301 |       6816 | (11, 11, 96) | 1219680 |    48.832  |
| MaxPooling2D                 | max_pooling2d_131          |          0 |  (5, 5, 96)  |       0 |     0.000  |
| BatchNormalization           | batch_normalization_396    |   384(192) |  (5, 5, 96)  |       0 |     0.762  |
| Activation(tanh)             | activation_403             |          0 |  (5, 5, 96)  |       0 |     0.000  |
| QuantSeparableConv2D(linear) | quant_separable_conv2d_302 |      50528 | (3, 3, 512)  | 2400768 |   577.242  |
| AveragePooling2D             | average_pooling2d_19       |          0 | (1, 1, 512)  |       0 |     0.000  |
| BatchNormalization           | batch_normalization_397    | 2048(1024) | (1, 1, 512)  |       0 |     4.012  |
| Activation(tanh)             | activation_404             |          0 | (1, 1, 512)  |       0 |     0.000  |
| Flatten                      | flatten_81                 |          0 |    (512,)    |       0 |     0.000  |
| Dropout                      | dropout_81                 |          0 |    (512,)    |       0 |     0.000  |
| Dense(linear)                | dense_80                   |       5130 |    (10,)     |    5120 |    20.078  |
| Activation(softmax)          | activation_405             |          0 |    (10,)     |       0 |     0.000  |
+------------------------------+----------------------------+------------+--------------+---------+------------+
Total params (NT)....: 66906(1376)
Total size in KiB....: 655.668
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
