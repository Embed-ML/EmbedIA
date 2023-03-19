/* EmbedIA model definition file*/
#ifndef FASHIONMNIST_MODEL_H
#define FASHIONMNIST_MODEL_H

/*

+--------------------+------------------------+------------+--------------+---------+------------+
| Layer(activation)  | Name                   | #Param(NT) |    Shape     |    MACs | Size (KiB) |
+--------------------+------------------------+------------+--------------+---------+------------+
| QuantConv2D(relu)  | quant_conv2d_3         |        160 | (26, 26, 16) |   97344 |     0.219  |
| MaxPooling2D       | max_pooling2d_6        |          0 | (13, 13, 16) |       0 |     0.000  |
| BatchNormalization | batch_normalization_12 |     64(32) | (13, 13, 16) |       0 |     0.074  |
| QuantConv2D(relu)  | quant_conv2d_4         |       9280 | (11, 11, 64) | 1115136 |     1.875  |
| MaxPooling2D       | max_pooling2d_7        |          0 |  (5, 5, 64)  |       0 |     0.000  |
| BatchNormalization | batch_normalization_13 |   256(128) |  (5, 5, 64)  |       0 |     0.262  |
| QuantConv2D(relu)  | quant_conv2d_5         |      36928 |  (3, 3, 64)  |  331776 |     5.125  |
| BatchNormalization | batch_normalization_14 |   256(128) |  (3, 3, 64)  |       0 |     0.262  |
| Flatten            | flatten_3              |          0 |    (576,)    |       0 |     0.000  |
| Dropout            | dropout_3              |          0 |    (576,)    |       0 |     0.000  |
| QuantDense(relu)   | quant_dense_1          |      36928 |    (64,)     |   36864 |     4.875  |
| BatchNormalization | batch_normalization_15 |   256(128) |    (64,)     |       0 |     0.262  |
| Dense(softmax)     | dense_5                |        650 |    (10,)     |     640 |     1.309  |
+--------------------+------------------------+------------+--------------+---------+------------+
Total params (NT)....: 84778(416)
Total size in KiB....: 14.262
Total MACs operations: 1581760

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
