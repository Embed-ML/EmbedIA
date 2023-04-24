/* EmbedIA model definition file*/
#ifndef FASHIONMNIST_MODEL_H
#define FASHIONMNIST_MODEL_H

/*

+--------------------+------------------------+------------+--------------+---------+------------+
| Layer(activation)  | Name                   | #Param(NT) |    Shape     |    MACs | Size (KiB) |
+--------------------+------------------------+------------+--------------+---------+------------+
| Conv2D(relu)       | conv2d_3               |        160 | (26, 26, 16) |   97344 |     0.750  |
| MaxPooling2D       | max_pooling2d_4        |          0 | (13, 13, 16) |       0 |     0.000  |
| BatchNormalization | batch_normalization_8  |     64(32) | (13, 13, 16) |       0 |     0.137  |
| Conv2D(relu)       | conv2d_4               |       9280 | (11, 11, 64) | 1115136 |    36.750  |
| MaxPooling2D       | max_pooling2d_5        |          0 |  (5, 5, 64)  |       0 |     0.000  |
| BatchNormalization | batch_normalization_9  |   256(128) |  (5, 5, 64)  |       0 |     0.512  |
| Conv2D(relu)       | conv2d_5               |      36928 |  (3, 3, 64)  |  331776 |   144.750  |
| BatchNormalization | batch_normalization_10 |   256(128) |  (3, 3, 64)  |       0 |     0.512  |
| Flatten            | flatten_2              |          0 |    (576,)    |       0 |     0.000  |
| Dropout            | dropout_2              |          0 |    (576,)    |       0 |     0.000  |
| Dense(relu)        | dense_3                |      36928 |    (64,)     |   36864 |   144.500  |
| BatchNormalization | batch_normalization_11 |   256(128) |    (64,)     |       0 |     0.512  |
| Dense(softmax)     | dense_4                |        650 |    (10,)     |     640 |     2.578  |
+--------------------+------------------------+------------+--------------+---------+------------+
Total params (NT)....: 84778(416)
Total size in KiB....: 331.000
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
