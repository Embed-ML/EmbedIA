/* EmbedIA model definition file*/
#ifndef QUANTSEPARABLECONV2DTEST_MODEL_H
#define QUANTSEPARABLECONV2DTEST_MODEL_H

/*

+----------------------+---------------------------------------+------------+--------------+---------+------------+
| EmbedIA Layer        | Name                                  | #Param(NT) |    Shape     |    MACs | Size (KiB) |
+----------------------+---------------------------------------+------------+--------------+---------+------------+
| ChannelsAdapter      | channels_adapter                      |          0 | (32, 32, 3)  |       0 |     0.000  |
| Conv2D               | conv2d_4                              |          0 | (30, 30, 16) |  388800 |     1.938  |
| Activation           | conv2d_41                             |          0 | (30, 30, 16) |       0 |     0.000  |
| Pooling              | max_pooling2d_130                     |          0 | (15, 15, 16) |       0 |     0.000  |
| BatchNormalization   | batch_normalization_394               |          0 | (15, 15, 16) |       0 |     0.137  |
| Activation           | activation_401                        |          0 | (15, 15, 16) |       0 |     0.000  |
| QuantSeparableConv2D | larq_quant_separable_conv2_d_wrapper  |          0 | (13, 13, 64) |  789568 |     2.219  |
| Activation           | quant_separable_conv2d_300            |          0 | (13, 13, 64) |       0 |     0.000  |
| BatchNormalization   | batch_normalization_395               |          0 | (13, 13, 64) |       0 |     0.512  |
| Activation           | activation_402                        |          0 | (13, 13, 64) |       0 |     0.000  |
| QuantSeparableConv2D | larq_quant_separable_conv2_d_wrapper1 |          0 | (11, 11, 96) | 1219680 |    48.832  |
| Activation           | quant_separable_conv2d_301            |          0 | (11, 11, 96) |       0 |     0.000  |
| Pooling              | max_pooling2d_131                     |          0 |  (5, 5, 96)  |       0 |     0.000  |
| BatchNormalization   | batch_normalization_396               |          0 |  (5, 5, 96)  |       0 |     0.762  |
| Activation           | activation_403                        |          0 |  (5, 5, 96)  |       0 |     0.000  |
| QuantSeparableConv2D | larq_quant_separable_conv2_d_wrapper2 |          0 | (3, 3, 512)  | 2400768 |   577.242  |
| Activation           | quant_separable_conv2d_302            |          0 | (3, 3, 512)  |       0 |     0.000  |
| Pooling              | average_pooling2d_19                  |          0 | (1, 1, 512)  |       0 |     0.000  |
| BatchNormalization   | batch_normalization_397               |          0 | (1, 1, 512)  |       0 |     4.012  |
| Activation           | activation_404                        |          0 | (1, 1, 512)  |       0 |     0.000  |
| Flatten              | flatten_81                            |          0 |    (512,)    |       0 |     0.000  |
| DummyLayer           | dropout_81                            |          0 |    (512,)    |       0 |     0.000  |
| Dense                | dense_80                              |          0 |    (10,)     |    5120 |    20.078  |
| Activation           | dense_801                             |          0 |    (10,)     |       0 |     0.000  |
| Activation           | activation_405                        |          0 |    (10,)     |       0 |     0.000  |
+----------------------+---------------------------------------+------------+--------------+---------+------------+
Total params (NT)....: 0
Total size in KiB....: 655.730
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
