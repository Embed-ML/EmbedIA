/* EmbedIA model definition file*/
#ifndef RAIN_PREDICTOR_MODEL_H
#define RAIN_PREDICTOR_MODEL_H

/*

+---------------+-------------------------------+------------+-------+------+------------+
| EmbedIA Layer | Name                          | #Param(NT) | Shape | MACs | Size (KiB) |
+---------------+-------------------------------+------------+-------+------+------------+
| Normalization | s_k_l_standard_scaler_wrapper |          0 | (13,) |   13 |     0.021  |
| Dense         | dense                         |          0 | (16,) |  208 |     0.359  |
| Activation    | dense1                        |          0 | (16,) |    0 |     0.000  |
| DummyLayer    | dropout                       |          0 | (16,) |    0 |     0.000  |
| Dense         | dense_1                       |          0 |  (8,) |  128 |     0.203  |
| Activation    | dense_11                      |          0 |  (8,) |    0 |     0.000  |
| Activation    | activation                    |          0 |  (8,) |    0 |     0.000  |
| DummyLayer    | dropout_1                     |          0 |  (8,) |    0 |     0.000  |
| Dense         | dense_2                       |          0 |  (1,) |    8 |     0.018  |
| Activation    | dense_21                      |          0 |  (1,) |    0 |     0.000  |
+---------------+-------------------------------+------------+-------+------+------------+
Total params (NT)....: 0
Total size in KiB....: 0.601
Total MACs operations: 357

*/

#include "embedia.h"

#define INPUT_LENGTH 13

#define INPUT_SIZE 13


void model_init();

void model_predict(data1d_t input, data1d_t * output);

int model_predict_class(data1d_t input, data1d_t * results);

#endif
