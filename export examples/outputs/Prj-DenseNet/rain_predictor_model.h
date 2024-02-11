/* EmbedIA model definition file*/
#ifndef RAIN_PREDICTOR_MODEL_H
#define RAIN_PREDICTOR_MODEL_H

/*

+-------------------+----------------+------------+-------+------+------------+
| Layer(activation) | Name           | #Param(NT) | Shape | MACs | Size (KiB) |
+-------------------+----------------+------------+-------+------+------------+
| StandardScaler    | StandardScaler |          0 | (13,) |   13 |     0.021  |
| Dense(relu)       | dense          |        224 | (16,) |  208 |     0.359  |
| Dropout           | dropout        |          0 | (16,) |    0 |     0.000  |
| Dense(linear)     | dense_1        |        136 |  (8,) |  128 |     0.203  |
| Activation(relu)  | activation     |          0 |  (8,) |    0 |     0.000  |
| Dropout           | dropout_1      |          0 |  (8,) |    0 |     0.000  |
| Dense(sigmoid)    | dense_2        |          9 |  (1,) |    8 |     0.018  |
+-------------------+----------------+------------+-------+------+------------+
Total params (NT)....: 369
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
