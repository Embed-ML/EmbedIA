/* EmbedIA model definition file*/
#ifndef _SKL_KNN_IRIS_MODEL_H_H
#define _SKL_KNN_IRIS_MODEL_H_H

/*

+----------------------+--------------------+------------+-------+------+------------+
| EmbedIA Layer        | Name               | #Param(NT) | Shape | MACs | Size (KiB) |
+----------------------+--------------------+------------+-------+------+------------+
| Normalization        | Standard_Scaler    |       8(8) |  (4,) |    4 |     0.012  |
| KNeighborsClassifier | SKL_KNN_iris_model |   600(600) |  (3,) | 1205 |     1.172  |
+----------------------+--------------------+------------+-------+------+------------+
Total params (NT)....: 608(608)
Total size in KiB....: 1.184
Total MACs operations: 1209

*/

#include "common.h"

#define INPUT_LENGTH 4

#define INPUT_SIZE 4


void model_init();

void model_predict(data1d_t input, data1d_t * output);

int model_predict_class(data1d_t input, data1d_t * results);

#endif
