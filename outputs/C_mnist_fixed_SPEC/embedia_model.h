/* EmbedIA model definition file*/
#ifndef EMBEDIA_MODEL_H
#define EMBEDIA_MODEL_H

/*

+-------------------+---------------+------------+--------------+--------+------------+
| Layer(activation) | Name          | #Param(NT) |    Shape     |   MACs | Size (KiB) |
+-------------------+---------------+------------+--------------+--------+------------+
| Melspec           | Melspec       |          0 | (23, 32, 1)  |      0 |     0.000  |
| Conv2D(relu)      | conv2d        |        160 | (21, 30, 16) |  90720 |     0.438  |
| MaxPooling2D      | max_pooling2d |          0 | (10, 15, 16) |      0 |     0.000  |
| Conv2D(relu)      | conv2d_1      |       2320 | (8, 13, 16)  | 239616 |     4.656  |
| Flatten           | flatten       |          0 |   (1664,)    |      0 |     0.000  |
| Dense(relu)       | dense         |      53280 |    (32,)     |  53248 |   104.188  |
| Dense(softmax)    | dense_1       |        198 |     (6,)     |    192 |     0.410  |
+-------------------+---------------+------------+--------------+--------+------------+
Total params (NT)....: 55958
Total size in KiB....: 109.691
Total MACs operations: 383776

*/

#include "embedia.h"

#define INPUT_LENGTH 6000

#define INPUT_SIZE 6000


void model_init();

void model_predict(data1d_t input, data1d_t * output);

int model_predict_class(data1d_t input, data1d_t * results);

#endif
