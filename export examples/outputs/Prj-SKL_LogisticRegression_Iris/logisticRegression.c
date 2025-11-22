#include "logisticRegression.h"
#include <math.h> 

void logistic_regression_layer(logistic_regression_layer_t lr, data1d_t input, data1d_t * output){ 
  
  //Si no se pasan bien la cantidad de parámetros
  if (lr.n_features < 1){ 
      //printf("Mal pasaje de parámetros del modelo hacia C");
      return;
  }
//----------------------Clasificación binaria-------------------//
  if (lr.n_classes == 2){ 

    float z_b = lr.bias[0];

    for (int i=0; i< lr.n_features; i++){ //Se recorren los features
      z_b += lr.weights[i] * input.data[i]; //Se hace la predicción
    }
    
    float p = 1.0f / (1.0f + expf(-z_b));
    output->data[0] = (p >= 0.5f) ? lr.classes[1] : lr.classes[0];
    
  }else{
//--------Clasificiación multiclase, devuelve la clase ganadora-------//
    float z[lr.n_classes]; 
    
    for (int j = 0; j < lr.n_classes; j++){ //Se recorre por cada clase
  
      z[j] = lr.bias[j];
  
      for (int k = 0; k < lr.n_features; k++){ //Se recorre por cada feature
        int w_index = (j * lr.n_features) + k; //Se calcula el índice 1D 
        z[j] += input.data[k] * lr.weights[w_index]; 
      }
    }
        
    float max = z[0];
    int max_prob_index = 0;
    for (int l = 0; l < lr.n_classes; l++){
        if (z[l] > max){
          max = z[l];
          max_prob_index = l;
        }  
    }
    
    //Guardamos el resultado en la salida
    output->data[0] = (float)lr.classes[max_prob_index]; 
  }
}
