#include <stdio.h>
#include "embedia.h"
#include "mnist_digits_model.h"

#define INPUT_SIZE (MNIST_DIGITS_MODEL_CHANNELS*MNIST_DIGITS_MODEL_WIDTH*MNIST_DIGITS_MODEL_HEIGHT)

// Buffer with number 5 example for test
float input_data[INPUT_SIZE]= {
  0.0, 6.0, 13.0, 5.0, 8.0, 8.0, 1.0, 0.0, 0.0, 8.0, 16.0, 16.0, 16.0, 16.0, 6.0, 0.0, 0.0, 6.0, 16.0, 9.0, 6.0, 4.0, 0.0, 
  0.0, 0.0, 6.0, 16.0, 16.0, 15.0, 5.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 15.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 16.0, 9.0, 
  0.0, 0.0, 0.0, 1.0, 8.0, 13.0, 15.0, 3.0, 0.0, 0.0, 0.0, 4.0, 16.0, 15.0, 3.0, 0.0, 0.0, 0.0}; 

// Structure with input data for the inference function
data_t input = { MNIST_DIGITS_MODEL_CHANNELS, MNIST_DIGITS_MODEL_WIDTH, MNIST_DIGITS_MODEL_HEIGHT, input_data };

// Structure with inference output results
flatten_data_t results;

int main(void){

  // model initialization
  model_init();
    
  // model inference
  int prediction = model_predict(input, &results);    
    
  // print predicted class id
  printf("Prediction class id: %d\n", prediction); 

  return 0;
}
  