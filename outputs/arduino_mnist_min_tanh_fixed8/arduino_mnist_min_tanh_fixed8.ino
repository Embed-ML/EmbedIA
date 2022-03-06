
#include "embedia.h"
#include "mnist_digits_model.h"

#define INPUT_SIZE (MNIST_DIGITS_MODEL_CHANNELS*MNIST_DIGITS_MODEL_WIDTH*MNIST_DIGITS_MODEL_HEIGHT)

// Buffer with number 5 example for test
fixed input_data[INPUT_SIZE]= {
  FL2FX(0.0), FL2FX(6.0), FL2FX(13.0), FL2FX(5.0), FL2FX(8.0), FL2FX(8.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(8.0), 
  FL2FX(16.0), FL2FX(16.0), FL2FX(16.0), FL2FX(16.0), FL2FX(6.0), FL2FX(0.0), FL2FX(0.0), FL2FX(6.0), FL2FX(16.0), FL2FX(9.0), 
  FL2FX(6.0), FL2FX(4.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(6.0), FL2FX(16.0), FL2FX(16.0), FL2FX(15.0), FL2FX(5.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(4.0), FL2FX(5.0), FL2FX(15.0), FL2FX(12.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(3.0), FL2FX(16.0), FL2FX(9.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), 
  FL2FX(8.0), FL2FX(13.0), FL2FX(15.0), FL2FX(3.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(4.0), FL2FX(16.0), FL2FX(15.0), 
  FL2FX(3.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)}; 

// Structure with input data for the inference function
data_t input = { MNIST_DIGITS_MODEL_CHANNELS, MNIST_DIGITS_MODEL_WIDTH, MNIST_DIGITS_MODEL_HEIGHT, input_data };

// Structure with inference output results
flatten_data_t results;

void setup(){
      
    Serial.begin(9600);

    // model initialization
    model_init();
      
} 
        
void loop(){
      
    
  // model inference
  int prediction = model_predict(input, &results);    
    
  // print predicted class id
  Serial.print("Prediction class id: ");
    Serial.println(prediction); 


}
  