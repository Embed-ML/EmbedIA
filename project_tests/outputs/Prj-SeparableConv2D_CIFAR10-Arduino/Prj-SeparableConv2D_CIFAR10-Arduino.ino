#include "Arduino.h"
#include "embedia.h"
#include "separable_model.h"
#include "example_file.h"


data3d_t input = { INPUT_CHANNELS, INPUT_WIDTH, INPUT_HEIGHT,  NULL};


data1d_t results;


void setup(){

    // Serial inicialization
    Serial.begin(9600);

    // Model initialization
    model_init();

}

void loop(){

    
    // sample intitialization
    input.data = sample_data;

    // model initialization
    model_init();


    // make model prediction
    // uncomment corresponding code
    /*
    int prediction = model_predict_class(input, &results);

    // print predicted class id
    Serial.print("Prediction class id: ");
    Serial.println(prediction);

    */

    model_predict(input, &results);
    printf("prediccion: %5f", results.data[0]);



}