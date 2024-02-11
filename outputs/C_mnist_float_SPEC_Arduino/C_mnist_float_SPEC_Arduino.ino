#include "Arduino.h"
#include "embedia.h"
#include "embedia_model.h"
#include "example_file.h"


data1d_t input = { INPUT_LENGTH,  NULL};


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

    int prediction = model_predict_class(input, &results);

    // print predicted class id
    Serial.print("Prediction class id: ");
    Serial.println(prediction);

    Serial.print("   Example class id: ");
    Serial.println(sample_data_id);

    /*

    model_predict(input, &results);

    printf("prediccion: %.5f", results.data[0]);

    */



    delay(5000);

}