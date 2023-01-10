#include "Arduino.h"
#include "embedia.h"
#include "person_detection_model.h"
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


  unsigned long int tiempo = millis();
    
    int prediction = model_predict_class(input, &results);

    tiempo = millis() - tiempo;

    // print predicted class id
    Serial.print("Prediction class id: ");
    Serial.println(prediction);

    Serial.print("tiempo ms: ");
    Serial.println(tiempo);



}
