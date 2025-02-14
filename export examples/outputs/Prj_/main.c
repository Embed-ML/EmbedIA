#include <stdio.h>
#include "neural_net.h"
#include "layerstest01_model.h"


 // esto no iría aca, solo sería para arduino


data3d_t input = { INPUT_CHANNELS, INPUT_WIDTH, INPUT_HEIGHT,  NULL};


data1d_t results;



int main(void){
	
    
    // model initialization
    model_init();


    // make model prediction
    // uncomment corresponding code

    int prediction = model_predict_class(input, &results);

    // print predicted class id
    printf("Prediction class id: %d\n", prediction);

    /*

    model_predict(input, &results);

    printf("prediccion: %.5f", results.data[0]);

    */



	return 0;
}