#include <stdio.h>
#include "neural_net.h"
#include "logistic_regression_model.h"
#include "example_file.h"


 // esto no iría aca, solo sería para arduino


data1d_t input = { INPUT_LENGTH,  NULL  };


data1d_t results;



int main(void){
	
        // model initialization
    model_init();
    
    // make model prediction
    // uncomment corresponding code
    
    // int prediction = model_predict_class(input, &results);
    
    // print predicted class id
    int i, ok=0, prediction;
    printf("example_file.h tests\n");
    printf("Error | Cls | Pred \n");
    printf("------|-----|------\n");
    for (i=0; i<TEST_SAMPLES; i++) {
        input.data = sample_data[i];
        prediction = model_predict_class(input, &results);
        if (prediction == sample_data_ids[i][0]) {
            ok++;
            printf("       |  %2d |  %2d  \n", sample_data_ids[i][0], prediction);
        }
        else {
            printf("   X   |  %2d |  %2d  \n", sample_data_ids[i][0], prediction);
        }
    }
    printf("\n%d correct out of %d (Accuracy: %.2f%%)\n", ok, TEST_SAMPLES, (100.0 * ok)/TEST_SAMPLES);

	return 0;
}