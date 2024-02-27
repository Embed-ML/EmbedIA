#include "zero_padding2d_test_model.h"
#include "embedia_debug.h"

// Initialization function prototypes


// Global Variables


void model_init(){

}

void model_predict(data3d_t input, data3d_t * output){
  
    prepare_buffers();
    
    //<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//
    // Layer name: channels_adapter
    data3d_t output0;
    channel_adapt_layer(input, &output0);
    // Debug function for layer channels_adapter
    print_data3d_t("channels_adapter", output0);
    
    //******************** LAYER 0 *******************//
    // Layer name: zero_padding2d
    input = output0;
    zero_padding2d_layer((2, 2), (1, 1), input, &output0);
    // Debug function for layer zero_padding2d
    print_data3d_t("zero_padding2d", output0);
    

    *output = output0;

}

int model_predict_class(data3d_t input, data3d_t * results){
  
   
    model_predict(input, results);
    
    //TO DO: argmax with data2d_t and data3d_t
    return -1; 
    //return argmax(data1d_t);

}

// Implementation of initialization functions


