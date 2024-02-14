#include "softmax_test_model.h"
#include "embedia_debug.h"

// Initialization function prototypes
dense_layer_t init_dense_data(void);


// Global Variables
dense_layer_t dense_data;


void model_init(){
    dense_data = init_dense_data();

}

void model_predict(data1d_t input, data2d_t * output){
  
    prepare_buffers();
    
    //******************** LAYER 0 *******************//
    // Layer name: dense
    data1d_t output0;
    dense_layer(dense_data, input, &output0);
    
    // Debug function for layer dense
    print_data1d_t("dense", output0);
    
    //******************** LAYER 1 *******************//
    // Layer name: softmax
    softmax_activation(output0.data, 100);
    // Debug function for layer softmax
    print_data2d_t("softmax", output0);
    

    *output = output0;

}

int model_predict_class(data1d_t input, data2d_t * results){
  
   
    model_predict(input, results);
    
    //TO DO: argmax with data2d_t and data3d_t
    return -1; 
    //return argmax(data1d_t);

}

// Implementation of initialization functions


dense_layer_t init_dense_data(void){

    static neuron_t neurons[10];

    /* [-0.39257756] 0.0*/
    static const quant8 weights0[] ={
    0
    };
    
    static const neuron_t neuron0 = {weights0, 255 , { 0.001539519838258332, 255 } };
    neurons[0]=neuron0;

    /* [0.06460512] 0.0*/
    static const quant8 weights1[] ={
    255
    };
    
    static const neuron_t neuron1 = {weights1, 0 , { 0.0002533533993889304, 0 } };
    neurons[1]=neuron1;

    /* [-0.5471277] 0.0*/
    static const quant8 weights2[] ={
    0
    };
    
    static const neuron_t neuron2 = {weights2, 255 , { 0.0021455989164464616, 255 } };
    neurons[2]=neuron2;

    /* [-0.61707383] 0.0*/
    static const quant8 weights3[] ={
    0
    };
    
    static const neuron_t neuron3 = {weights3, 255 , { 0.0024198973880094642, 255 } };
    neurons[3]=neuron3;

    /* [-0.47769684] 0.0*/
    static const quant8 weights4[] ={
    0
    };
    
    static const neuron_t neuron4 = {weights4, 255 , { 0.001873320925469492, 255 } };
    neurons[4]=neuron4;

    /* [-0.47027737] 0.0*/
    static const quant8 weights5[] ={
    0
    };
    
    static const neuron_t neuron5 = {weights5, 255 , { 0.0018442249765583112, 255 } };
    neurons[5]=neuron5;

    /* [0.6207257] 0.0*/
    static const quant8 weights6[] ={
    255
    };
    
    static const neuron_t neuron6 = {weights6, 0 , { 0.002434218397327498, 0 } };
    neurons[6]=neuron6;

    /* [-0.47601646] 0.0*/
    static const quant8 weights7[] ={
    0
    };
    
    static const neuron_t neuron7 = {weights7, 255 , { 0.0018667312229380887, 255 } };
    neurons[7]=neuron7;

    /* [-0.7361061] 0.0*/
    static const quant8 weights8[] ={
    0
    };
    
    static const neuron_t neuron8 = {weights8, 255 , { 0.002886690579208673, 255 } };
    neurons[8]=neuron8;

    /* [-0.17727464] 0.0*/
    static const quant8 weights9[] ={
    0
    };
    
    static const neuron_t neuron9 = {weights9, 255 , { 0.000695194683822931, 255 } };
    neurons[9]=neuron9;

    dense_layer_t layer= { 10, neurons};
    return layer;
}

