#include "cifar10_model_dpw_micro_model.h"
#include "embedia_debug.h"

// Initialization function prototypes
depthwise_conv2d_layer_t init_depthwise_conv2d_data(void);
conv2d_layer_t init_conv2d_data(void);
dense_layer_t init_dense_data(void);


// Global Variables
depthwise_conv2d_layer_t depthwise_conv2d_data;
conv2d_layer_t conv2d_data;
dense_layer_t dense_data;


void model_init(){
    depthwise_conv2d_data = init_depthwise_conv2d_data();
    conv2d_data = init_conv2d_data();
    dense_data = init_dense_data();

}

void model_predict(data3d_t input, data1d_t * output){
  
    prepare_buffers();
    
    //<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//
    // Layer name: channels_adapter
    data3d_t output0;
    channel_adapt_layer(input, &output0);
    // Debug function for layer channels_adapter
    print_data3d_t("channels_adapter", output0);
    
    //******************** LAYER 0 *******************//
    // Layer name: depthwise_conv2d
    input = output0;
    depthwise_conv2d_layer(depthwise_conv2d_data, input, &output0);
    // Activation layer for depthwise_conv2d
    relu_activation(output0.data, 27);
    // Debug function for layer depthwise_conv2d
    print_data3d_t("depthwise_conv2d", output0);
    
    //******************** LAYER 1 *******************//
    // Layer name: conv2d
    input = output0;
    conv2d_layer(conv2d_data, input, &output0);
    // Activation layer for conv2d
    relu_activation(output0.data, 4);
    // Debug function for layer conv2d
    print_data3d_t("conv2d", output0);
    
    //******************** LAYER 2 *******************//
    // Layer name: flatten
    input = output0;
    data1d_t output1;
    flatten3d_layer(input, &output1);
    // Debug function for layer flatten
    print_data1d_t("flatten", output1);
    
    //******************** LAYER 3 *******************//
    // Layer name: dense
    data1d_t input1;
    input1 = output1;
    dense_layer(dense_data, input1, &output1);
    
    // Activation layer for dense
    softmax_activation(output1.data, 10);
    // Debug function for layer dense
    print_data1d_t("dense", output1);
    

    *output = output1;

}

int model_predict_class(data3d_t input, data1d_t * results){
  
   
    model_predict(input, results);
    
    return argmax(*results);
    //return argmax(data1d_t);

}

// Implementation of initialization functions


depthwise_conv2d_layer_t init_depthwise_conv2d_data(void){

    static const quant8 weights[]={
        0, 19, /* [-0.33705595 -0.22630575] */
        41, 55, /* [-0.09488136 -0.01066789] */

        194, 147, /* [0.81266463 0.53549987] */
        201, 84, /* [0.85063416 0.16018569] */

        255, 156, /* [1.17190123 0.5859018 ] */
        77, 43 /* [ 0.11600907 -0.08175512] */
    };
    static const quant8 biases[]={
        240, /* -0.012253438122570515 */
        255, /* 0.008653609082102776 */
        0 /* -0.35621219873428345 */
    };

    depthwise_conv2d_layer_t layer = {3, 2, weights, biases,{ 0.005917479127061133, 57 },{ 0.0014308463124667898, 249 } };
        
    return layer;
}


conv2d_layer_t init_conv2d_data(void){

        static filter_t filters[1];
        
        static const quant8 weights0[]={
           196,    202, /* [0.32989269 0.38138127] */
           105,    157, /* [-0.50289249 -0.02840698] */
           242,    194, /* [0.75026613 0.31238958] */
           255,    168, /* [0.87125576 0.07324774] */
           57,    90, /* [-0.94076091 -0.6415019 ] */
           0,    132 /* [-1.46403098 -0.2572526 ] */
        };
        static filter_t filter0 = {3, 2, weights0, 169};  //0.0802343338727951
        filters[0]=filter0;
            
        conv2d_layer_t layer = {1, filters,{ 0.009157987201915067, 160 } };
        return layer;
}
        
dense_layer_t init_dense_data(void){

    static neuron_t neurons[10];

    /* [-0.4252746  -1.2841107  -0.92804694 -0.2405821 ] 1.866814136505127*/
    static const quant8 weights0[] ={
    70, 0, 29, 85
    };
    
    static const neuron_t neuron0 = {weights0, 255 , { 0.0123565673828125, 104 } };
    neurons[0]=neuron0;

    /* [ 0.8033238  -0.5547778  -0.15371989 -1.7993286 ] 1.4004520177841187*/
    static const quant8 weights1[] ={
    207, 99, 131, 0
    };
    
    static const neuron_t neuron1 = {weights1, 255 , { 0.012548158683028875, 143 } };
    neurons[1]=neuron1;

    /* [ 0.9187252  -0.11535958 -0.40307298  0.4963748 ] -0.8099570870399475*/
    static const quant8 weights2[] ={
    255, 102, 60, 192
    };
    
    static const neuron_t neuron2 = {weights2, 0 , { 0.006779146194458008, 119 } };
    neurons[2]=neuron2;

    /* [-0.20064855  1.1903534  -0.37728393 -0.09389934] -0.4036671817302704*/
    static const quant8 weights3[] ={
    33, 255, 5, 50
    };
    
    static const neuron_t neuron3 = {weights3, 0 , { 0.0062510611964207066, 65 } };
    neurons[3]=neuron3;

    /* [0.35876903 0.3583703  0.44948962 0.5222938 ] -1.748803973197937*/
    static const quant8 weights4[] ={
    236, 236, 246, 255
    };
    
    static const neuron_t neuron4 = {weights4, 0 , { 0.008906265333587049, 196 } };
    neurons[4]=neuron4;

    /* [ 0.78234226  0.7720397  -0.6205929  -0.40840942] -0.4085862338542938*/
    static const quant8 weights5[] ={
    255, 253, 0, 39
    };
    
    static const neuron_t neuron5 = {weights5, 39 , { 0.005501706459942986, 113 } };
    neurons[5]=neuron5;

    /* [ 0.07507578  1.3719721  -0.36975783  1.1442033 ] -2.3730998039245605*/
    static const quant8 weights6[] ={
    167, 255, 137, 240
    };
    
    static const neuron_t neuron6 = {weights6, 0 , { 0.01468655642341165, 162 } };
    neurons[6]=neuron6;

    /* [ 0.2939142 -1.2788681  1.0955516  0.7053189] -1.0461962223052979*/
    static const quant8 weights7[] ={
    169, 0, 255, 213
    };
    
    static const neuron_t neuron7 = {weights7, 25 , { 0.009311449761484184, 137 } };
    neurons[7]=neuron7;

    /* [-1.0086966  -0.0509311  -0.97559124 -1.4185598 ] 2.2768301963806152*/
    static const quant8 weights8[] ={
    28, 94, 31, 0
    };
    
    static const neuron_t neuron8 = {weights8, 255 , { 0.014491725435443952, 98 } };
    neurons[8]=neuron8;

    /* [-0.7743209  -1.1501267   0.75631166 -0.6127048 ] 1.1171876192092896*/
    static const quant8 weights9[] ={
    42, 0, 214, 60
    };
    
    static const neuron_t neuron9 = {weights9, 255 , { 0.00889142915314319, 129 } };
    neurons[9]=neuron9;

    dense_layer_t layer= { 10, neurons};
    return layer;
}

