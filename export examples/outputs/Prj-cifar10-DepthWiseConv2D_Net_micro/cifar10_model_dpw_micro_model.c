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

        
    static const float weights[]={
        -0.3370559513568878, -0.22630575299263, 
        -0.09488135576248169, -0.010667894035577774, 

        0.8126646280288696, 0.5354998707771301, 
        0.8506341576576233, 0.16018569469451904, 

        1.1719012260437012, 0.5859017968177795, 
        0.11600907146930695, -0.0817551240324974
    };
    static const float biases[]={
        -0.012253438122570515,
        0.008653609082102776,
        -0.35621219873428345
    };

    depthwise_conv2d_layer_t layer = {3, 2, weights, biases };
        
    return layer;
}


conv2d_layer_t init_conv2d_data(void){

        static filter_t filters[1];
        
        static const float weights0[]={
           0.32989269495010376,    0.3813812732696533, //0,0,0..1=>  0.32989269495010376,   0.3813812732696533, 
           -0.5028924942016602,    -0.028406977653503418, //0,1,0..1=>  -0.5028924942016602,   -0.028406977653503418, 
           0.7502661347389221,    0.3123895823955536, //1,0,0..1=>  0.7502661347389221,   0.3123895823955536, 
           0.8712557554244995,    0.07324773818254471, //1,1,0..1=>  0.8712557554244995,   0.07324773818254471, 
           -0.9407609105110168,    -0.6415019035339355, //2,0,0..1=>  -0.9407609105110168,   -0.6415019035339355, 
           -1.4640309810638428,    -0.25725260376930237, //2,1,0..1=>  -1.4640309810638428,   -0.25725260376930237, 
        
        };
        static filter_t filter0 = {3, 2, weights0, 0.0802343338727951}; 
        filters[0]=filter0;
            
        conv2d_layer_t layer = {1, filters };
        return layer;
}
        
dense_layer_t init_dense_data(void){

    static neuron_t neurons[10];

    /* [-0.4252746  -1.2841107  -0.92804694 -0.2405821 ] 1.866814136505127*/
    static const float weights0[] ={
    -0.4252746105194092, -1.28411066532135, -0.9280469417572021, -0.24058209359645844, 
  
    };
    
    static const neuron_t neuron0 = {weights0, 1.866814136505127  };
    neurons[0]=neuron0;

    /* [ 0.8033238  -0.5547778  -0.15371989 -1.7993286 ] 1.4004520177841187*/
    static const float weights1[] ={
    0.8033238053321838, -0.5547778010368347, -0.15371988713741302, -1.7993285655975342, 
  
    };
    
    static const neuron_t neuron1 = {weights1, 1.4004520177841187  };
    neurons[1]=neuron1;

    /* [ 0.9187252  -0.11535958 -0.40307298  0.4963748 ] -0.8099570870399475*/
    static const float weights2[] ={
    0.9187251925468445, -0.1153595820069313, -0.4030729830265045, 0.49637478590011597, 
  
    };
    
    static const neuron_t neuron2 = {weights2, -0.8099570870399475  };
    neurons[2]=neuron2;

    /* [-0.20064855  1.1903534  -0.37728393 -0.09389934] -0.4036671817302704*/
    static const float weights3[] ={
    -0.20064854621887207, 1.1903533935546875, -0.3772839307785034, -0.09389933943748474, 
  
    };
    
    static const neuron_t neuron3 = {weights3, -0.4036671817302704  };
    neurons[3]=neuron3;

    /* [0.35876903 0.3583703  0.44948962 0.5222938 ] -1.748803973197937*/
    static const float weights4[] ={
    0.358769029378891, 0.358370304107666, 0.44948962330818176, 0.5222938060760498
    };
    
    static const neuron_t neuron4 = {weights4, -1.748803973197937  };
    neurons[4]=neuron4;

    /* [ 0.78234226  0.7720397  -0.6205929  -0.40840942] -0.4085862338542938*/
    static const float weights5[] ={
    0.782342255115509, 0.7720397114753723, -0.6205928921699524, -0.4084094166755676, 
  
    };
    
    static const neuron_t neuron5 = {weights5, -0.4085862338542938  };
    neurons[5]=neuron5;

    /* [ 0.07507578  1.3719721  -0.36975783  1.1442033 ] -2.3730998039245605*/
    static const float weights6[] ={
    0.07507577538490295, 1.3719720840454102, -0.36975783109664917, 1.1442033052444458, 
  
    };
    
    static const neuron_t neuron6 = {weights6, -2.3730998039245605  };
    neurons[6]=neuron6;

    /* [ 0.2939142 -1.2788681  1.0955516  0.7053189] -1.0461962223052979*/
    static const float weights7[] ={
    0.29391419887542725, -1.2788680791854858, 1.095551609992981, 0.7053189277648926, 
  
    };
    
    static const neuron_t neuron7 = {weights7, -1.0461962223052979  };
    neurons[7]=neuron7;

    /* [-1.0086966  -0.0509311  -0.97559124 -1.4185598 ] 2.2768301963806152*/
    static const float weights8[] ={
    -1.0086965560913086, -0.05093110352754593, -0.975591242313385, -1.4185597896575928, 
  
    };
    
    static const neuron_t neuron8 = {weights8, 2.2768301963806152  };
    neurons[8]=neuron8;

    /* [-0.7743209  -1.1501267   0.75631166 -0.6127048 ] 1.1171876192092896*/
    static const float weights9[] ={
    -0.7743209004402161, -1.1501266956329346, 0.7563116550445557, -0.6127048134803772, 
  
    };
    
    static const neuron_t neuron9 = {weights9, 1.1171876192092896  };
    neurons[9]=neuron9;

    dense_layer_t layer= { 10, neurons};
    return layer;
}

