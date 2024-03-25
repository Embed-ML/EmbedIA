#include "rain_predictor_model.h"
#include "embedia_debug.h"

// Initialization function prototypes
normalization_layer_t init_s_k_l_standard_scaler_wrapper_data(void);
dense_layer_t init_dense_data(void);
dense_layer_t init_dense_1_data(void);
dense_layer_t init_dense_2_data(void);


// Global Variables
normalization_layer_t s_k_l_standard_scaler_wrapper_data;
dense_layer_t dense_data;
dense_layer_t dense_1_data;
dense_layer_t dense_2_data;


void model_init(){
    s_k_l_standard_scaler_wrapper_data = init_s_k_l_standard_scaler_wrapper_data();
    dense_data = init_dense_data();
    dense_1_data = init_dense_1_data();
    dense_2_data = init_dense_2_data();

}

void model_predict(data1d_t input, data1d_t * output){
  
    prepare_buffers();
    
    //******************** LAYER 0 *******************//
    // Layer name: s_k_l_standard_scaler_wrapper
    data1d_t output0;
    standard_norm_layer(s_k_l_standard_scaler_wrapper_data, input, &output0);
    // Debug function for layer s_k_l_standard_scaler_wrapper
    print_data1d_t("s_k_l_standard_scaler_wrapper", output0);
    
    //******************** LAYER 1 *******************//
    // Layer name: dense
    input = output0;
    dense_layer(dense_data, input, &output0);
    
    // Debug function for layer dense
    print_data1d_t("dense", output0);
    
    //******************** LAYER 2 *******************//
    // Layer name: dense1
    relu_activation(output0.data, 16);
    // Debug function for layer dense1
    print_data1d_t("dense1", output0);
    
    //******************** LAYER 3 *******************//
    // Layer name: dropout
    input = output0;
    
    // Debug function for layer dropout
    
    
    //******************** LAYER 4 *******************//
    // Layer name: dense_1
    input = output0;
    dense_layer(dense_1_data, input, &output0);
    
    // Debug function for layer dense_1
    print_data1d_t("dense_1", output0);
    
    //******************** LAYER 5 *******************//
    // Layer name: dense_11
    
    // Debug function for layer dense_11
    print_data1d_t("dense_11", output0);
    
    //******************** LAYER 6 *******************//
    // Layer name: activation
    relu_activation(output0.data, 8);
    // Debug function for layer activation
    print_data1d_t("activation", output0);
    
    //******************** LAYER 7 *******************//
    // Layer name: dropout_1
    input = output0;
    
    // Debug function for layer dropout_1
    
    
    //******************** LAYER 8 *******************//
    // Layer name: dense_2
    input = output0;
    dense_layer(dense_2_data, input, &output0);
    
    // Debug function for layer dense_2
    print_data1d_t("dense_2", output0);
    
    //******************** LAYER 9 *******************//
    // Layer name: dense_21
    sigmoid_activation(output0.data, 1);
    // Debug function for layer dense_21
    print_data1d_t("dense_21", output0);
    

    *output = output0;

}

int model_predict_class(data1d_t input, data1d_t * results){
  
   
    model_predict(input, results);
    
    return results->data[0] >= 0.5;
    //return argmax(data1d_t);

}

// Implementation of initialization functions


normalization_layer_t init_s_k_l_standard_scaler_wrapper_data(void){
    /*[ 2.59901768e+01  1.17721022e+01  1.00964173e+03  6.70353635e+01
  2.22318271e+01  3.25088409e+01  1.95186640e+01  1.59666012e+01
  7.16895874e+00  1.01242633e+03  1.00540668e+03 -6.54297191e-02
  1.97393819e-02]*/
    static const float sub_val[] ={
    25.990176817288802, 11.772102161100197, 1009.6417288801567, 67.03536345776031, 22.231827111984284, 
    32.50884086444008, 19.518664047151276, 15.966601178781925, 7.168958742632613, 1012.426326129666, 
    1005.4066797642436, -0.06542971911333069, 0.01973938189331926
    };
    /*[0.1603709  0.11630081 0.15450669 0.04583385 0.05265399 0.1676429
 0.14537339 0.13948787 0.08309928 0.15392047 0.13950817 1.47043861
 1.36994631]*/
    static const float inv_div_val[] ={
    0.16037089785923392, 0.11630081492430955, 0.15450669325054786, 0.045833850028244354, 
    0.052653985865521405, 0.16764289633360457, 0.1453733874841405, 0.1394878671234759, 
    0.08309927757766968, 0.1539204749347877, 0.13950817302774016, 1.4704386093316715, 
    1.3699463082523822
    };

    static const normalization_layer_t norm = { sub_val, inv_div_val  };
    return norm;
}

dense_layer_t init_dense_data(void){

    static neuron_t neurons[16];

    /* [ 0.07477052  0.18821196 -0.31022128  0.26096275 -0.35748366 -0.12674505
 -0.17246446 -0.36610463 -0.3993438   0.07953915 -0.23265055 -0.4625952
 -0.38713437] 0.16402770578861237*/
    static const quant8 weights0[] ={
    189, 229, 54, 255, 37, 118, 102, 34, 22, 191, 81, 0, 27
    };
    
    static const neuron_t neuron0 = {weights0, 221 , { 0.002837482153200636, 163 } };
    neurons[0]=neuron0;

    /* [-0.46836677 -0.15161344  0.00940471  0.13277254 -0.2650925  -0.00165856
 -0.3666864   0.24842465 -0.1210469  -0.09509674 -0.06223702 -0.38244227
  0.51487964] 0.12276548147201538*/
    static const quant8 weights1[] ={
    0, 82, 123, 155, 52, 121, 26, 185, 90, 96, 105, 22, 255
    };
    
    static const neuron_t neuron1 = {weights1, 153 , { 0.00385586841433656, 121 } };
    neurons[1]=neuron1;

    /* [-0.16033639 -0.10578484  0.11444841 -0.56316715 -0.1031829   0.01869397
  0.24031888  0.07032599  0.3976534  -0.31813008  0.2013956  -0.44131634
  0.03148782] 0.0828813835978508*/
    static const quant8 weights2[] ={
    106, 121, 179, 0, 122, 154, 213, 168, 255, 65, 202, 32, 157
    };
    
    static const neuron_t neuron2 = {weights2, 171 , { 0.0037679237477919636, 149 } };
    neurons[2]=neuron2;

    /* [-0.1456319  -0.17181091 -0.24932198  0.04128495 -0.37185958 -0.06810457
 -0.25493565  0.09846975 -0.02819531  0.36349833 -0.09554975  0.23111044
  0.04678858] -0.14431288838386536*/
    static const quant8 weights3[] ={
    78, 69, 43, 143, 0, 105, 41, 163, 119, 255, 96, 209, 145
    };
    
    static const neuron_t neuron3 = {weights3, 79 , { 0.002883756394479789, 129 } };
    neurons[3]=neuron3;

    /* [ 0.31139758 -0.20782745  0.13240202 -0.37353528 -0.39002922 -0.25124058
 -0.26382816  0.35558924  0.404634   -0.10956903 -0.08988525 -0.39560565
 -0.09683752] -0.03136235103011131*/
    static const quant8 weights4[] ={
    225, 60, 168, 7, 2, 46, 42, 239, 255, 91, 97, 0, 95
    };
    
    static const neuron_t neuron4 = {weights4, 116 , { 0.0031381948321473367, 126 } };
    neurons[4]=neuron4;

    /* [-0.17289661  0.19072206 -0.02450229  0.03306852  0.18028975  0.01420854
 -0.44025257 -0.4964326  -0.37562886  0.27386177  0.15537511 -0.1331667
  0.11743953] -0.20874863862991333*/
    static const quant8 weights5[] ={
    107, 227, 156, 175, 224, 169, 18, 0, 40, 255, 215, 120, 203
    };
    
    static const neuron_t neuron5 = {weights5, 95 , { 0.0030207622284982717, 164 } };
    neurons[5]=neuron5;

    /* [ 0.29837546  0.23475346 -0.35112688  0.1360474   0.29534686 -0.38667813
 -0.06437257 -0.1964145  -0.17627354  0.25644904 -0.2901877   0.30585277
  0.35403365] -0.12059317529201508*/
    static const quant8 weights6[] ={
    236, 214, 12, 180, 235, 0, 111, 65, 72, 221, 33, 238, 255
    };
    
    static const neuron_t neuron6 = {weights6, 91 , { 0.0029047521890378466, 133 } };
    neurons[6]=neuron6;

    /* [ 0.3834489   0.11146511 -0.16770078  0.31377777 -0.21574582 -0.33373374
 -0.279015   -0.13489848 -0.29464683  0.16133635  0.17392458  0.2638826
 -0.17488635] -0.1116621121764183*/
    static const quant8 weights7[] ={
    255, 159, 59, 231, 42, 0, 20, 71, 14, 176, 181, 213, 57
    };
    
    static const neuron_t neuron7 = {weights7, 79 , { 0.002812480926513672, 119 } };
    neurons[7]=neuron7;

    /* [-0.4481842   0.14934222  0.1694171  -0.47083265 -0.18088719  0.33870217
 -0.37887242 -0.4738073  -0.19109558  0.71226466 -0.21240686 -0.48533687
 -0.38171345] 0.2069426029920578*/
    static const quant8 weights8[] ={
    8, 135, 139, 3, 64, 175, 22, 2, 62, 255, 58, 0, 22
    };
    
    static const neuron_t neuron8 = {weights8, 147 , { 0.004696476693246879, 103 } };
    neurons[8]=neuron8;

    /* [ 0.18508084 -0.5469461  -0.22049917 -0.40266004  0.05872586 -0.1670955
  0.05725744 -0.6114371  -0.37275362  0.5389226   0.55725366 -0.40234986
  0.09533851] -0.033608485013246536*/
    static const quant8 weights9[] ={
    173, 14, 85, 45, 146, 97, 145, 0, 52, 251, 255, 45, 154
    };
    
    static const neuron_t neuron9 = {weights9, 126 , { 0.004583100711598116, 133 } };
    neurons[9]=neuron9;

    /* [ 0.16251189 -0.28912187  0.25269935  0.14681901 -0.19674121 -0.20625553
  0.27938595 -0.07020696 -0.2457219  -0.29473627 -0.29588294 -0.3883527
 -0.08130022] -0.0777340680360794*/
    static const quant8 weights10[] ={
    210, 38, 245, 204, 73, 69, 255, 121, 54, 35, 35, 0, 117
    };
    
    static const neuron_t neuron10 = {weights10, 118 , { 0.002618583043416341, 148 } };
    neurons[10]=neuron10;

    /* [-0.1107872  -0.33976465  0.06502116 -0.3309748  -0.4209577   0.31255943
 -0.25899315 -0.1996503   0.20953508 -0.24595267  0.04478132 -0.37459782
 -0.31158134] 0.13238975405693054*/
    static const quant8 weights11[] ={
    107, 28, 169, 31, 0, 255, 56, 77, 219, 60, 162, 16, 38
    };
    
    static const neuron_t neuron11 = {weights11, 192 , { 0.002876537921381932, 146 } };
    neurons[11]=neuron11;

    /* [-0.38404307 -0.40407526 -0.16603407  0.32631817  0.49619326  0.11323727
  0.27813748  0.24934328 -0.27759865  0.3070503  -0.33662483 -0.06658834
 -0.29125383] -0.18592236936092377*/
    static const quant8 weights12[] ={
    5, 0, 67, 206, 255, 146, 193, 185, 35, 201, 19, 95, 32
    };
    
    static const neuron_t neuron12 = {weights12, 61 , { 0.0035304649203431373, 114 } };
    neurons[12]=neuron12;

    /* [-0.17677826  0.18150781  0.14303683  0.28199673  0.12799723 -0.06145582
 -0.09232465  0.34081313 -0.19230215 -0.39443445  0.37609383  0.2402891
 -0.07268163] -0.06934040039777756*/
    static const quant8 weights13[] ={
    72, 191, 178, 224, 173, 111, 100, 244, 67, 0, 255, 211, 107
    };
    
    static const neuron_t neuron13 = {weights13, 108 , { 0.003021679672540403, 131 } };
    neurons[13]=neuron13;

    /* [ 0.23606779  0.26549697  0.36735597 -0.21020457 -0.37860376  0.03381604
 -0.07382872  0.264764   -0.00447221 -0.2778665   0.3801751   0.0667705
 -0.09164418] -0.1232815608382225*/
    static const quant8 weights14[] ={
    206, 216, 250, 56, 0, 138, 102, 216, 125, 34, 255, 149, 96
    };
    
    static const neuron_t neuron14 = {weights14, 86 , { 0.002975603412179386, 127 } };
    neurons[14]=neuron14;

    /* [ 0.04161454 -0.39180198  0.18596265 -0.3881799   0.21827197 -0.20377985
  0.1521275  -0.40840197 -0.26752728  0.3309193  -0.30381235 -0.298933
  0.2089586 ] -0.10482733696699142*/
    static const quant8 weights15[] ={
    155, 6, 205, 7, 216, 71, 193, 0, 49, 255, 36, 38, 213
    };
    
    static const neuron_t neuron15 = {weights15, 105 , { 0.0028992989484001607, 141 } };
    neurons[15]=neuron15;

    dense_layer_t layer= { 16, neurons};
    return layer;
}

dense_layer_t init_dense_1_data(void){

    static neuron_t neurons[8];

    /* [-0.53550696 -0.1713382   0.05235767  0.08924907  0.189863   -0.16881153
  0.48123854 -0.03225986 -0.0690899  -0.4191022   0.10090049 -0.57869166
  0.3806008   0.25919333  0.29740265 -0.50917   ] -0.19951002299785614*/
    static const quant8 weights0[] ={
    10, 98, 152, 160, 185, 98, 255, 131, 122, 38, 163, 0, 231, 201, 211, 17
    };
    
    static const neuron_t neuron0 = {weights0, 91 , { 0.004156589040569231, 139 } };
    neurons[0]=neuron0;

    /* [-0.25214916 -0.4455108   0.09540664 -0.25712678  0.24530582  0.28154072
 -0.36674297  0.30774295 -0.52017653 -0.327518    0.30020592 -0.18867508
  0.6281454  -0.24354456  0.05239502  0.33481097] -0.19112174212932587*/
    static const quant8 weights1[] ={
    60, 17, 137, 59, 170, 179, 35, 184, 0, 43, 183, 74, 255, 62, 128, 190
    };
    
    static const neuron_t neuron1 = {weights1, 74 , { 0.0045032230078005326, 116 } };
    neurons[1]=neuron1;

    /* [-0.17454718  0.3599786  -0.27291635 -0.21649306 -0.2766662  -0.27608255
 -0.29992568 -0.0937286  -0.5017065  -0.24150935  0.30341768 -0.2925527
 -0.13212715 -0.34280562  0.16728021  0.396549  ] -0.14100980758666992*/
    static const quant8 weights2[] ={
    92, 244, 65, 81, 63, 64, 57, 115, 0, 73, 228, 59, 104, 45, 189, 255
    };
    
    static const neuron_t neuron2 = {weights2, 102 , { 0.0035225704604504157, 142 } };
    neurons[2]=neuron2;

    /* [ 0.11275136 -0.08747412  0.02496984 -0.11722086  0.59986055 -0.10056303
 -0.4052912  -0.11072075  0.706256    0.3520824   0.55623287 -0.14828071
 -0.03833193 -0.27186254 -0.02465677  0.656244  ] 0.23744064569473267*/
    static const quant8 weights3[] ={
    119, 73, 99, 66, 231, 70, 0, 68, 255, 174, 221, 59, 84, 31, 87, 244
    };
    
    static const neuron_t neuron3 = {weights3, 147 , { 0.004359008751663508, 93 } };
    neurons[3]=neuron3;

    /* [-0.08112346  0.41320336  0.585519   -0.155398    0.11275487  0.11821771
 -0.3401699   0.27199328  0.49013987  0.48560226  0.27358887  0.36589673
  0.0055095  -0.14797318  0.13554136 -0.00542052] 0.18755173683166504*/
    static const quant8 weights4[] ={
    72, 208, 255, 51, 125, 127, 0, 169, 229, 228, 169, 195, 96, 53, 131, 93
    };
    
    static const neuron_t neuron4 = {weights4, 146 , { 0.0036301526368833054, 94 } };
    neurons[4]=neuron4;

    /* [ 0.5095873   0.36300486  0.11174337  0.41168183  0.45435885  0.31642526
 -0.57062864  0.00216176  0.11675161  0.4502989  -0.05098     0.43372074
 -0.50031686  0.11278214  0.19709945 -0.00667844] 0.10105646401643753*/
    static const quant8 weights5[] ={
    255, 221, 161, 232, 242, 210, 0, 136, 163, 241, 123, 237, 17, 162, 182, 133
    };
    
    static const neuron_t neuron5 = {weights5, 159 , { 0.004236140905642042, 135 } };
    neurons[5]=neuron5;

    /* [-0.15198787 -0.22230464  0.6980225  -0.02828806  0.29915458  0.5170933
  0.08910124 -0.3059761   0.826219    0.589584   -0.06105056 -0.00462323
 -0.25533822 -0.36150977  0.02852144  0.15287834] 0.1470976620912552*/
    static const quant8 weights6[] ={
    45, 30, 228, 72, 142, 189, 97, 12, 255, 205, 65, 77, 23, 0, 84, 111
    };
    
    static const neuron_t neuron6 = {weights6, 110 , { 0.004657759853437835, 78 } };
    neurons[6]=neuron6;

    /* [ 0.4827647   0.00134873  0.27708983  0.45488134  0.36249888  0.13629335
 -0.35020342  0.04593245  0.6677213   0.08964708  0.42644587 -0.05147675
 -0.17899376 -0.16772969 -0.0115309   0.44027615] 0.20837624371051788*/
    static const quant8 weights7[] ={
    209, 88, 157, 202, 179, 122, 0, 100, 255, 110, 195, 75, 43, 46, 85, 198
    };
    
    static const neuron_t neuron7 = {weights7, 140 , { 0.003991861436881271, 88 } };
    neurons[7]=neuron7;

    dense_layer_t layer= { 8, neurons};
    return layer;
}

dense_layer_t init_dense_2_data(void){

    static neuron_t neurons[1];

    /* [ 0.63990426  0.67669904  0.07067166 -0.9549373  -0.5433483  -0.45682818
 -0.62079483 -0.77979636] -0.3737352192401886*/
    static const quant8 weights0[] ={
    249, 255, 160, 0, 64, 78, 52, 27
    };
    
    static const neuron_t neuron0 = {weights0, 91 , { 0.006398574043722714, 149 } };
    neurons[0]=neuron0;

    dense_layer_t layer= { 1, neurons};
    return layer;
}

