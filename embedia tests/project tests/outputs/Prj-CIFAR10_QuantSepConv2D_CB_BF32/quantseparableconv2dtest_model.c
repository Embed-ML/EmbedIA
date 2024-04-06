#include "quantseparableconv2dtest_model.h"

// Initialization function prototypes
conv2d_layer_t init_conv2d_4_data(void);
batch_normalization_layer_t init_batch_normalization_394_data(void);
quant_separable_conv2d_layer_t init_larq_quant_separable_conv2_d_wrapper_data(void);
batch_normalization_layer_t init_batch_normalization_395_data(void);
quant_separable_conv2d_layer_t init_larq_quant_separable_conv2_d_wrapper1_data(void);
batch_normalization_layer_t init_batch_normalization_396_data(void);
quant_separable_conv2d_layer_t init_larq_quant_separable_conv2_d_wrapper2_data(void);
batch_normalization_layer_t init_batch_normalization_397_data(void);
dense_layer_t init_dense_80_data(void);


// Global Variables
conv2d_layer_t conv2d_4_data;
batch_normalization_layer_t batch_normalization_394_data;
quant_separable_conv2d_layer_t larq_quant_separable_conv2_d_wrapper_data;
batch_normalization_layer_t batch_normalization_395_data;
quant_separable_conv2d_layer_t larq_quant_separable_conv2_d_wrapper1_data;
batch_normalization_layer_t batch_normalization_396_data;
quant_separable_conv2d_layer_t larq_quant_separable_conv2_d_wrapper2_data;
batch_normalization_layer_t batch_normalization_397_data;
dense_layer_t dense_80_data;


void model_init(){
    conv2d_4_data = init_conv2d_4_data();
    batch_normalization_394_data = init_batch_normalization_394_data();
    larq_quant_separable_conv2_d_wrapper_data = init_larq_quant_separable_conv2_d_wrapper_data();
    batch_normalization_395_data = init_batch_normalization_395_data();
    larq_quant_separable_conv2_d_wrapper1_data = init_larq_quant_separable_conv2_d_wrapper1_data();
    batch_normalization_396_data = init_batch_normalization_396_data();
    larq_quant_separable_conv2_d_wrapper2_data = init_larq_quant_separable_conv2_d_wrapper2_data();
    batch_normalization_397_data = init_batch_normalization_397_data();
    dense_80_data = init_dense_80_data();

}

void model_predict(data3d_t input, data1d_t * output){
  
    prepare_buffers();
    
    //<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//
    // Layer name: channels_adapter
    data3d_t output0;
    channel_adapt_layer(input, &output0);
    
    //******************** LAYER 0 *******************//
    // Layer name: conv2d_4
    input = output0;
    conv2d_layer(conv2d_4_data, input, &output0);
    
    //******************** LAYER 1 *******************//
    // Layer name: conv2d_41
    
    
    //******************** LAYER 2 *******************//
    // Layer name: max_pooling2d_130
    input = output0;
    static const pooling2d_layer_t max_pooling2d_130_data = { 2, 2 };
    max_pooling2d_layer(max_pooling2d_130_data, input, &output0);
    
    //******************** LAYER 3 *******************//
    // Layer name: batch_normalization_394
    batch_normalization3d_layer(batch_normalization_394_data, &output0);
    
    //******************** LAYER 4 *******************//
    // Layer name: activation_401
    tanh_activation(output0.data, 3600);
    
    //******************** LAYER 5 *******************//
    // Layer name: larq_quant_separable_conv2_d_wrapper
    input = output0;
        quantSeparableConv2D_layer(larq_quant_separable_conv2_d_wrapper_data, input, &output0);
    
    
    //******************** LAYER 6 *******************//
    // Layer name: quant_separable_conv2d_300
    
    
    //******************** LAYER 7 *******************//
    // Layer name: batch_normalization_395
    batch_normalization3d_layer(batch_normalization_395_data, &output0);
    
    //******************** LAYER 8 *******************//
    // Layer name: activation_402
    tanh_activation(output0.data, 10816);
    
    //******************** LAYER 9 *******************//
    // Layer name: larq_quant_separable_conv2_d_wrapper1
    input = output0;
        quantSeparableConv2D_layer(larq_quant_separable_conv2_d_wrapper1_data, input, &output0);
    
    
    //******************** LAYER 10 *******************//
    // Layer name: quant_separable_conv2d_301
    
    
    //******************** LAYER 11 *******************//
    // Layer name: max_pooling2d_131
    input = output0;
    static const pooling2d_layer_t max_pooling2d_131_data = { 2, 2 };
    max_pooling2d_layer(max_pooling2d_131_data, input, &output0);
    
    //******************** LAYER 12 *******************//
    // Layer name: batch_normalization_396
    batch_normalization3d_layer(batch_normalization_396_data, &output0);
    
    //******************** LAYER 13 *******************//
    // Layer name: activation_403
    tanh_activation(output0.data, 2400);
    
    //******************** LAYER 14 *******************//
    // Layer name: larq_quant_separable_conv2_d_wrapper2
    input = output0;
        quantSeparableConv2D_layer(larq_quant_separable_conv2_d_wrapper2_data, input, &output0);
    
    
    //******************** LAYER 15 *******************//
    // Layer name: quant_separable_conv2d_302
    
    
    //******************** LAYER 16 *******************//
    // Layer name: average_pooling2d_19
    input = output0;
    static const pooling2d_layer_t average_pooling2d_19_data = { 3, 3 };
    avg_pooling2d_layer(average_pooling2d_19_data, input, &output0);
    
    //******************** LAYER 17 *******************//
    // Layer name: batch_normalization_397
    batch_normalization3d_layer(batch_normalization_397_data, &output0);
    
    //******************** LAYER 18 *******************//
    // Layer name: activation_404
    tanh_activation(output0.data, 512);
    
    //******************** LAYER 19 *******************//
    // Layer name: flatten_81
    input = output0;
    data1d_t output1;
    flatten3d_layer(input, &output1);
    
    //******************** LAYER 20 *******************//
    // Layer name: dropout_81
    data1d_t input1;
    input1 = output1;
    
    
    //******************** LAYER 21 *******************//
    // Layer name: dense_80
    input1 = output1;
    dense_layer(dense_80_data, input1, &output1);
    
    
    //******************** LAYER 22 *******************//
    // Layer name: dense_801
    
    
    //******************** LAYER 23 *******************//
    // Layer name: activation_405
    softmax_activation(output1.data, 10);
    

    *output = output1;

}

int model_predict_class(data3d_t input, data1d_t * results){
  
   
    model_predict(input, results);
    
    return argmax(*results);
    //return argmax(data1d_t);

}

// Implementation of initialization functions



conv2d_layer_t init_conv2d_4_data(void){

        static filter_t filters[16];
        
        static const fixed weights0[]={
           10786,    -12635,    -6056, /* [ 0.08228804 -0.09639557 -0.04620003] */
           2704,    -16360,    12571, /* [ 0.02062697 -0.12481505  0.09590933] */
           8511,    -253,    1161, /* [ 0.06493749 -0.00192997  0.00885862] */
           11332,    3662,    6233, /* [0.08645451 0.02793767 0.0475543 ] */
           22875,    3732,    10499, /* [0.17451881 0.02846949 0.08010041] */
           43662,    12994,    9819, /* [0.33311313 0.09913936 0.07491443] */
           -22742,    6781,    -4379, /* [-0.17350939  0.05173779 -0.03341234] */
           -1736,    -28933,    4434, /* [-0.01324396 -0.22073907  0.03383235] */
           -50485,    -8644,    -21727 /* [-0.38517028 -0.06594853 -0.16576371] */
        };
        static filter_t filter0 = { weights0, 615};  //0.004690014757215977
        filters[0]=filter0;
            
        static const fixed weights1[]={
           13384,    -16346,    -577, /* [ 0.10211329 -0.12470993 -0.00439851] */
           -7995,    15530,    27299, /* [-0.06099838  0.11848389  0.20827629] */
           4247,    -2013,    -25861, /* [ 0.03240259 -0.01535442 -0.19730066] */
           -18,    21220,    -18418, /* [-1.34628062e-04  1.61898106e-01 -1.40521482e-01] */
           -26190,    -15184,    24754, /* [-0.19981657 -0.11584361  0.18885715] */
           8394,    -8822,    -27194, /* [ 0.06403829 -0.06730986 -0.20747207] */
           12278,    -2545,    -22291, /* [ 0.09367457 -0.01941933 -0.17006795] */
           -17846,    -4293,    17582, /* [-0.13615423 -0.03275391  0.13413803] */
           8894,    9838,    10627 /* [0.06785587 0.07506065 0.08107941] */
        };
        static filter_t filter1 = { weights1, -3406};  //-0.02598356269299984
        filters[1]=filter1;
            
        static const fixed weights2[]={
           -22012,    4152,    -10055, /* [-0.16793938  0.03168069 -0.07671015] */
           -33863,    -5586,    -11906, /* [-0.25835061 -0.04262115 -0.09083451] */
           1001,    41015,    27937, /* [0.00764057 0.31291762 0.21314499] */
           30905,    23203,    2337, /* [0.23578645 0.177022   0.01782919] */
           -4708,    -13820,    -19426, /* [-0.03591581 -0.10543452 -0.14821096] */
           -2623,    4192,    -15536, /* [-0.02001038  0.03198579 -0.11853166] */
           5358,    15396,    23041, /* [0.04087684 0.11746351 0.17579207] */
           -7796,    -2936,    -25556, /* [-0.05947623 -0.02239987 -0.19497745] */
           5980,    -10123,    -2271 /* [ 0.04562316 -0.07723048 -0.01732742] */
        };
        static filter_t filter2 = { weights2, -3049};  //-0.02325921133160591
        filters[2]=filter2;
            
        static const fixed weights3[]={
           3958,    -3198,    47889, /* [ 0.03019676 -0.02440036  0.36536714] */
           -11638,    1486,    13826, /* [-0.08878727  0.01133588  0.1054827 ] */
           -10643,    4809,    -2995, /* [-0.08119975  0.03668797 -0.02284695] */
           8306,    -20094,    919, /* [ 0.06337292 -0.15330355  0.00701239] */
           10285,    9923,    -12807, /* [ 0.07847097  0.07570832 -0.09771154] */
           -13711,    12622,    3113, /* [-0.1046077   0.09629963  0.02375382] */
           -9982,    837,    -27574, /* [-0.07615583  0.0063838  -0.21037243] */
           20936,    -4060,    -28493, /* [ 0.15973133 -0.03097801 -0.21738444] */
           2810,    2971,    -261 /* [ 0.02143575  0.02267    -0.00199198] */
        };
        static filter_t filter3 = { weights3, -2053};  //-0.015665389597415924
        filters[3]=filter3;
            
        static const fixed weights4[]={
           10235,    -43925,    7127, /* [ 0.07808686 -0.33512387  0.05437323] */
           3820,    -15882,    7500, /* [ 0.02914229 -0.12116801  0.05722129] */
           -15180,    -24448,    -3946, /* [-0.11581441 -0.18652219 -0.0301034 ] */
           -17914,    22405,    -4556, /* [-0.13667162  0.1709336  -0.0347582 ] */
           30007,    15344,    14820, /* [0.22893605 0.11706513 0.11306778] */
           3260,    24822,    21300, /* [0.02487521 0.18937942 0.16250694] */
           5829,    4185,    -12692, /* [ 0.0444749   0.03193109 -0.09683395] */
           -16268,    2363,    5670, /* [-0.12411447  0.01803017  0.04325628] */
           13432,    -19972,    -18690 /* [ 0.10247809 -0.15237206 -0.14259081] */
        };
        static filter_t filter4 = { weights4, 2010};  //0.015336208045482635
        filters[4]=filter4;
            
        static const fixed weights5[]={
           5454,    -19071,    10813, /* [ 0.04161179 -0.14550085  0.08249944] */
           7971,    -5905,    8007, /* [ 0.06081399 -0.04505342  0.06108688] */
           -4095,    1607,    -7405, /* [-0.03123951  0.01226306 -0.05649837] */
           12829,    -12048,    -9026, /* [ 0.09787384 -0.09191826 -0.06886423] */
           12927,    -31765,    12476, /* [ 0.09862851 -0.2423498   0.09518626] */
           14795,    -38343,    20985, /* [ 0.11287878 -0.29253057  0.16010334] */
           3456,    -23897,    20960, /* [ 0.02636591 -0.18231826  0.15991166] */
           12616,    -11181,    9365, /* [ 0.09625604 -0.08530512  0.07144801] */
           14342,    -32993,    14052 /* [ 0.10942022 -0.2517184   0.10721128] */
        };
        static filter_t filter5 = { weights5, 292};  //0.002228367142379284
        filters[5]=filter5;
            
        static const fixed weights6[]={
           -904,    21134,    -12898, /* [-0.0068972   0.16123793 -0.09840334] */
           -19457,    23130,    -14904, /* [-0.14844438  0.17646475 -0.11370702] */
           -9182,    19526,    19630, /* [-0.07005379  0.14897068  0.14976484] */
           -807,    -1466,    -22217, /* [-0.00615327 -0.01118801 -0.16950606] */
           325,    27703,    -29466, /* [ 0.00247659  0.21135476 -0.22480962] */
           -8824,    30374,    -17088, /* [-0.06731913  0.23173687 -0.1303744 ] */
           -6057,    18843,    -13349, /* [-0.04621316  0.14375752 -0.10184236] */
           -5489,    20637,    3253, /* [-0.04187711  0.15744698  0.02481951] */
           -21657,    6067,    -12379 /* [-0.165226    0.04628922 -0.09444298] */
        };
        static filter_t filter6 = { weights6, 2166};  //0.0165290255099535
        filters[6]=filter6;
            
        static const fixed weights7[]={
           9952,    -3286,    11228, /* [ 0.07592779 -0.0250731   0.08566467] */
           -2840,    14814,    1541, /* [-0.02166785  0.11302012  0.01175871] */
           229,    -30275,    658, /* [ 0.00174443 -0.23097742  0.00501911] */
           -6974,    19737,    -26706, /* [-0.05320489  0.15058005 -0.20374699] */
           4205,    -9615,    -4322, /* [ 0.03207786 -0.07335863 -0.03297125] */
           -15343,    -9578,    -12670, /* [-0.11705447 -0.07307436 -0.09666412] */
           -32083,    -45983,    18229, /* [-0.24477129 -0.3508206   0.1390737 ] */
           -2213,    -12158,    9448, /* [-0.01688384 -0.09275685  0.07208054] */
           4975,    24475,    29463 /* [0.03795616 0.18672918 0.22478165] */
        };
        static filter_t filter7 = { weights7, 841};  //0.006412619259208441
        filters[7]=filter7;
            
        static const fixed weights8[]={
           29326,    -35854,    18562, /* [ 0.22373863 -0.273541    0.14161697] */
           -27193,    18790,    11494, /* [-0.20746866  0.14335872  0.08769561] */
           5966,    5260,    -3570, /* [ 0.04551625  0.04012812 -0.02723914] */
           18454,    -36332,    4489, /* [ 0.14079136 -0.27719462  0.03424738] */
           -7430,    17855,    -14596, /* [-0.05668613  0.13622254 -0.11135814] */
           -9018,    3239,    -11057, /* [-0.06880483  0.02471034 -0.08435514] */
           14986,    -29360,    -6886, /* [ 0.11433204 -0.22399813 -0.05253929] */
           12357,    8858,    7989, /* [0.09427871 0.06758083 0.06094882] */
           1470,    5175,    -4936 /* [ 0.01121422  0.039482   -0.03766105] */
        };
        static filter_t filter8 = { weights8, -1046};  //-0.0079775620251894
        filters[8]=filter8;
            
        static const fixed weights9[]={
           -8590,    -2951,    11182, /* [-0.06553872 -0.02251601  0.08531228] */
           -13372,    16780,    -6383, /* [-0.10202303  0.12802038 -0.04869568] */
           -23984,    -8267,    6472, /* [-0.18298352 -0.06307412  0.04938069] */
           -15199,    -9105,    32819, /* [-0.11595538 -0.06946827  0.25039178] */
           -1989,    17052,    9081, /* [-0.01517624  0.13009375  0.06928147] */
           -7035,    -5989,    10801, /* [-0.05367019 -0.04569005  0.08240823] */
           -28348,    -12612,    16808, /* [-0.21627556 -0.09622134  0.12823781] */
           -39069,    -1280,    25878, /* [-0.29807124 -0.00976371  0.19743179] */
           -19354,    14174,    22619 /* [-0.14766145  0.10813999  0.17256878] */
        };
        static filter_t filter9 = { weights9, 582};  //0.004438153468072414
        filters[9]=filter9;
            
        static const fixed weights10[]={
           -10954,    -8462,    -3885, /* [-0.08356974 -0.06455706 -0.02963909] */
           27682,    32144,    27959, /* [0.21119338 0.24523613 0.21330678] */
           -9975,    -28159,    -18235, /* [-0.07610131 -0.21483663 -0.13912019] */
           23418,    -19597,    14390, /* [ 0.178664   -0.14951634  0.10979053] */
           9464,    -7360,    7504, /* [ 0.07220171 -0.0561514   0.0572539 ] */
           -38848,    569,    -12300, /* [-0.29638651  0.00434277 -0.09383981] */
           -12405,    14404,    -14527, /* [-0.09464127  0.10989702 -0.11082949] */
           -12885,    3970,    -6964, /* [-0.09830819  0.03029098 -0.05313423] */
           17934,    17808,    1497 /* [0.13682376 0.1358636  0.01142209] */
        };
        static filter_t filter10 = { weights10, 1279};  //0.009756894782185555
        filters[10]=filter10;
            
        static const fixed weights11[]={
           -14862,    4473,    -8293, /* [-0.11338529  0.03412838 -0.06327087] */
           -1954,    39301,    -6765, /* [-0.01490776  0.29984063 -0.05161199] */
           18345,    16104,    -23078, /* [ 0.13995865  0.12286589 -0.17607248] */
           -35911,    13916,    -16451, /* [-0.2739791   0.10617372 -0.12551109] */
           17583,    23811,    -17603, /* [ 0.13414855  0.18166555 -0.1342971 ] */
           -17794,    9194,    -14794, /* [-0.1357605   0.07014161 -0.1128724 ] */
           -687,    13090,    27111, /* [-0.00524234  0.09986751  0.20684136] */
           -9078,    2584,    -152, /* [-0.0692599   0.01971629 -0.00116242] */
           -12817,    -18971,    -6950 /* [-0.09778727 -0.14473781 -0.05302323] */
        };
        static filter_t filter11 = { weights11, 392};  //0.0029919417575001717
        filters[11]=filter11;
            
        static const fixed weights12[]={
           3172,    -20420,    -36579, /* [ 0.02420161 -0.15579559 -0.27907425] */
           13656,    2080,    25666, /* [0.10418497 0.01587108 0.19581893] */
           -5773,    10171,    24782, /* [-0.04404765  0.07760029  0.18907523] */
           -20672,    -9309,    10162, /* [-0.15771583 -0.07102402  0.07753273] */
           1617,    7673,    -15192, /* [ 0.01233626  0.05853912 -0.11590529] */
           14221,    -14838,    -451, /* [ 0.10849415 -0.11320262 -0.00343927] */
           -3714,    23696,    24865, /* [-0.02833403  0.18078512  0.1897046 ] */
           222,    5389,    5275, /* [0.00169419 0.04111513 0.04024589] */
           543,    11626,    -22671 /* [ 0.00414256  0.08870301 -0.17296921] */
        };
        static filter_t filter12 = { weights12, 2338};  //0.01783960498869419
        filters[12]=filter12;
            
        static const fixed weights13[]={
           -8922,    -4612,    3982, /* [-0.06806991 -0.03518535  0.03037935] */
           25041,    2028,    -17445, /* [ 0.19104797  0.01546935 -0.13309143] */
           -15558,    -2247,    18674, /* [-0.11869738 -0.01714441  0.14246757] */
           -20060,    20540,    16926, /* [-0.15304451  0.15670902  0.1291365 ] */
           9509,    6965,    -37749, /* [ 0.07254715  0.0531365  -0.28800398] */
           8122,    -15002,    8683, /* [ 0.06196538 -0.11445916  0.06624334] */
           -3557,    -25948,    10990, /* [-0.02714    -0.19796464  0.08384965] */
           16554,    25293,    -29571, /* [ 0.12629592  0.19297296 -0.22560944] */
           -16517,    -10467,    30833 /* [-0.12601188 -0.07985393  0.23523915] */
        };
        static filter_t filter13 = { weights13, 821};  //0.00626011099666357
        filters[13]=filter13;
            
        static const fixed weights14[]={
           23244,    -10772,    32252, /* [ 0.17733575 -0.08218385  0.24606639] */
           -454,    33694,    1561, /* [-0.00346256  0.25706127  0.01190945] */
           11330,    -21374,    -31831, /* [ 0.08644083 -0.16307291 -0.24285112] */
           23224,    -961,    -5503, /* [ 0.17718264 -0.00733132 -0.04198332] */
           -4816,    13958,    18761, /* [-0.03674389  0.10649218  0.14313512] */
           -16141,    -20005,    -7676, /* [-0.12314546 -0.15262458 -0.05856122] */
           277,    -12455,    13376, /* [ 0.00211208 -0.09502789  0.10205114] */
           -9532,    7653,    -4310, /* [-0.07272452  0.05839076 -0.03287901] */
           4982,    4444,    -14505 /* [ 0.03800736  0.03390756 -0.11066172] */
        };
        static filter_t filter14 = { weights14, 1258};  //0.009596770629286766
        filters[14]=filter14;
            
        static const fixed weights15[]={
           3561,    -32335,    19231, /* [ 0.02716806 -0.24669607  0.1467243 ] */
           6896,    5400,    -8239, /* [ 0.05261041  0.04119669 -0.06286038] */
           -10237,    8119,    -7920, /* [-0.07810066  0.06194629 -0.06042365] */
           10409,    -34976,    15644, /* [ 0.07941245 -0.26684728  0.1193547 ] */
           16,    33661,    -13467, /* [ 1.19263830e-04  2.56816119e-01 -1.02744229e-01] */
           -1364,    -7523,    15464, /* [-0.01040565 -0.05739953  0.11798129] */
           2422,    -16245,    -13260, /* [ 0.01848017 -0.12393809 -0.10116377] */
           -19893,    39696,    -5666, /* [-0.15176821  0.30285993 -0.04322505] */
           5528,    1555,    -682 /* [ 0.04217758  0.01186136 -0.00520491] */
        };
        static filter_t filter15 = { weights15, -1958};  //-0.0149376280605793
        filters[15]=filter15;
            
        conv2d_layer_t layer = {16, filters, 3, { 3, 3 }, 0, {1, 1} };
        return layer;
}
        
batch_normalization_layer_t init_batch_normalization_394_data(void){

    static const fixed inv_gamma_dev[] ={
    1217455, 1357074, 2360605, 919025, 1156108, 2445063, 365473, 740771, 1611300, 733958, 
    2142139, 693654, 1223667, 3324934, 433328, 2963324
    };
    static const fixed std_beta[] ={
    -167924, -55730, -228314, -860, -73891, -384027, -66646, -188801, -23709, -243858, 
    -277497, -94186, -165765, -278179, -112859, -39392
    };

    static const batch_normalization_layer_t norm = { 16, inv_gamma_dev, std_beta  };
    return norm;
}


            quant_separable_conv2d_layer_t init_larq_quant_separable_conv2_d_wrapper_data(void){

            
            static const uint32_t depth_weights[]={
    
    
    1300234240,
  
    
    
    2273312768,
  
    
    
    1241513984,
  
    
    
    310378496,
  
    
    
    117440512,
  
    
    
    2717908992,
  
    
    
    3523215360,
  
    
    
    2323644416,
  
    
    
    1291845632,
  
    
    
    117440512,
  
    
    
    1082130432,
  
    
    
    746586112,
  
    
    
    2315255808,
  
    
    
    3850371072,
  
    
    
    1904214016,
  
    
    
    2189426688,
  
            };
            static quant_filter_t depth_filter_b = {16, 3, depth_weights, 0};

            static quant_filter_t point_filters_b[64];
            
            static const uint32_t point_weights0[]={3934846976,
            };
            static quant_filter_t point_filter0 = {16, 1, point_weights0, -1.9114568203804083e-06};
            point_filters_b[0] = point_filter0;
            
            static const uint32_t point_weights1[]={1989476352,
            };
            static quant_filter_t point_filter1 = {16, 1, point_weights1, 5.864599734195508e-05};
            point_filters_b[1] = point_filter1;
            
            static const uint32_t point_weights2[]={4267376640,
            };
            static quant_filter_t point_filter2 = {16, 1, point_weights2, -0.00010940575157292187};
            point_filters_b[2] = point_filter2;
            
            static const uint32_t point_weights3[]={1594556416,
            };
            static quant_filter_t point_filter3 = {16, 1, point_weights3, 0.00013222251436673105};
            point_filters_b[3] = point_filter3;
            
            static const uint32_t point_weights4[]={3842768896,
            };
            static quant_filter_t point_filter4 = {16, 1, point_weights4, 0.00014769332483410835};
            point_filters_b[4] = point_filter4;
            
            static const uint32_t point_weights5[]={2323185664,
            };
            static quant_filter_t point_filter5 = {16, 1, point_weights5, 0.00011229305528104305};
            point_filters_b[5] = point_filter5;
            
            static const uint32_t point_weights6[]={367263744,
            };
            static quant_filter_t point_filter6 = {16, 1, point_weights6, -1.3614139788842294e-05};
            point_filters_b[6] = point_filter6;
            
            static const uint32_t point_weights7[]={1343356928,
            };
            static quant_filter_t point_filter7 = {16, 1, point_weights7, -3.944483250961639e-05};
            point_filters_b[7] = point_filter7;
            
            static const uint32_t point_weights8[]={629669888,
            };
            static quant_filter_t point_filter8 = {16, 1, point_weights8, -4.63791293441318e-05};
            point_filters_b[8] = point_filter8;
            
            static const uint32_t point_weights9[]={2822569984,
            };
            static quant_filter_t point_filter9 = {16, 1, point_weights9, -5.199246515985578e-06};
            point_filters_b[9] = point_filter9;
            
            static const uint32_t point_weights10[]={79888384,
            };
            static quant_filter_t point_filter10 = {16, 1, point_weights10, 0.00030810374300926924};
            point_filters_b[10] = point_filter10;
            
            static const uint32_t point_weights11[]={2570059776,
            };
            static quant_filter_t point_filter11 = {16, 1, point_weights11, 0.00025902281049638987};
            point_filters_b[11] = point_filter11;
            
            static const uint32_t point_weights12[]={3153788928,
            };
            static quant_filter_t point_filter12 = {16, 1, point_weights12, 2.911651426984463e-05};
            point_filters_b[12] = point_filter12;
            
            static const uint32_t point_weights13[]={3451453440,
            };
            static quant_filter_t point_filter13 = {16, 1, point_weights13, 0.00015398519462905824};
            point_filters_b[13] = point_filter13;
            
            static const uint32_t point_weights14[]={3715891200,
            };
            static quant_filter_t point_filter14 = {16, 1, point_weights14, -1.4673311852675397e-05};
            point_filters_b[14] = point_filter14;
            
            static const uint32_t point_weights15[]={4294115328,
            };
            static quant_filter_t point_filter15 = {16, 1, point_weights15, -0.00010982093954226002};
            point_filters_b[15] = point_filter15;
            
            static const uint32_t point_weights16[]={1051918336,
            };
            static quant_filter_t point_filter16 = {16, 1, point_weights16, 0.0001191798728541471};
            point_filters_b[16] = point_filter16;
            
            static const uint32_t point_weights17[]={2612133888,
            };
            static quant_filter_t point_filter17 = {16, 1, point_weights17, -3.533227936713956e-05};
            point_filters_b[17] = point_filter17;
            
            static const uint32_t point_weights18[]={18284544,
            };
            static quant_filter_t point_filter18 = {16, 1, point_weights18, 3.7726626032963395e-05};
            point_filters_b[18] = point_filter18;
            
            static const uint32_t point_weights19[]={2030043136,
            };
            static quant_filter_t point_filter19 = {16, 1, point_weights19, 4.469027044251561e-05};
            point_filters_b[19] = point_filter19;
            
            static const uint32_t point_weights20[]={2105081856,
            };
            static quant_filter_t point_filter20 = {16, 1, point_weights20, -6.750765169272199e-05};
            point_filters_b[20] = point_filter20;
            
            static const uint32_t point_weights21[]={651231232,
            };
            static quant_filter_t point_filter21 = {16, 1, point_weights21, 9.043109457707033e-05};
            point_filters_b[21] = point_filter21;
            
            static const uint32_t point_weights22[]={281608192,
            };
            static quant_filter_t point_filter22 = {16, 1, point_weights22, 3.724572161445394e-05};
            point_filters_b[22] = point_filter22;
            
            static const uint32_t point_weights23[]={2976448512,
            };
            static quant_filter_t point_filter23 = {16, 1, point_weights23, -0.0001496763143222779};
            point_filters_b[23] = point_filter23;
            
            static const uint32_t point_weights24[]={158990336,
            };
            static quant_filter_t point_filter24 = {16, 1, point_weights24, 7.849071698728949e-05};
            point_filters_b[24] = point_filter24;
            
            static const uint32_t point_weights25[]={2552496128,
            };
            static quant_filter_t point_filter25 = {16, 1, point_weights25, -0.00011245145287830383};
            point_filters_b[25] = point_filter25;
            
            static const uint32_t point_weights26[]={1985085440,
            };
            static quant_filter_t point_filter26 = {16, 1, point_weights26, -0.00022235374490264803};
            point_filters_b[26] = point_filter26;
            
            static const uint32_t point_weights27[]={3700228096,
            };
            static quant_filter_t point_filter27 = {16, 1, point_weights27, -1.819624776544515e-05};
            point_filters_b[27] = point_filter27;
            
            static const uint32_t point_weights28[]={4084596736,
            };
            static quant_filter_t point_filter28 = {16, 1, point_weights28, -1.3667968232766725e-06};
            point_filters_b[28] = point_filter28;
            
            static const uint32_t point_weights29[]={3114532864,
            };
            static quant_filter_t point_filter29 = {16, 1, point_weights29, -5.486210648086853e-05};
            point_filters_b[29] = point_filter29;
            
            static const uint32_t point_weights30[]={802357248,
            };
            static quant_filter_t point_filter30 = {16, 1, point_weights30, 9.935826528817415e-05};
            point_filters_b[30] = point_filter30;
            
            static const uint32_t point_weights31[]={2379743232,
            };
            static quant_filter_t point_filter31 = {16, 1, point_weights31, -3.002185439981986e-05};
            point_filters_b[31] = point_filter31;
            
            static const uint32_t point_weights32[]={262602752,
            };
            static quant_filter_t point_filter32 = {16, 1, point_weights32, 5.938534013694152e-05};
            point_filters_b[32] = point_filter32;
            
            static const uint32_t point_weights33[]={3757113344,
            };
            static quant_filter_t point_filter33 = {16, 1, point_weights33, 0.00022995077597443014};
            point_filters_b[33] = point_filter33;
            
            static const uint32_t point_weights34[]={1790443520,
            };
            static quant_filter_t point_filter34 = {16, 1, point_weights34, 4.820654066861607e-05};
            point_filters_b[34] = point_filter34;
            
            static const uint32_t point_weights35[]={2735210496,
            };
            static quant_filter_t point_filter35 = {16, 1, point_weights35, -0.00011398330389056355};
            point_filters_b[35] = point_filter35;
            
            static const uint32_t point_weights36[]={1749221376,
            };
            static quant_filter_t point_filter36 = {16, 1, point_weights36, -0.0001777698053047061};
            point_filters_b[36] = point_filter36;
            
            static const uint32_t point_weights37[]={2694905856,
            };
            static quant_filter_t point_filter37 = {16, 1, point_weights37, -0.0004398952005431056};
            point_filters_b[37] = point_filter37;
            
            static const uint32_t point_weights38[]={1428488192,
            };
            static quant_filter_t point_filter38 = {16, 1, point_weights38, -0.0003183860389981419};
            point_filters_b[38] = point_filter38;
            
            static const uint32_t point_weights39[]={405536768,
            };
            static quant_filter_t point_filter39 = {16, 1, point_weights39, 2.718296309467405e-05};
            point_filters_b[39] = point_filter39;
            
            static const uint32_t point_weights40[]={871628800,
            };
            static quant_filter_t point_filter40 = {16, 1, point_weights40, 1.2229882486280985e-05};
            point_filters_b[40] = point_filter40;
            
            static const uint32_t point_weights41[]={1516371968,
            };
            static quant_filter_t point_filter41 = {16, 1, point_weights41, -5.789747228845954e-05};
            point_filters_b[41] = point_filter41;
            
            static const uint32_t point_weights42[]={3031236608,
            };
            static quant_filter_t point_filter42 = {16, 1, point_weights42, 4.746984632220119e-05};
            point_filters_b[42] = point_filter42;
            
            static const uint32_t point_weights43[]={3851943936,
            };
            static quant_filter_t point_filter43 = {16, 1, point_weights43, -4.1960211092373356e-05};
            point_filters_b[43] = point_filter43;
            
            static const uint32_t point_weights44[]={4118020096,
            };
            static quant_filter_t point_filter44 = {16, 1, point_weights44, 0.00013552102609537542};
            point_filters_b[44] = point_filter44;
            
            static const uint32_t point_weights45[]={3276996608,
            };
            static quant_filter_t point_filter45 = {16, 1, point_weights45, -7.782935426803306e-05};
            point_filters_b[45] = point_filter45;
            
            static const uint32_t point_weights46[]={428802048,
            };
            static quant_filter_t point_filter46 = {16, 1, point_weights46, 2.3105805667000823e-05};
            point_filters_b[46] = point_filter46;
            
            static const uint32_t point_weights47[]={201261056,
            };
            static quant_filter_t point_filter47 = {16, 1, point_weights47, 2.4252405637525953e-05};
            point_filters_b[47] = point_filter47;
            
            static const uint32_t point_weights48[]={3505717248,
            };
            static quant_filter_t point_filter48 = {16, 1, point_weights48, 9.883695020107552e-05};
            point_filters_b[48] = point_filter48;
            
            static const uint32_t point_weights49[]={2776563712,
            };
            static quant_filter_t point_filter49 = {16, 1, point_weights49, 0.00023823819356039166};
            point_filters_b[49] = point_filter49;
            
            static const uint32_t point_weights50[]={1212809216,
            };
            static quant_filter_t point_filter50 = {16, 1, point_weights50, -4.4666125177172944e-05};
            point_filters_b[50] = point_filter50;
            
            static const uint32_t point_weights51[]={920715264,
            };
            static quant_filter_t point_filter51 = {16, 1, point_weights51, -5.16112704644911e-05};
            point_filters_b[51] = point_filter51;
            
            static const uint32_t point_weights52[]={3232366592,
            };
            static quant_filter_t point_filter52 = {16, 1, point_weights52, 1.8016202375292778e-05};
            point_filters_b[52] = point_filter52;
            
            static const uint32_t point_weights53[]={1726611456,
            };
            static quant_filter_t point_filter53 = {16, 1, point_weights53, 5.985629468341358e-05};
            point_filters_b[53] = point_filter53;
            
            static const uint32_t point_weights54[]={2674327552,
            };
            static quant_filter_t point_filter54 = {16, 1, point_weights54, -1.614650864212308e-05};
            point_filters_b[54] = point_filter54;
            
            static const uint32_t point_weights55[]={1336213504,
            };
            static quant_filter_t point_filter55 = {16, 1, point_weights55, -4.119361256016418e-05};
            point_filters_b[55] = point_filter55;
            
            static const uint32_t point_weights56[]={3594715136,
            };
            static quant_filter_t point_filter56 = {16, 1, point_weights56, -0.0001536144409328699};
            point_filters_b[56] = point_filter56;
            
            static const uint32_t point_weights57[]={972750848,
            };
            static quant_filter_t point_filter57 = {16, 1, point_weights57, -9.078490984393284e-05};
            point_filters_b[57] = point_filter57;
            
            static const uint32_t point_weights58[]={2263089152,
            };
            static quant_filter_t point_filter58 = {16, 1, point_weights58, 4.419885954121128e-05};
            point_filters_b[58] = point_filter58;
            
            static const uint32_t point_weights59[]={1485111296,
            };
            static quant_filter_t point_filter59 = {16, 1, point_weights59, -8.903845446184278e-05};
            point_filters_b[59] = point_filter59;
            
            static const uint32_t point_weights60[]={3164536832,
            };
            static quant_filter_t point_filter60 = {16, 1, point_weights60, 6.868227046652464e-06};
            point_filters_b[60] = point_filter60;
            
            static const uint32_t point_weights61[]={1812463616,
            };
            static quant_filter_t point_filter61 = {16, 1, point_weights61, -6.958672747714445e-05};
            point_filters_b[61] = point_filter61;
            
            static const uint32_t point_weights62[]={1618870272,
            };
            static quant_filter_t point_filter62 = {16, 1, point_weights62, 1.3084872080071364e-05};
            point_filters_b[62] = point_filter62;
            
            static const uint32_t point_weights63[]={1767178240,
            };
            static quant_filter_t point_filter63 = {16, 1, point_weights63, 7.989905861904845e-05};
            point_filters_b[63] = point_filter63;
            
            quant_separable_conv2d_layer_t layer = {64, depth_filter_b, point_filters_b};
            return layer;
            }
            
batch_normalization_layer_t init_batch_normalization_395_data(void){

    static const fixed inv_gamma_dev[] ={
    734, 7824, 9562, 4826, 2364, 83, 1105, -3906, 2556, 14035, 10047, 14028, 5787, 14076, 
    128, 10296, 13504, 817, -711, 18744, 6424, 18073, 24525, 29317, 834, 1939, 13222, 
    686, 249, 17945, 12694, 4840, 12388, 9868, 25753, 17426, 14734, 22283, 9158, 10497, 
    1759, -1645, 21711, 635, 18268, 21500, 10521, 38, 16326, 7910, 3197, 13634, 14878, 
    -804, 1841, -2203, 22385, 14135, 1528, 10474, -217, 14501, 1683, 1643
    };
    static const fixed std_beta[] ={
    140656, 46030, -56799, 23422, 265, -1889, 3384, -21056, -67910, 1685, 73089, 27513, 
    -52907, -55283, -3279, -103579, -1563, -28873, -19042, 32010, -77088, -104768, 133351, 
    -167733, -12936, -9251, 52118, 13361, -2987, 71050, -50666, -105873, 3141, 18818, 
    -154801, -38115, -52489, 1600, 13965, -4927, 3327, 22803, -34, 12573, -36184, 14199, 
    82015, 3421, -7070, -64771, -609, 25384, -60387, 21967, -212524, 19999, 92009, -154071, 
    6064, 103814, 448, -48077, 49213, 26373
    };

    static const batch_normalization_layer_t norm = { 64, inv_gamma_dev, std_beta  };
    return norm;
}


            quant_separable_conv2d_layer_t init_larq_quant_separable_conv2_d_wrapper1_data(void){

            
            static const uint32_t depth_weights[]={
    
    
    1333788672,
  
    
    
    2986344448,
  
    
    
    1073741824,
  
    
    
    192937984,
  
    
    
    3263168512,
  
    
    
    738197504,
  
    
    
    2256535552,
  
    
    
    1451229184,
  
    
    
    1912602624,
  
    
    
    260046848,
  
    
    
    1702887424,
  
    
    
    50331648,
  
    
    
    1845493760,
  
    
    
    1535115264,
  
    
    
    3019898880,
  
    
    
    494927872,
  
    
    
    2348810240,
  
    
    
    2323644416,
  
    
    
    2793406464,
  
    
    
    3724541952,
  
    
    
    2256535552,
  
    
    
    3254779904,
  
    
    
    3145728000,
  
    
    
    4143972352,
  
    
    
    2189426688,
  
    
    
    1543503872,
  
    
    
    50331648,
  
    
    
    1560281088,
  
    
    
    3456106496,
  
    
    
    1837105152,
  
    
    
    3858759680,
  
    
    
    3732930560,
  
    
    
    1677721600,
  
    
    
    3078619136,
  
    
    
    4143972352,
  
    
    
    2139095040,
  
    
    
    1140850688,
  
    
    
    16777216,
  
    
    
    2164260864,
  
    
    
    226492416,
  
    
    
    4060086272,
  
    
    
    3330277376,
  
    
    
    3254779904,
  
    
    
    3506438144,
  
    
    
    4253024256,
  
    
    
    3758096384,
  
    
    
    2415919104,
  
    
    
    3808428032,
  
    
    
    637534208,
  
    
    
    117440512,
  
    
    
    3682598912,
  
    
    
    2214592512,
  
    
    
    3087007744,
  
    
    
    2826960896,
  
    
    
    998244352,
  
    
    
    3430940672,
  
    
    
    285212672,
  
    
    
    2197815296,
  
    
    
    3405774848,
  
    
    
    142606336,
  
    
    
    662700032,
  
    
    
    218103808,
  
    
    
    2961178624,
  
    
    
    1954545664,
  
            };
            static quant_filter_t depth_filter_b = {64, 3, depth_weights, 0};

            static quant_filter_t point_filters_b[96];
            
            static const uint32_t point_weights0[]={2447258785,4636480,
            };
            static quant_filter_t point_filter0 = {64, 1, point_weights0, 3.802917944994988e-06};
            point_filters_b[0] = point_filter0;
            
            static const uint32_t point_weights1[]={1605179206,4290192924,
            };
            static quant_filter_t point_filter1 = {64, 1, point_weights1, -2.666166255949065e-05};
            point_filters_b[1] = point_filter1;
            
            static const uint32_t point_weights2[]={957565065,268833175,
            };
            static quant_filter_t point_filter2 = {64, 1, point_weights2, -2.4053386368905194e-05};
            point_filters_b[2] = point_filter2;
            
            static const uint32_t point_weights3[]={388344808,488175386,
            };
            static quant_filter_t point_filter3 = {64, 1, point_weights3, 2.602060885692481e-05};
            point_filters_b[3] = point_filter3;
            
            static const uint32_t point_weights4[]={793021342,232546012,
            };
            static quant_filter_t point_filter4 = {64, 1, point_weights4, -7.861572157707997e-06};
            point_filters_b[4] = point_filter4;
            
            static const uint32_t point_weights5[]={4071061897,3375806891,
            };
            static quant_filter_t point_filter5 = {64, 1, point_weights5, 1.1419471775298007e-05};
            point_filters_b[5] = point_filter5;
            
            static const uint32_t point_weights6[]={3070240826,221349829,
            };
            static quant_filter_t point_filter6 = {64, 1, point_weights6, 1.6185185813810676e-05};
            point_filters_b[6] = point_filter6;
            
            static const uint32_t point_weights7[]={2698549296,1713433381,
            };
            static quant_filter_t point_filter7 = {64, 1, point_weights7, 1.8928712961496785e-05};
            point_filters_b[7] = point_filter7;
            
            static const uint32_t point_weights8[]={3369291543,4276716726,
            };
            static quant_filter_t point_filter8 = {64, 1, point_weights8, 3.074922460655216e-06};
            point_filters_b[8] = point_filter8;
            
            static const uint32_t point_weights9[]={3805801430,850659074,
            };
            static quant_filter_t point_filter9 = {64, 1, point_weights9, 2.316045902261976e-05};
            point_filters_b[9] = point_filter9;
            
            static const uint32_t point_weights10[]={3501082430,3951540895,
            };
            static quant_filter_t point_filter10 = {64, 1, point_weights10, -8.429996523773298e-05};
            point_filters_b[10] = point_filter10;
            
            static const uint32_t point_weights11[]={2807696163,3130491610,
            };
            static quant_filter_t point_filter11 = {64, 1, point_weights11, -2.9500932214432396e-05};
            point_filters_b[11] = point_filter11;
            
            static const uint32_t point_weights12[]={1497735733,2987296036,
            };
            static quant_filter_t point_filter12 = {64, 1, point_weights12, -2.99847101814521e-06};
            point_filters_b[12] = point_filter12;
            
            static const uint32_t point_weights13[]={395602276,872265936,
            };
            static quant_filter_t point_filter13 = {64, 1, point_weights13, 7.326875220314832e-06};
            point_filters_b[13] = point_filter13;
            
            static const uint32_t point_weights14[]={3646664789,2594173457,
            };
            static quant_filter_t point_filter14 = {64, 1, point_weights14, 1.0240674782835413e-05};
            point_filters_b[14] = point_filter14;
            
            static const uint32_t point_weights15[]={3502610150,519687859,
            };
            static quant_filter_t point_filter15 = {64, 1, point_weights15, 8.048278687056154e-06};
            point_filters_b[15] = point_filter15;
            
            static const uint32_t point_weights16[]={1889760747,690151680,
            };
            static quant_filter_t point_filter16 = {64, 1, point_weights16, -1.7524629583931528e-05};
            point_filters_b[16] = point_filter16;
            
            static const uint32_t point_weights17[]={4119676127,324999663,
            };
            static quant_filter_t point_filter17 = {64, 1, point_weights17, 3.930285311071202e-05};
            point_filters_b[17] = point_filter17;
            
            static const uint32_t point_weights18[]={2612467726,2484254805,
            };
            static quant_filter_t point_filter18 = {64, 1, point_weights18, 4.3735079088946804e-05};
            point_filters_b[18] = point_filter18;
            
            static const uint32_t point_weights19[]={2251549074,3542930340,
            };
            static quant_filter_t point_filter19 = {64, 1, point_weights19, 1.89229231182253e-05};
            point_filters_b[19] = point_filter19;
            
            static const uint32_t point_weights20[]={572296624,857653969,
            };
            static quant_filter_t point_filter20 = {64, 1, point_weights20, 2.3903578039607964e-05};
            point_filters_b[20] = point_filter20;
            
            static const uint32_t point_weights21[]={3680075211,1033269281,
            };
            static quant_filter_t point_filter21 = {64, 1, point_weights21, -5.994508228468476e-06};
            point_filters_b[21] = point_filter21;
            
            static const uint32_t point_weights22[]={1158685024,2492443883,
            };
            static quant_filter_t point_filter22 = {64, 1, point_weights22, 9.918091382132843e-05};
            point_filters_b[22] = point_filter22;
            
            static const uint32_t point_weights23[]={627143380,3029656210,
            };
            static quant_filter_t point_filter23 = {64, 1, point_weights23, -2.167544334952254e-05};
            point_filters_b[23] = point_filter23;
            
            static const uint32_t point_weights24[]={463469520,3890989737,
            };
            static quant_filter_t point_filter24 = {64, 1, point_weights24, 9.330242392024957e-07};
            point_filters_b[24] = point_filter24;
            
            static const uint32_t point_weights25[]={4068855880,717134539,
            };
            static quant_filter_t point_filter25 = {64, 1, point_weights25, 3.6609308153856546e-05};
            point_filters_b[25] = point_filter25;
            
            static const uint32_t point_weights26[]={548933625,64310244,
            };
            static quant_filter_t point_filter26 = {64, 1, point_weights26, -3.130350523861125e-05};
            point_filters_b[26] = point_filter26;
            
            static const uint32_t point_weights27[]={2979560696,2409665136,
            };
            static quant_filter_t point_filter27 = {64, 1, point_weights27, -7.69995094742626e-05};
            point_filters_b[27] = point_filter27;
            
            static const uint32_t point_weights28[]={3022049113,3832576336,
            };
            static quant_filter_t point_filter28 = {64, 1, point_weights28, 6.493594992207363e-05};
            point_filters_b[28] = point_filter28;
            
            static const uint32_t point_weights29[]={1723936708,2544757830,
            };
            static quant_filter_t point_filter29 = {64, 1, point_weights29, -2.7763440812123008e-05};
            point_filters_b[29] = point_filter29;
            
            static const uint32_t point_weights30[]={682521186,616861222,
            };
            static quant_filter_t point_filter30 = {64, 1, point_weights30, -2.5490155621810118e-06};
            point_filters_b[30] = point_filter30;
            
            static const uint32_t point_weights31[]={3122723283,3806608921,
            };
            static quant_filter_t point_filter31 = {64, 1, point_weights31, 7.087571430020034e-05};
            point_filters_b[31] = point_filter31;
            
            static const uint32_t point_weights32[]={2495234178,3606497320,
            };
            static quant_filter_t point_filter32 = {64, 1, point_weights32, 5.1820450607920066e-05};
            point_filters_b[32] = point_filter32;
            
            static const uint32_t point_weights33[]={667835167,34492465,
            };
            static quant_filter_t point_filter33 = {64, 1, point_weights33, -6.112546998338075e-06};
            point_filters_b[33] = point_filter33;
            
            static const uint32_t point_weights34[]={3591134395,3000609662,
            };
            static quant_filter_t point_filter34 = {64, 1, point_weights34, -2.007344846788328e-05};
            point_filters_b[34] = point_filter34;
            
            static const uint32_t point_weights35[]={543232682,2486053493,
            };
            static quant_filter_t point_filter35 = {64, 1, point_weights35, -1.070239522960037e-05};
            point_filters_b[35] = point_filter35;
            
            static const uint32_t point_weights36[]={1418401495,853264129,
            };
            static quant_filter_t point_filter36 = {64, 1, point_weights36, -1.6979323845589533e-05};
            point_filters_b[36] = point_filter36;
            
            static const uint32_t point_weights37[]={423219735,1114327821,
            };
            static quant_filter_t point_filter37 = {64, 1, point_weights37, -1.789226007531397e-05};
            point_filters_b[37] = point_filter37;
            
            static const uint32_t point_weights38[]={3537523685,2181969320,
            };
            static quant_filter_t point_filter38 = {64, 1, point_weights38, 2.883426168409642e-05};
            point_filters_b[38] = point_filter38;
            
            static const uint32_t point_weights39[]={3872556971,503061016,
            };
            static quant_filter_t point_filter39 = {64, 1, point_weights39, 2.978460179292597e-05};
            point_filters_b[39] = point_filter39;
            
            static const uint32_t point_weights40[]={1192844973,1894122468,
            };
            static quant_filter_t point_filter40 = {64, 1, point_weights40, 2.237125409010332e-05};
            point_filters_b[40] = point_filter40;
            
            static const uint32_t point_weights41[]={3691782572,3113941999,
            };
            static quant_filter_t point_filter41 = {64, 1, point_weights41, -2.5210709281964228e-05};
            point_filters_b[41] = point_filter41;
            
            static const uint32_t point_weights42[]={3416873701,2725182579,
            };
            static quant_filter_t point_filter42 = {64, 1, point_weights42, -3.372091305209324e-05};
            point_filters_b[42] = point_filter42;
            
            static const uint32_t point_weights43[]={1802429764,1572121883,
            };
            static quant_filter_t point_filter43 = {64, 1, point_weights43, -2.6235000404994935e-05};
            point_filters_b[43] = point_filter43;
            
            static const uint32_t point_weights44[]={2289937239,4117038459,
            };
            static quant_filter_t point_filter44 = {64, 1, point_weights44, 8.727171007194556e-06};
            point_filters_b[44] = point_filter44;
            
            static const uint32_t point_weights45[]={3030713913,3996318195,
            };
            static quant_filter_t point_filter45 = {64, 1, point_weights45, 0.000104468286735937};
            point_filters_b[45] = point_filter45;
            
            static const uint32_t point_weights46[]={1519776114,1023308365,
            };
            static quant_filter_t point_filter46 = {64, 1, point_weights46, 3.71231471945066e-05};
            point_filters_b[46] = point_filter46;
            
            static const uint32_t point_weights47[]={2622187870,2227036932,
            };
            static quant_filter_t point_filter47 = {64, 1, point_weights47, 4.048793471156387e-06};
            point_filters_b[47] = point_filter47;
            
            static const uint32_t point_weights48[]={2194927252,730346217,
            };
            static quant_filter_t point_filter48 = {64, 1, point_weights48, -1.2791735571227036e-05};
            point_filters_b[48] = point_filter48;
            
            static const uint32_t point_weights49[]={2394754071,89833926,
            };
            static quant_filter_t point_filter49 = {64, 1, point_weights49, 9.895674338622484e-06};
            point_filters_b[49] = point_filter49;
            
            static const uint32_t point_weights50[]={2313051337,3215974568,
            };
            static quant_filter_t point_filter50 = {64, 1, point_weights50, 3.720280437846668e-05};
            point_filters_b[50] = point_filter50;
            
            static const uint32_t point_weights51[]={1084072241,3014834760,
            };
            static quant_filter_t point_filter51 = {64, 1, point_weights51, -1.0050480341305956e-05};
            point_filters_b[51] = point_filter51;
            
            static const uint32_t point_weights52[]={1644777713,681052229,
            };
            static quant_filter_t point_filter52 = {64, 1, point_weights52, 5.277550735627301e-05};
            point_filters_b[52] = point_filter52;
            
            static const uint32_t point_weights53[]={1807052776,64407928,
            };
            static quant_filter_t point_filter53 = {64, 1, point_weights53, -2.40495937759988e-05};
            point_filters_b[53] = point_filter53;
            
            static const uint32_t point_weights54[]={449737871,1796276535,
            };
            static quant_filter_t point_filter54 = {64, 1, point_weights54, 1.7346788808936253e-05};
            point_filters_b[54] = point_filter54;
            
            static const uint32_t point_weights55[]={2619414185,1355004390,
            };
            static quant_filter_t point_filter55 = {64, 1, point_weights55, 4.1202642023563385e-05};
            point_filters_b[55] = point_filter55;
            
            static const uint32_t point_weights56[]={3583499045,424990551,
            };
            static quant_filter_t point_filter56 = {64, 1, point_weights56, -7.94257994130021e-06};
            point_filters_b[56] = point_filter56;
            
            static const uint32_t point_weights57[]={35316190,2918882575,
            };
            static quant_filter_t point_filter57 = {64, 1, point_weights57, 8.076782251009718e-05};
            point_filters_b[57] = point_filter57;
            
            static const uint32_t point_weights58[]={1878448651,3070999574,
            };
            static quant_filter_t point_filter58 = {64, 1, point_weights58, -3.1785708415554836e-05};
            point_filters_b[58] = point_filter58;
            
            static const uint32_t point_weights59[]={567538283,422955013,
            };
            static quant_filter_t point_filter59 = {64, 1, point_weights59, 6.069421942811459e-05};
            point_filters_b[59] = point_filter59;
            
            static const uint32_t point_weights60[]={1107548796,3059799368,
            };
            static quant_filter_t point_filter60 = {64, 1, point_weights60, 2.5835359338088892e-05};
            point_filters_b[60] = point_filter60;
            
            static const uint32_t point_weights61[]={2308117275,2572663477,
            };
            static quant_filter_t point_filter61 = {64, 1, point_weights61, 4.664510015572887e-06};
            point_filters_b[61] = point_filter61;
            
            static const uint32_t point_weights62[]={3600345329,1061902098,
            };
            static quant_filter_t point_filter62 = {64, 1, point_weights62, -6.864574243081734e-05};
            point_filters_b[62] = point_filter62;
            
            static const uint32_t point_weights63[]={110097086,4015939814,
            };
            static quant_filter_t point_filter63 = {64, 1, point_weights63, -3.885979822371155e-06};
            point_filters_b[63] = point_filter63;
            
            static const uint32_t point_weights64[]={1719654242,2981771081,
            };
            static quant_filter_t point_filter64 = {64, 1, point_weights64, 3.5113120247842744e-05};
            point_filters_b[64] = point_filter64;
            
            static const uint32_t point_weights65[]={438988546,2242283966,
            };
            static quant_filter_t point_filter65 = {64, 1, point_weights65, -2.2258336684899405e-05};
            point_filters_b[65] = point_filter65;
            
            static const uint32_t point_weights66[]={635365531,4045745394,
            };
            static quant_filter_t point_filter66 = {64, 1, point_weights66, 1.7321163170436193e-08};
            point_filters_b[66] = point_filter66;
            
            static const uint32_t point_weights67[]={4060907705,3938958175,
            };
            static quant_filter_t point_filter67 = {64, 1, point_weights67, 6.0892019973834977e-05};
            point_filters_b[67] = point_filter67;
            
            static const uint32_t point_weights68[]={20817594,1596012343,
            };
            static quant_filter_t point_filter68 = {64, 1, point_weights68, 1.0618985470500775e-05};
            point_filters_b[68] = point_filter68;
            
            static const uint32_t point_weights69[]={605696275,1499055331,
            };
            static quant_filter_t point_filter69 = {64, 1, point_weights69, -1.9366912965779193e-05};
            point_filters_b[69] = point_filter69;
            
            static const uint32_t point_weights70[]={2093279282,3555257872,
            };
            static quant_filter_t point_filter70 = {64, 1, point_weights70, -2.3684504412813112e-05};
            point_filters_b[70] = point_filter70;
            
            static const uint32_t point_weights71[]={631808164,1642142209,
            };
            static quant_filter_t point_filter71 = {64, 1, point_weights71, -2.4103819669107907e-05};
            point_filters_b[71] = point_filter71;
            
            static const uint32_t point_weights72[]={1362780910,3059565106,
            };
            static quant_filter_t point_filter72 = {64, 1, point_weights72, 3.915532033715863e-06};
            point_filters_b[72] = point_filter72;
            
            static const uint32_t point_weights73[]={4100935946,3399906221,
            };
            static quant_filter_t point_filter73 = {64, 1, point_weights73, 7.77511468186276e-06};
            point_filters_b[73] = point_filter73;
            
            static const uint32_t point_weights74[]={2767544191,2222003596,
            };
            static quant_filter_t point_filter74 = {64, 1, point_weights74, 1.4128109796729404e-05};
            point_filters_b[74] = point_filter74;
            
            static const uint32_t point_weights75[]={4201286794,2069886277,
            };
            static quant_filter_t point_filter75 = {64, 1, point_weights75, -3.82256657758262e-05};
            point_filters_b[75] = point_filter75;
            
            static const uint32_t point_weights76[]={1797422526,2440438625,
            };
            static quant_filter_t point_filter76 = {64, 1, point_weights76, 2.136686998710502e-05};
            point_filters_b[76] = point_filter76;
            
            static const uint32_t point_weights77[]={702358804,3896323230,
            };
            static quant_filter_t point_filter77 = {64, 1, point_weights77, 4.700637873611413e-06};
            point_filters_b[77] = point_filter77;
            
            static const uint32_t point_weights78[]={1793143623,732213132,
            };
            static quant_filter_t point_filter78 = {64, 1, point_weights78, -2.220439819211606e-05};
            point_filters_b[78] = point_filter78;
            
            static const uint32_t point_weights79[]={3425984313,2445498165,
            };
            static quant_filter_t point_filter79 = {64, 1, point_weights79, 1.8576129150460474e-05};
            point_filters_b[79] = point_filter79;
            
            static const uint32_t point_weights80[]={2283296538,3164576771,
            };
            static quant_filter_t point_filter80 = {64, 1, point_weights80, -1.880077616078779e-05};
            point_filters_b[80] = point_filter80;
            
            static const uint32_t point_weights81[]={1769129526,894417226,
            };
            static quant_filter_t point_filter81 = {64, 1, point_weights81, 2.6542980776866898e-05};
            point_filters_b[81] = point_filter81;
            
            static const uint32_t point_weights82[]={1368547417,2017870473,
            };
            static quant_filter_t point_filter82 = {64, 1, point_weights82, -3.13655509671662e-05};
            point_filters_b[82] = point_filter82;
            
            static const uint32_t point_weights83[]={3894486626,4071083982,
            };
            static quant_filter_t point_filter83 = {64, 1, point_weights83, -6.22160077909939e-05};
            point_filters_b[83] = point_filter83;
            
            static const uint32_t point_weights84[]={121099118,839602281,
            };
            static quant_filter_t point_filter84 = {64, 1, point_weights84, -5.239272013568552e-06};
            point_filters_b[84] = point_filter84;
            
            static const uint32_t point_weights85[]={463724060,546358545,
            };
            static quant_filter_t point_filter85 = {64, 1, point_weights85, -1.5648171029170044e-05};
            point_filters_b[85] = point_filter85;
            
            static const uint32_t point_weights86[]={854887818,2643332711,
            };
            static quant_filter_t point_filter86 = {64, 1, point_weights86, -1.5026986147859134e-05};
            point_filters_b[86] = point_filter86;
            
            static const uint32_t point_weights87[]={1304278994,4152804804,
            };
            static quant_filter_t point_filter87 = {64, 1, point_weights87, 7.560411177109927e-06};
            point_filters_b[87] = point_filter87;
            
            static const uint32_t point_weights88[]={929368689,2457349542,
            };
            static quant_filter_t point_filter88 = {64, 1, point_weights88, -8.42249755805824e-06};
            point_filters_b[88] = point_filter88;
            
            static const uint32_t point_weights89[]={549139238,653135643,
            };
            static quant_filter_t point_filter89 = {64, 1, point_weights89, -1.7920461687026545e-05};
            point_filters_b[89] = point_filter89;
            
            static const uint32_t point_weights90[]={3723877601,1586021414,
            };
            static quant_filter_t point_filter90 = {64, 1, point_weights90, -1.800507106963778e-06};
            point_filters_b[90] = point_filter90;
            
            static const uint32_t point_weights91[]={3878908094,3519010876,
            };
            static quant_filter_t point_filter91 = {64, 1, point_weights91, -2.29576944548171e-05};
            point_filters_b[91] = point_filter91;
            
            static const uint32_t point_weights92[]={515375701,3359047881,
            };
            static quant_filter_t point_filter92 = {64, 1, point_weights92, 2.4124943593051285e-05};
            point_filters_b[92] = point_filter92;
            
            static const uint32_t point_weights93[]={1780090791,209843871,
            };
            static quant_filter_t point_filter93 = {64, 1, point_weights93, -3.105865107499994e-05};
            point_filters_b[93] = point_filter93;
            
            static const uint32_t point_weights94[]={3396855314,52325692,
            };
            static quant_filter_t point_filter94 = {64, 1, point_weights94, 3.289394271632773e-06};
            point_filters_b[94] = point_filter94;
            
            static const uint32_t point_weights95[]={1021987344,1094617543,
            };
            static quant_filter_t point_filter95 = {64, 1, point_weights95, 1.203549800266046e-05};
            point_filters_b[95] = point_filter95;
            
            quant_separable_conv2d_layer_t layer = {96, depth_filter_b, point_filters_b};
            return layer;
            }
            
batch_normalization_layer_t init_batch_normalization_396_data(void){

    static const fixed inv_gamma_dev[] ={
    5002, 8610, 10418, 9495, 7740, 7521, 7031, 7323, 9256, 9967, 7460, 8327, 8583, 7626, 
    6876, 4417, 10216, 7027, 8407, 3026, 6576, 9206, 8178, 8699, 6325, 7948, 7804, 9387, 
    7515, 7317, 8173, 7315, 8164, 8356, 9494, 7847, 7932, 6480, 6808, 7162, 6742, 8924, 
    2018, 6521, 8720, 8007, 7233, 7984, 7245, 7908, 8637, 3059, 7429, 8895, 7440, 7342, 
    5642, 7746, 9363, 8311, 5518, 7411, 8946, 5878, 8956, 7334, 7609, 9862, 8151, 5787, 
    9125, 7834, 9644, 8099, 5222, 6444, 8585, 8210, 7476, 8151, 8469, 6443, 9026, 6367, 
    3114, 8503, 6071, 8656, 7705, 7472, 10186, 9689, 6888, 6558, 8344, 8874
    };
    static const fixed std_beta[] ={
    -30745, -98086, -65493, -25011, -226466, -147871, -83184, -83670, -274915, -117576, 
    -228363, -156527, 38998, -181037, -46337, -80731, -91563, -213461, -83635, -57767, 
    -37614, -55040, -112917, -60855, -154904, -209533, -76080, -230932, -99868, -207939, 
    -57378, -49781, -306186, -161838, -124250, 31715, -137904, -50461, -104341, -90811, 
    -75948, -140735, -42039, -141768, -163633, -214758, -62962, -105221, -197659, -140959, 
    -243032, 47534, -84989, -220291, -102636, -126112, -66857, -161506, -120504, -57066, 
    26196, -26452, -156838, -222491, -93146, -169071, -106227, -155881, -66624, -76774, 
    -132497, -90306, -106677, -226635, -92274, -203878, -132506, -190887, -13693, -44479, 
    -183117, -150146, -27999, -95875, -14373, -107444, -113715, -153912, -53181, -124044, 
    -301707, -304740, -120743, -7032, -171433, -274132
    };

    static const batch_normalization_layer_t norm = { 96, inv_gamma_dev, std_beta  };
    return norm;
}


            quant_separable_conv2d_layer_t init_larq_quant_separable_conv2_d_wrapper2_data(void){

            
            static const uint32_t depth_weights[]={
    
    
    67108864,
  
    
    
    1887436800,
  
    
    
    2264924160,
  
    
    
    771751936,
  
    
    
    3623878656,
  
    
    
    729808896,
  
    
    
    318767104,
  
    
    
    1543503872,
  
    
    
    3279945728,
  
    
    
    3103784960,
  
    
    
    394264576,
  
    
    
    838860800,
  
    
    
    2113929216,
  
    
    
    1157627904,
  
    
    
    746586112,
  
    
    
    1283457024,
  
    
    
    1803550720,
  
    
    
    4253024256,
  
    
    
    1795162112,
  
    
    
    2357198848,
  
    
    
    1300234240,
  
    
    
    1317011456,
  
    
    
    704643072,
  
    
    
    1040187392,
  
    
    
    2415919104,
  
    
    
    1149239296,
  
    
    
    4102029312,
  
    
    
    805306368,
  
    
    
    4177526784,
  
    
    
    3632267264,
  
    
    
    4034920448,
  
    
    
    2751463424,
  
    
    
    3011510272,
  
    
    
    3732930560,
  
    
    
    1249902592,
  
    
    
    1291845632,
  
    
    
    2776629248,
  
    
    
    100663296,
  
    
    
    1828716544,
  
    
    
    3581935616,
  
    
    
    67108864,
  
    
    
    4160749568,
  
    
    
    41943040,
  
    
    
    3539992576,
  
    
    
    3690987520,
  
    
    
    1358954496,
  
    
    
    2063597568,
  
    
    
    4227858432,
  
    
    
    67108864,
  
    
    
    2902458368,
  
    
    
    1140850688,
  
    
    
    0,
  
    
    
    3472883712,
  
    
    
    855638016,
  
    
    
    1593835520,
  
    
    
    234881024,
  
    
    
    3137339392,
  
    
    
    4085252096,
  
    
    
    360710144,
  
    
    
    218103808,
  
    
    
    142606336,
  
    
    
    1660944384,
  
    
    
    2424307712,
  
    
    
    1249902592,
  
    
    
    3045064704,
  
    
    
    2281701376,
  
    
    
    1711276032,
  
    
    
    2575302656,
  
    
    
    3498049536,
  
    
    
    2181038080,
  
    
    
    3632267264,
  
    
    
    1820327936,
  
    
    
    1023410176,
  
    
    
    3833593856,
  
    
    
    427819008,
  
    
    
    3976200192,
  
    
    
    3422552064,
  
    
    
    2315255808,
  
    
    
    2222981120,
  
    
    
    385875968,
  
    
    
    914358272,
  
    
    
    2583691264,
  
    
    
    2659188736,
  
    
    
    3967811584,
  
    
    
    2189426688,
  
    
    
    3640655872,
  
    
    
    2558525440,
  
    
    
    2709520384,
  
    
    
    3640655872,
  
    
    
    1207959552,
  
    
    
    864026624,
  
    
    
    3112173568,
  
    
    
    1325400064,
  
    
    
    3833593856,
  
    
    
    1887436800,
  
    
    
    1895825408,
  
            };
            static quant_filter_t depth_filter_b = {96, 3, depth_weights, 0};

            static quant_filter_t point_filters_b[512];
            
            static const uint32_t point_weights0[]={2239660030,230888816,3061997685,
            };
            static quant_filter_t point_filter0 = {96, 1, point_weights0, 5.110653091833228e-06};
            point_filters_b[0] = point_filter0;
            
            static const uint32_t point_weights1[]={820612050,2443226538,3829382969,
            };
            static quant_filter_t point_filter1 = {96, 1, point_weights1, 4.409341272548772e-06};
            point_filters_b[1] = point_filter1;
            
            static const uint32_t point_weights2[]={1334086747,3151105265,3606340627,
            };
            static quant_filter_t point_filter2 = {96, 1, point_weights2, -3.2139444101630943e-06};
            point_filters_b[2] = point_filter2;
            
            static const uint32_t point_weights3[]={2099075459,1950053588,1420312833,
            };
            static quant_filter_t point_filter3 = {96, 1, point_weights3, 1.955316292878706e-05};
            point_filters_b[3] = point_filter3;
            
            static const uint32_t point_weights4[]={126964537,187439750,22760093,
            };
            static quant_filter_t point_filter4 = {96, 1, point_weights4, 4.46086460215156e-06};
            point_filters_b[4] = point_filter4;
            
            static const uint32_t point_weights5[]={1078620588,4216559173,714077112,
            };
            static quant_filter_t point_filter5 = {96, 1, point_weights5, -5.993784384372702e-07};
            point_filters_b[5] = point_filter5;
            
            static const uint32_t point_weights6[]={1314988002,3745859105,3044289705,
            };
            static quant_filter_t point_filter6 = {96, 1, point_weights6, 7.019918939477066e-06};
            point_filters_b[6] = point_filter6;
            
            static const uint32_t point_weights7[]={2762213070,2247707820,395565138,
            };
            static quant_filter_t point_filter7 = {96, 1, point_weights7, 1.5459394262506976e-06};
            point_filters_b[7] = point_filter7;
            
            static const uint32_t point_weights8[]={3907375208,4021966049,3269648600,
            };
            static quant_filter_t point_filter8 = {96, 1, point_weights8, 1.914301265060203e-06};
            point_filters_b[8] = point_filter8;
            
            static const uint32_t point_weights9[]={2727853003,2346898051,3544910388,
            };
            static quant_filter_t point_filter9 = {96, 1, point_weights9, -5.67709776078118e-06};
            point_filters_b[9] = point_filter9;
            
            static const uint32_t point_weights10[]={1921766531,248958499,1379838011,
            };
            static quant_filter_t point_filter10 = {96, 1, point_weights10, 3.6652743347076466e-06};
            point_filters_b[10] = point_filter10;
            
            static const uint32_t point_weights11[]={3681902671,3655842189,1838988673,
            };
            static quant_filter_t point_filter11 = {96, 1, point_weights11, 2.3675938791711815e-06};
            point_filters_b[11] = point_filter11;
            
            static const uint32_t point_weights12[]={322055978,2514113792,3396935892,
            };
            static quant_filter_t point_filter12 = {96, 1, point_weights12, -4.751902906718897e-06};
            point_filters_b[12] = point_filter12;
            
            static const uint32_t point_weights13[]={1935385897,3363477397,3220613892,
            };
            static quant_filter_t point_filter13 = {96, 1, point_weights13, 2.4504247448930983e-06};
            point_filters_b[13] = point_filter13;
            
            static const uint32_t point_weights14[]={622133987,4187930184,1774750970,
            };
            static quant_filter_t point_filter14 = {96, 1, point_weights14, -4.334719051257707e-06};
            point_filters_b[14] = point_filter14;
            
            static const uint32_t point_weights15[]={2976752364,3665495014,1746677913,
            };
            static quant_filter_t point_filter15 = {96, 1, point_weights15, 7.317293693631655e-06};
            point_filters_b[15] = point_filter15;
            
            static const uint32_t point_weights16[]={46024391,210230248,2980195103,
            };
            static quant_filter_t point_filter16 = {96, 1, point_weights16, 3.191475798303145e-06};
            point_filters_b[16] = point_filter16;
            
            static const uint32_t point_weights17[]={3285721141,919662890,4243974436,
            };
            static quant_filter_t point_filter17 = {96, 1, point_weights17, 5.01396561958245e-06};
            point_filters_b[17] = point_filter17;
            
            static const uint32_t point_weights18[]={1039355302,4224302258,2597550013,
            };
            static quant_filter_t point_filter18 = {96, 1, point_weights18, -1.0834217391675338e-05};
            point_filters_b[18] = point_filter18;
            
            static const uint32_t point_weights19[]={2633874159,318064978,4038254976,
            };
            static quant_filter_t point_filter19 = {96, 1, point_weights19, 6.124447736510774e-06};
            point_filters_b[19] = point_filter19;
            
            static const uint32_t point_weights20[]={3605998047,3452997743,2528122455,
            };
            static quant_filter_t point_filter20 = {96, 1, point_weights20, -5.087554654892301e-06};
            point_filters_b[20] = point_filter20;
            
            static const uint32_t point_weights21[]={3113919024,1483057935,4034382258,
            };
            static quant_filter_t point_filter21 = {96, 1, point_weights21, 8.434782472477309e-08};
            point_filters_b[21] = point_filter21;
            
            static const uint32_t point_weights22[]={144578891,515455605,2465974757,
            };
            static quant_filter_t point_filter22 = {96, 1, point_weights22, -1.3699846022063866e-06};
            point_filters_b[22] = point_filter22;
            
            static const uint32_t point_weights23[]={2420776784,1973142344,2706572307,
            };
            static quant_filter_t point_filter23 = {96, 1, point_weights23, -5.883688913854712e-07};
            point_filters_b[23] = point_filter23;
            
            static const uint32_t point_weights24[]={3489644551,3681474509,827439221,
            };
            static quant_filter_t point_filter24 = {96, 1, point_weights24, 1.3357156376514467e-06};
            point_filters_b[24] = point_filter24;
            
            static const uint32_t point_weights25[]={88787089,4015606097,2537905578,
            };
            static quant_filter_t point_filter25 = {96, 1, point_weights25, -3.896643193002092e-06};
            point_filters_b[25] = point_filter25;
            
            static const uint32_t point_weights26[]={155938477,464390308,3626675884,
            };
            static quant_filter_t point_filter26 = {96, 1, point_weights26, 1.698428036434052e-06};
            point_filters_b[26] = point_filter26;
            
            static const uint32_t point_weights27[]={3106453647,3167404214,1735900176,
            };
            static quant_filter_t point_filter27 = {96, 1, point_weights27, -3.1764491268404527e-06};
            point_filters_b[27] = point_filter27;
            
            static const uint32_t point_weights28[]={4098846976,109243913,1978456788,
            };
            static quant_filter_t point_filter28 = {96, 1, point_weights28, 8.408435860474128e-06};
            point_filters_b[28] = point_filter28;
            
            static const uint32_t point_weights29[]={1415327578,3630279096,2490038422,
            };
            static quant_filter_t point_filter29 = {96, 1, point_weights29, -2.8385834411892574e-06};
            point_filters_b[29] = point_filter29;
            
            static const uint32_t point_weights30[]={1419514356,1354201356,169011794,
            };
            static quant_filter_t point_filter30 = {96, 1, point_weights30, -2.8593924525921466e-06};
            point_filters_b[30] = point_filter30;
            
            static const uint32_t point_weights31[]={2815064104,2178085795,3510634361,
            };
            static quant_filter_t point_filter31 = {96, 1, point_weights31, -4.716842340712901e-06};
            point_filters_b[31] = point_filter31;
            
            static const uint32_t point_weights32[]={3545160859,4115236065,2557467188,
            };
            static quant_filter_t point_filter32 = {96, 1, point_weights32, 4.0418049138679635e-06};
            point_filters_b[32] = point_filter32;
            
            static const uint32_t point_weights33[]={2575129359,2936176689,3164112074,
            };
            static quant_filter_t point_filter33 = {96, 1, point_weights33, 1.8158211787522305e-06};
            point_filters_b[33] = point_filter33;
            
            static const uint32_t point_weights34[]={1272233028,2043302756,3707815042,
            };
            static quant_filter_t point_filter34 = {96, 1, point_weights34, -3.7888510178163415e-06};
            point_filters_b[34] = point_filter34;
            
            static const uint32_t point_weights35[]={3287791865,923199336,1782411546,
            };
            static quant_filter_t point_filter35 = {96, 1, point_weights35, -1.7981432165470324e-06};
            point_filters_b[35] = point_filter35;
            
            static const uint32_t point_weights36[]={3471865365,3317047940,103421111,
            };
            static quant_filter_t point_filter36 = {96, 1, point_weights36, 6.13376141700428e-06};
            point_filters_b[36] = point_filter36;
            
            static const uint32_t point_weights37[]={4241841016,1369277489,277738789,
            };
            static quant_filter_t point_filter37 = {96, 1, point_weights37, 1.6889690357402287e-07};
            point_filters_b[37] = point_filter37;
            
            static const uint32_t point_weights38[]={1627744964,4057874072,1739637651,
            };
            static quant_filter_t point_filter38 = {96, 1, point_weights38, 8.944472824623517e-07};
            point_filters_b[38] = point_filter38;
            
            static const uint32_t point_weights39[]={4223965153,2366855425,3966885821,
            };
            static quant_filter_t point_filter39 = {96, 1, point_weights39, 1.284591121475387e-06};
            point_filters_b[39] = point_filter39;
            
            static const uint32_t point_weights40[]={2537205567,3429498398,4218946406,
            };
            static quant_filter_t point_filter40 = {96, 1, point_weights40, 5.916866939514875e-06};
            point_filters_b[40] = point_filter40;
            
            static const uint32_t point_weights41[]={912165951,3596325756,2824610089,
            };
            static quant_filter_t point_filter41 = {96, 1, point_weights41, -4.570780674839625e-06};
            point_filters_b[41] = point_filter41;
            
            static const uint32_t point_weights42[]={528868322,253694733,473310658,
            };
            static quant_filter_t point_filter42 = {96, 1, point_weights42, 3.2857235510164173e-07};
            point_filters_b[42] = point_filter42;
            
            static const uint32_t point_weights43[]={1104272280,4093893887,1008560276,
            };
            static quant_filter_t point_filter43 = {96, 1, point_weights43, 7.2378867344014e-07};
            point_filters_b[43] = point_filter43;
            
            static const uint32_t point_weights44[]={1727985452,1220281328,3347329699,
            };
            static quant_filter_t point_filter44 = {96, 1, point_weights44, -8.775239621172659e-06};
            point_filters_b[44] = point_filter44;
            
            static const uint32_t point_weights45[]={1890221643,1763861415,1630140561,
            };
            static quant_filter_t point_filter45 = {96, 1, point_weights45, -1.2666391739912797e-06};
            point_filters_b[45] = point_filter45;
            
            static const uint32_t point_weights46[]={3938585138,146168132,3944638440,
            };
            static quant_filter_t point_filter46 = {96, 1, point_weights46, -1.7721622498356737e-05};
            point_filters_b[46] = point_filter46;
            
            static const uint32_t point_weights47[]={2801955853,174172455,2081676765,
            };
            static quant_filter_t point_filter47 = {96, 1, point_weights47, -9.853767551248893e-06};
            point_filters_b[47] = point_filter47;
            
            static const uint32_t point_weights48[]={331924450,226699957,3442972944,
            };
            static quant_filter_t point_filter48 = {96, 1, point_weights48, 2.159222731279442e-06};
            point_filters_b[48] = point_filter48;
            
            static const uint32_t point_weights49[]={1608135803,4001781749,1845332807,
            };
            static quant_filter_t point_filter49 = {96, 1, point_weights49, -1.8281019720234326e-07};
            point_filters_b[49] = point_filter49;
            
            static const uint32_t point_weights50[]={497790914,3202797681,3839943502,
            };
            static quant_filter_t point_filter50 = {96, 1, point_weights50, 4.467666713026119e-06};
            point_filters_b[50] = point_filter50;
            
            static const uint32_t point_weights51[]={2785182066,1286522667,122114991,
            };
            static quant_filter_t point_filter51 = {96, 1, point_weights51, -3.3484911909908988e-06};
            point_filters_b[51] = point_filter51;
            
            static const uint32_t point_weights52[]={3175987699,1005614997,283010849,
            };
            static quant_filter_t point_filter52 = {96, 1, point_weights52, 3.590991582314018e-06};
            point_filters_b[52] = point_filter52;
            
            static const uint32_t point_weights53[]={2565036930,4211812366,3496384183,
            };
            static quant_filter_t point_filter53 = {96, 1, point_weights53, -1.4181306369209778e-06};
            point_filters_b[53] = point_filter53;
            
            static const uint32_t point_weights54[]={1395010774,1116631698,1897283453,
            };
            static quant_filter_t point_filter54 = {96, 1, point_weights54, 2.239807827209006e-06};
            point_filters_b[54] = point_filter54;
            
            static const uint32_t point_weights55[]={4059157311,554769167,406591908,
            };
            static quant_filter_t point_filter55 = {96, 1, point_weights55, -7.46907471693703e-06};
            point_filters_b[55] = point_filter55;
            
            static const uint32_t point_weights56[]={3840276916,2160325395,1877660135,
            };
            static quant_filter_t point_filter56 = {96, 1, point_weights56, 1.1061997611250263e-05};
            point_filters_b[56] = point_filter56;
            
            static const uint32_t point_weights57[]={1312413403,1611868142,422020015,
            };
            static quant_filter_t point_filter57 = {96, 1, point_weights57, -5.075646640761988e-06};
            point_filters_b[57] = point_filter57;
            
            static const uint32_t point_weights58[]={1045773161,3566619428,407804913,
            };
            static quant_filter_t point_filter58 = {96, 1, point_weights58, 6.834593932580901e-06};
            point_filters_b[58] = point_filter58;
            
            static const uint32_t point_weights59[]={377059402,666384002,1632714522,
            };
            static quant_filter_t point_filter59 = {96, 1, point_weights59, 3.887089405907318e-06};
            point_filters_b[59] = point_filter59;
            
            static const uint32_t point_weights60[]={840400126,2030030228,120181132,
            };
            static quant_filter_t point_filter60 = {96, 1, point_weights60, 1.6858322169355233e-06};
            point_filters_b[60] = point_filter60;
            
            static const uint32_t point_weights61[]={140037053,1148271401,2248827937,
            };
            static quant_filter_t point_filter61 = {96, 1, point_weights61, 5.171449174667941e-06};
            point_filters_b[61] = point_filter61;
            
            static const uint32_t point_weights62[]={3795576385,4239672410,904212940,
            };
            static quant_filter_t point_filter62 = {96, 1, point_weights62, 1.0501561291675898e-06};
            point_filters_b[62] = point_filter62;
            
            static const uint32_t point_weights63[]={2808418500,2863052234,491071497,
            };
            static quant_filter_t point_filter63 = {96, 1, point_weights63, -3.922683845303254e-06};
            point_filters_b[63] = point_filter63;
            
            static const uint32_t point_weights64[]={3594379684,808599341,3378195949,
            };
            static quant_filter_t point_filter64 = {96, 1, point_weights64, 2.3603688532602973e-06};
            point_filters_b[64] = point_filter64;
            
            static const uint32_t point_weights65[]={2219159360,4044772835,463984918,
            };
            static quant_filter_t point_filter65 = {96, 1, point_weights65, 8.490640084346524e-07};
            point_filters_b[65] = point_filter65;
            
            static const uint32_t point_weights66[]={2402828877,861634708,3280140861,
            };
            static quant_filter_t point_filter66 = {96, 1, point_weights66, -3.6852904941042652e-06};
            point_filters_b[66] = point_filter66;
            
            static const uint32_t point_weights67[]={3411134484,2620830954,3371209505,
            };
            static quant_filter_t point_filter67 = {96, 1, point_weights67, 2.3168732354861277e-07};
            point_filters_b[67] = point_filter67;
            
            static const uint32_t point_weights68[]={3136045348,3182371305,3971047009,
            };
            static quant_filter_t point_filter68 = {96, 1, point_weights68, 3.357488594701863e-06};
            point_filters_b[68] = point_filter68;
            
            static const uint32_t point_weights69[]={4281070344,2905035577,1406039959,
            };
            static quant_filter_t point_filter69 = {96, 1, point_weights69, -3.7796530705236364e-06};
            point_filters_b[69] = point_filter69;
            
            static const uint32_t point_weights70[]={3617056298,2599066980,2371279739,
            };
            static quant_filter_t point_filter70 = {96, 1, point_weights70, -1.3257558748591691e-06};
            point_filters_b[70] = point_filter70;
            
            static const uint32_t point_weights71[]={610675572,1217767941,2102336015,
            };
            static quant_filter_t point_filter71 = {96, 1, point_weights71, 3.332370170028298e-06};
            point_filters_b[71] = point_filter71;
            
            static const uint32_t point_weights72[]={7126805,1472905320,997391911,
            };
            static quant_filter_t point_filter72 = {96, 1, point_weights72, -6.0378552007023245e-06};
            point_filters_b[72] = point_filter72;
            
            static const uint32_t point_weights73[]={1543700464,3113547470,2397675162,
            };
            static quant_filter_t point_filter73 = {96, 1, point_weights73, -2.3356885776593117e-06};
            point_filters_b[73] = point_filter73;
            
            static const uint32_t point_weights74[]={2522514238,4161784041,2454326911,
            };
            static quant_filter_t point_filter74 = {96, 1, point_weights74, -2.8963970635231817e-06};
            point_filters_b[74] = point_filter74;
            
            static const uint32_t point_weights75[]={379869172,2570514842,2414220615,
            };
            static quant_filter_t point_filter75 = {96, 1, point_weights75, 3.982076577813132e-06};
            point_filters_b[75] = point_filter75;
            
            static const uint32_t point_weights76[]={1237844425,1935138513,3468680760,
            };
            static quant_filter_t point_filter76 = {96, 1, point_weights76, -2.151108674297575e-06};
            point_filters_b[76] = point_filter76;
            
            static const uint32_t point_weights77[]={151175660,1453968717,1733682069,
            };
            static quant_filter_t point_filter77 = {96, 1, point_weights77, 3.480284021861735e-06};
            point_filters_b[77] = point_filter77;
            
            static const uint32_t point_weights78[]={2784311766,2852604812,249045113,
            };
            static quant_filter_t point_filter78 = {96, 1, point_weights78, 1.5155354731177795e-06};
            point_filters_b[78] = point_filter78;
            
            static const uint32_t point_weights79[]={2872487347,3069278663,853782168,
            };
            static quant_filter_t point_filter79 = {96, 1, point_weights79, -5.555553798330948e-06};
            point_filters_b[79] = point_filter79;
            
            static const uint32_t point_weights80[]={2564294200,3633858678,2655634262,
            };
            static quant_filter_t point_filter80 = {96, 1, point_weights80, -3.866882252623327e-06};
            point_filters_b[80] = point_filter80;
            
            static const uint32_t point_weights81[]={2671476949,1963605702,1178323108,
            };
            static quant_filter_t point_filter81 = {96, 1, point_weights81, 4.684719954184402e-07};
            point_filters_b[81] = point_filter81;
            
            static const uint32_t point_weights82[]={1168089710,2744762210,2470881147,
            };
            static quant_filter_t point_filter82 = {96, 1, point_weights82, -2.142819766959292e-06};
            point_filters_b[82] = point_filter82;
            
            static const uint32_t point_weights83[]={897662756,4250660018,3461126460,
            };
            static quant_filter_t point_filter83 = {96, 1, point_weights83, 2.2966466985963052e-06};
            point_filters_b[83] = point_filter83;
            
            static const uint32_t point_weights84[]={2398980259,875678249,2649228810,
            };
            static quant_filter_t point_filter84 = {96, 1, point_weights84, -1.1387959375497303e-06};
            point_filters_b[84] = point_filter84;
            
            static const uint32_t point_weights85[]={2617330096,2845397210,2759066908,
            };
            static quant_filter_t point_filter85 = {96, 1, point_weights85, 3.4361996767984238e-06};
            point_filters_b[85] = point_filter85;
            
            static const uint32_t point_weights86[]={3313392433,72185957,2467542732,
            };
            static quant_filter_t point_filter86 = {96, 1, point_weights86, -3.286447736172704e-06};
            point_filters_b[86] = point_filter86;
            
            static const uint32_t point_weights87[]={4220243264,2713415436,2095810206,
            };
            static quant_filter_t point_filter87 = {96, 1, point_weights87, -8.402981620747596e-06};
            point_filters_b[87] = point_filter87;
            
            static const uint32_t point_weights88[]={1856560085,1837452868,2720313302,
            };
            static quant_filter_t point_filter88 = {96, 1, point_weights88, -4.321388757944078e-07};
            point_filters_b[88] = point_filter88;
            
            static const uint32_t point_weights89[]={2179303185,1624821256,1257571996,
            };
            static quant_filter_t point_filter89 = {96, 1, point_weights89, -1.9313988559588324e-06};
            point_filters_b[89] = point_filter89;
            
            static const uint32_t point_weights90[]={2209034014,4121987243,332827659,
            };
            static quant_filter_t point_filter90 = {96, 1, point_weights90, 7.011850811977638e-06};
            point_filters_b[90] = point_filter90;
            
            static const uint32_t point_weights91[]={2283214724,2079587808,385473335,
            };
            static quant_filter_t point_filter91 = {96, 1, point_weights91, -2.242517894046614e-06};
            point_filters_b[91] = point_filter91;
            
            static const uint32_t point_weights92[]={4288426131,1082442753,3560481729,
            };
            static quant_filter_t point_filter92 = {96, 1, point_weights92, 4.111283544716571e-07};
            point_filters_b[92] = point_filter92;
            
            static const uint32_t point_weights93[]={4241942813,3649620992,151911216,
            };
            static quant_filter_t point_filter93 = {96, 1, point_weights93, 6.4703813222877216e-06};
            point_filters_b[93] = point_filter93;
            
            static const uint32_t point_weights94[]={3159136055,712087532,3848167364,
            };
            static quant_filter_t point_filter94 = {96, 1, point_weights94, 9.540191285850597e-07};
            point_filters_b[94] = point_filter94;
            
            static const uint32_t point_weights95[]={281024916,1113607833,4034952294,
            };
            static quant_filter_t point_filter95 = {96, 1, point_weights95, 5.2904147196386475e-06};
            point_filters_b[95] = point_filter95;
            
            static const uint32_t point_weights96[]={3716684795,2773032810,2019652982,
            };
            static quant_filter_t point_filter96 = {96, 1, point_weights96, -3.6513918644232035e-07};
            point_filters_b[96] = point_filter96;
            
            static const uint32_t point_weights97[]={21240016,1145195542,2833396768,
            };
            static quant_filter_t point_filter97 = {96, 1, point_weights97, 6.211749337126093e-07};
            point_filters_b[97] = point_filter97;
            
            static const uint32_t point_weights98[]={546977035,3874846820,4237596640,
            };
            static quant_filter_t point_filter98 = {96, 1, point_weights98, 1.3610729183710646e-06};
            point_filters_b[98] = point_filter98;
            
            static const uint32_t point_weights99[]={2572117568,2494116653,2991970608,
            };
            static quant_filter_t point_filter99 = {96, 1, point_weights99, 9.677426533016842e-06};
            point_filters_b[99] = point_filter99;
            
            static const uint32_t point_weights100[]={1286652526,1048757075,261617270,
            };
            static quant_filter_t point_filter100 = {96, 1, point_weights100, -2.3439035885530757e-06};
            point_filters_b[100] = point_filter100;
            
            static const uint32_t point_weights101[]={163690706,4192500102,2235709608,
            };
            static quant_filter_t point_filter101 = {96, 1, point_weights101, 4.478863502299646e-07};
            point_filters_b[101] = point_filter101;
            
            static const uint32_t point_weights102[]={3074300334,4082758486,1147300585,
            };
            static quant_filter_t point_filter102 = {96, 1, point_weights102, 5.790381578663073e-07};
            point_filters_b[102] = point_filter102;
            
            static const uint32_t point_weights103[]={2786904643,1452479336,74362983,
            };
            static quant_filter_t point_filter103 = {96, 1, point_weights103, 3.2623631796013797e-06};
            point_filters_b[103] = point_filter103;
            
            static const uint32_t point_weights104[]={502109392,3421049881,1553968147,
            };
            static quant_filter_t point_filter104 = {96, 1, point_weights104, -6.226438017620239e-06};
            point_filters_b[104] = point_filter104;
            
            static const uint32_t point_weights105[]={1675632749,4164284125,2653847539,
            };
            static quant_filter_t point_filter105 = {96, 1, point_weights105, -6.617810868192464e-06};
            point_filters_b[105] = point_filter105;
            
            static const uint32_t point_weights106[]={3019162681,2184967277,3866570096,
            };
            static quant_filter_t point_filter106 = {96, 1, point_weights106, 1.0257372196065262e-06};
            point_filters_b[106] = point_filter106;
            
            static const uint32_t point_weights107[]={1535870428,3460649804,3456072931,
            };
            static quant_filter_t point_filter107 = {96, 1, point_weights107, -1.2302101822569966e-05};
            point_filters_b[107] = point_filter107;
            
            static const uint32_t point_weights108[]={1214700585,1327508414,3006608141,
            };
            static quant_filter_t point_filter108 = {96, 1, point_weights108, -8.74184081567364e-07};
            point_filters_b[108] = point_filter108;
            
            static const uint32_t point_weights109[]={218027638,1730090736,3146542588,
            };
            static quant_filter_t point_filter109 = {96, 1, point_weights109, -8.7556645667064e-06};
            point_filters_b[109] = point_filter109;
            
            static const uint32_t point_weights110[]={2631476733,31279628,1798217264,
            };
            static quant_filter_t point_filter110 = {96, 1, point_weights110, -7.358355560427299e-06};
            point_filters_b[110] = point_filter110;
            
            static const uint32_t point_weights111[]={1791590725,2537041297,3173511128,
            };
            static quant_filter_t point_filter111 = {96, 1, point_weights111, 1.450111312806257e-06};
            point_filters_b[111] = point_filter111;
            
            static const uint32_t point_weights112[]={1737552635,3507613298,160323644,
            };
            static quant_filter_t point_filter112 = {96, 1, point_weights112, 3.7618663100147387e-06};
            point_filters_b[112] = point_filter112;
            
            static const uint32_t point_weights113[]={3107779142,1954737147,4090963495,
            };
            static quant_filter_t point_filter113 = {96, 1, point_weights113, 7.835004907974508e-06};
            point_filters_b[113] = point_filter113;
            
            static const uint32_t point_weights114[]={1478621602,3857093614,643218654,
            };
            static quant_filter_t point_filter114 = {96, 1, point_weights114, 4.1244302337872796e-06};
            point_filters_b[114] = point_filter114;
            
            static const uint32_t point_weights115[]={1064177684,1929463081,3891339312,
            };
            static quant_filter_t point_filter115 = {96, 1, point_weights115, -1.4397153336176416e-06};
            point_filters_b[115] = point_filter115;
            
            static const uint32_t point_weights116[]={1537679652,1328069016,2244095059,
            };
            static quant_filter_t point_filter116 = {96, 1, point_weights116, -3.612285809140303e-06};
            point_filters_b[116] = point_filter116;
            
            static const uint32_t point_weights117[]={2369825070,3436238640,2045220918,
            };
            static quant_filter_t point_filter117 = {96, 1, point_weights117, -6.2223239183367696e-06};
            point_filters_b[117] = point_filter117;
            
            static const uint32_t point_weights118[]={2129997967,3634134725,1209925094,
            };
            static quant_filter_t point_filter118 = {96, 1, point_weights118, 2.4189168925659033e-06};
            point_filters_b[118] = point_filter118;
            
            static const uint32_t point_weights119[]={3405846542,1767032920,2332315736,
            };
            static quant_filter_t point_filter119 = {96, 1, point_weights119, 1.0885991059694788e-06};
            point_filters_b[119] = point_filter119;
            
            static const uint32_t point_weights120[]={3102968508,3465888656,968467692,
            };
            static quant_filter_t point_filter120 = {96, 1, point_weights120, 6.273795406741556e-06};
            point_filters_b[120] = point_filter120;
            
            static const uint32_t point_weights121[]={1592088483,1891567864,1895026853,
            };
            static quant_filter_t point_filter121 = {96, 1, point_weights121, -1.7083591501432238e-06};
            point_filters_b[121] = point_filter121;
            
            static const uint32_t point_weights122[]={532644537,1206428999,2052831825,
            };
            static quant_filter_t point_filter122 = {96, 1, point_weights122, 2.3747286377329146e-06};
            point_filters_b[122] = point_filter122;
            
            static const uint32_t point_weights123[]={3488624708,2898433448,2046918933,
            };
            static quant_filter_t point_filter123 = {96, 1, point_weights123, 1.1482416084618308e-05};
            point_filters_b[123] = point_filter123;
            
            static const uint32_t point_weights124[]={1947067834,3440068803,2507173718,
            };
            static quant_filter_t point_filter124 = {96, 1, point_weights124, 6.123541425040457e-06};
            point_filters_b[124] = point_filter124;
            
            static const uint32_t point_weights125[]={1394189561,456266496,23814940,
            };
            static quant_filter_t point_filter125 = {96, 1, point_weights125, 2.907188900280744e-06};
            point_filters_b[125] = point_filter125;
            
            static const uint32_t point_weights126[]={2452155133,1724150156,3431017277,
            };
            static quant_filter_t point_filter126 = {96, 1, point_weights126, 5.563102831729339e-07};
            point_filters_b[126] = point_filter126;
            
            static const uint32_t point_weights127[]={3619658473,3448692435,40182827,
            };
            static quant_filter_t point_filter127 = {96, 1, point_weights127, 1.5175381804510835e-06};
            point_filters_b[127] = point_filter127;
            
            static const uint32_t point_weights128[]={1388522,4256964011,4073084701,
            };
            static quant_filter_t point_filter128 = {96, 1, point_weights128, -2.806824568324373e-06};
            point_filters_b[128] = point_filter128;
            
            static const uint32_t point_weights129[]={470834459,1769617032,3173278931,
            };
            static quant_filter_t point_filter129 = {96, 1, point_weights129, 2.3873451482359087e-06};
            point_filters_b[129] = point_filter129;
            
            static const uint32_t point_weights130[]={1799365604,2060780165,3398057285,
            };
            static quant_filter_t point_filter130 = {96, 1, point_weights130, -5.106891876494046e-06};
            point_filters_b[130] = point_filter130;
            
            static const uint32_t point_weights131[]={1918602515,3742777957,4173570796,
            };
            static quant_filter_t point_filter131 = {96, 1, point_weights131, 2.1795347038278123e-06};
            point_filters_b[131] = point_filter131;
            
            static const uint32_t point_weights132[]={703074784,69686505,4055944668,
            };
            static quant_filter_t point_filter132 = {96, 1, point_weights132, -2.9733030260103988e-06};
            point_filters_b[132] = point_filter132;
            
            static const uint32_t point_weights133[]={2056326826,3119360023,4087988244,
            };
            static quant_filter_t point_filter133 = {96, 1, point_weights133, 2.6623561097949278e-06};
            point_filters_b[133] = point_filter133;
            
            static const uint32_t point_weights134[]={2623557857,622876369,3832234148,
            };
            static quant_filter_t point_filter134 = {96, 1, point_weights134, 3.4547263112472137e-06};
            point_filters_b[134] = point_filter134;
            
            static const uint32_t point_weights135[]={1186768046,2968792130,788029225,
            };
            static quant_filter_t point_filter135 = {96, 1, point_weights135, 3.3005808290909044e-06};
            point_filters_b[135] = point_filter135;
            
            static const uint32_t point_weights136[]={2499557945,715528187,3923639819,
            };
            static quant_filter_t point_filter136 = {96, 1, point_weights136, -6.760688847862184e-06};
            point_filters_b[136] = point_filter136;
            
            static const uint32_t point_weights137[]={3709475899,2489060531,2728960546,
            };
            static quant_filter_t point_filter137 = {96, 1, point_weights137, 7.770346201141365e-06};
            point_filters_b[137] = point_filter137;
            
            static const uint32_t point_weights138[]={2208877457,2127764539,2855212392,
            };
            static quant_filter_t point_filter138 = {96, 1, point_weights138, 5.976398824714124e-07};
            point_filters_b[138] = point_filter138;
            
            static const uint32_t point_weights139[]={2596574243,3979027699,2697351015,
            };
            static quant_filter_t point_filter139 = {96, 1, point_weights139, 1.1572896255529486e-05};
            point_filters_b[139] = point_filter139;
            
            static const uint32_t point_weights140[]={1251742703,3131955995,692306172,
            };
            static quant_filter_t point_filter140 = {96, 1, point_weights140, 3.6945173178537516e-06};
            point_filters_b[140] = point_filter140;
            
            static const uint32_t point_weights141[]={2874580374,2231927041,1336675512,
            };
            static quant_filter_t point_filter141 = {96, 1, point_weights141, -1.6463171732539195e-06};
            point_filters_b[141] = point_filter141;
            
            static const uint32_t point_weights142[]={279422838,1867151276,1856540259,
            };
            static quant_filter_t point_filter142 = {96, 1, point_weights142, -8.30853150546318e-06};
            point_filters_b[142] = point_filter142;
            
            static const uint32_t point_weights143[]={2296764155,3862423038,1001991706,
            };
            static quant_filter_t point_filter143 = {96, 1, point_weights143, 1.0874628486590154e-07};
            point_filters_b[143] = point_filter143;
            
            static const uint32_t point_weights144[]={200403143,722262680,1220691483,
            };
            static quant_filter_t point_filter144 = {96, 1, point_weights144, -4.250181063980563e-06};
            point_filters_b[144] = point_filter144;
            
            static const uint32_t point_weights145[]={2305571620,504481723,86603163,
            };
            static quant_filter_t point_filter145 = {96, 1, point_weights145, 3.175117342379963e-07};
            point_filters_b[145] = point_filter145;
            
            static const uint32_t point_weights146[]={2493729361,283570779,128580903,
            };
            static quant_filter_t point_filter146 = {96, 1, point_weights146, -6.767302807020315e-08};
            point_filters_b[146] = point_filter146;
            
            static const uint32_t point_weights147[]={1674716329,1469941630,2170782350,
            };
            static quant_filter_t point_filter147 = {96, 1, point_weights147, 1.940496986208018e-06};
            point_filters_b[147] = point_filter147;
            
            static const uint32_t point_weights148[]={246883433,618760065,3661157380,
            };
            static quant_filter_t point_filter148 = {96, 1, point_weights148, 3.042681555598392e-06};
            point_filters_b[148] = point_filter148;
            
            static const uint32_t point_weights149[]={1573911334,2440599098,2828925746,
            };
            static quant_filter_t point_filter149 = {96, 1, point_weights149, 3.559607080205751e-07};
            point_filters_b[149] = point_filter149;
            
            static const uint32_t point_weights150[]={3794401581,3444798840,2519724505,
            };
            static quant_filter_t point_filter150 = {96, 1, point_weights150, -3.107382326561492e-06};
            point_filters_b[150] = point_filter150;
            
            static const uint32_t point_weights151[]={112383880,1833374522,835393509,
            };
            static quant_filter_t point_filter151 = {96, 1, point_weights151, 2.7724843221221818e-06};
            point_filters_b[151] = point_filter151;
            
            static const uint32_t point_weights152[]={204815078,2901127580,1457205298,
            };
            static quant_filter_t point_filter152 = {96, 1, point_weights152, -6.871817731735064e-06};
            point_filters_b[152] = point_filter152;
            
            static const uint32_t point_weights153[]={710247676,840851929,3009346934,
            };
            static quant_filter_t point_filter153 = {96, 1, point_weights153, 5.459700332721695e-06};
            point_filters_b[153] = point_filter153;
            
            static const uint32_t point_weights154[]={734957511,1012326182,3648411475,
            };
            static quant_filter_t point_filter154 = {96, 1, point_weights154, -7.2505536081735045e-06};
            point_filters_b[154] = point_filter154;
            
            static const uint32_t point_weights155[]={1426683319,151353,434884694,
            };
            static quant_filter_t point_filter155 = {96, 1, point_weights155, -7.532963081757771e-06};
            point_filters_b[155] = point_filter155;
            
            static const uint32_t point_weights156[]={4015112851,172755419,666262022,
            };
            static quant_filter_t point_filter156 = {96, 1, point_weights156, 1.3423978089122102e-06};
            point_filters_b[156] = point_filter156;
            
            static const uint32_t point_weights157[]={2659946649,1445208964,2541029353,
            };
            static quant_filter_t point_filter157 = {96, 1, point_weights157, -3.2024397569330176e-06};
            point_filters_b[157] = point_filter157;
            
            static const uint32_t point_weights158[]={912201221,3824874453,616113584,
            };
            static quant_filter_t point_filter158 = {96, 1, point_weights158, -1.7145904394055833e-06};
            point_filters_b[158] = point_filter158;
            
            static const uint32_t point_weights159[]={694651056,3708416216,3506855525,
            };
            static quant_filter_t point_filter159 = {96, 1, point_weights159, 7.023108992143534e-06};
            point_filters_b[159] = point_filter159;
            
            static const uint32_t point_weights160[]={3174553708,1847647651,2365225931,
            };
            static quant_filter_t point_filter160 = {96, 1, point_weights160, 2.3553143364551943e-06};
            point_filters_b[160] = point_filter160;
            
            static const uint32_t point_weights161[]={3249777413,2443540294,3872307992,
            };
            static quant_filter_t point_filter161 = {96, 1, point_weights161, 2.938994384749094e-06};
            point_filters_b[161] = point_filter161;
            
            static const uint32_t point_weights162[]={3905861871,3463668042,755367107,
            };
            static quant_filter_t point_filter162 = {96, 1, point_weights162, 5.208517450228101e-06};
            point_filters_b[162] = point_filter162;
            
            static const uint32_t point_weights163[]={2732150510,2381995482,4168172114,
            };
            static quant_filter_t point_filter163 = {96, 1, point_weights163, -5.310137566993944e-06};
            point_filters_b[163] = point_filter163;
            
            static const uint32_t point_weights164[]={2942305981,3016949893,4068095662,
            };
            static quant_filter_t point_filter164 = {96, 1, point_weights164, 8.894409120330238e-07};
            point_filters_b[164] = point_filter164;
            
            static const uint32_t point_weights165[]={441093424,3652009602,724648909,
            };
            static quant_filter_t point_filter165 = {96, 1, point_weights165, 2.2858448289753142e-07};
            point_filters_b[165] = point_filter165;
            
            static const uint32_t point_weights166[]={3268519639,2076167093,4178878085,
            };
            static quant_filter_t point_filter166 = {96, 1, point_weights166, 1.1301869562885258e-05};
            point_filters_b[166] = point_filter166;
            
            static const uint32_t point_weights167[]={4190339074,785469008,754397891,
            };
            static quant_filter_t point_filter167 = {96, 1, point_weights167, 2.169628260162426e-06};
            point_filters_b[167] = point_filter167;
            
            static const uint32_t point_weights168[]={4226995892,2945774658,2501814543,
            };
            static quant_filter_t point_filter168 = {96, 1, point_weights168, -4.1961775423260406e-06};
            point_filters_b[168] = point_filter168;
            
            static const uint32_t point_weights169[]={2180736210,780886198,3944155417,
            };
            static quant_filter_t point_filter169 = {96, 1, point_weights169, 2.392404439888196e-06};
            point_filters_b[169] = point_filter169;
            
            static const uint32_t point_weights170[]={1459754650,3720458338,2427825909,
            };
            static quant_filter_t point_filter170 = {96, 1, point_weights170, -3.538409600878367e-06};
            point_filters_b[170] = point_filter170;
            
            static const uint32_t point_weights171[]={1862895116,2369271545,3940737016,
            };
            static quant_filter_t point_filter171 = {96, 1, point_weights171, -3.8033265354897594e-06};
            point_filters_b[171] = point_filter171;
            
            static const uint32_t point_weights172[]={520489362,1840457851,3927243261,
            };
            static quant_filter_t point_filter172 = {96, 1, point_weights172, 4.530438218353083e-06};
            point_filters_b[172] = point_filter172;
            
            static const uint32_t point_weights173[]={2042983999,2130230587,307800935,
            };
            static quant_filter_t point_filter173 = {96, 1, point_weights173, -1.96159726328915e-06};
            point_filters_b[173] = point_filter173;
            
            static const uint32_t point_weights174[]={3371099887,3141014578,1597312010,
            };
            static quant_filter_t point_filter174 = {96, 1, point_weights174, -2.5804704364418285e-06};
            point_filters_b[174] = point_filter174;
            
            static const uint32_t point_weights175[]={3936766799,1552200556,62769212,
            };
            static quant_filter_t point_filter175 = {96, 1, point_weights175, -2.9667592116311425e-06};
            point_filters_b[175] = point_filter175;
            
            static const uint32_t point_weights176[]={987706479,406885130,387756805,
            };
            static quant_filter_t point_filter176 = {96, 1, point_weights176, 3.950000063923653e-06};
            point_filters_b[176] = point_filter176;
            
            static const uint32_t point_weights177[]={2049553671,3268220611,3915336446,
            };
            static quant_filter_t point_filter177 = {96, 1, point_weights177, 6.00346822920983e-07};
            point_filters_b[177] = point_filter177;
            
            static const uint32_t point_weights178[]={1917311617,3273974428,164367124,
            };
            static quant_filter_t point_filter178 = {96, 1, point_weights178, -6.559939720318653e-06};
            point_filters_b[178] = point_filter178;
            
            static const uint32_t point_weights179[]={4172330818,354770246,3768004154,
            };
            static quant_filter_t point_filter179 = {96, 1, point_weights179, -5.10534437125898e-06};
            point_filters_b[179] = point_filter179;
            
            static const uint32_t point_weights180[]={2188750312,2952436161,3018199194,
            };
            static quant_filter_t point_filter180 = {96, 1, point_weights180, 9.346231308882125e-06};
            point_filters_b[180] = point_filter180;
            
            static const uint32_t point_weights181[]={3919437119,1860109053,644873514,
            };
            static quant_filter_t point_filter181 = {96, 1, point_weights181, -2.6578554752632044e-06};
            point_filters_b[181] = point_filter181;
            
            static const uint32_t point_weights182[]={948466931,3641271874,2593526564,
            };
            static quant_filter_t point_filter182 = {96, 1, point_weights182, 3.320740688650403e-06};
            point_filters_b[182] = point_filter182;
            
            static const uint32_t point_weights183[]={1572366463,4048525383,2775206320,
            };
            static quant_filter_t point_filter183 = {96, 1, point_weights183, -4.3765530790551566e-06};
            point_filters_b[183] = point_filter183;
            
            static const uint32_t point_weights184[]={2868021540,2209636446,3510728092,
            };
            static quant_filter_t point_filter184 = {96, 1, point_weights184, 5.2113496167294215e-06};
            point_filters_b[184] = point_filter184;
            
            static const uint32_t point_weights185[]={958950486,1811711632,2720412643,
            };
            static quant_filter_t point_filter185 = {96, 1, point_weights185, -7.828404591236904e-07};
            point_filters_b[185] = point_filter185;
            
            static const uint32_t point_weights186[]={2847680901,1072864280,2427978976,
            };
            static quant_filter_t point_filter186 = {96, 1, point_weights186, -3.572660261852434e-06};
            point_filters_b[186] = point_filter186;
            
            static const uint32_t point_weights187[]={329044718,446847851,3845209622,
            };
            static quant_filter_t point_filter187 = {96, 1, point_weights187, 5.776395028078696e-06};
            point_filters_b[187] = point_filter187;
            
            static const uint32_t point_weights188[]={49673011,2760247388,2986055092,
            };
            static quant_filter_t point_filter188 = {96, 1, point_weights188, 1.0064957223221427e-06};
            point_filters_b[188] = point_filter188;
            
            static const uint32_t point_weights189[]={1941027334,2835795046,1203170382,
            };
            static quant_filter_t point_filter189 = {96, 1, point_weights189, 1.4404154171643313e-06};
            point_filters_b[189] = point_filter189;
            
            static const uint32_t point_weights190[]={3083809218,2129996772,1274376451,
            };
            static quant_filter_t point_filter190 = {96, 1, point_weights190, -2.9986308618390467e-06};
            point_filters_b[190] = point_filter190;
            
            static const uint32_t point_weights191[]={1835225241,1890643800,2241289793,
            };
            static quant_filter_t point_filter191 = {96, 1, point_weights191, 7.1499480327474885e-06};
            point_filters_b[191] = point_filter191;
            
            static const uint32_t point_weights192[]={1037639767,2294299680,1690500492,
            };
            static quant_filter_t point_filter192 = {96, 1, point_weights192, -6.51445930088812e-07};
            point_filters_b[192] = point_filter192;
            
            static const uint32_t point_weights193[]={4212742924,3053037151,2981891646,
            };
            static quant_filter_t point_filter193 = {96, 1, point_weights193, -1.538631636321952e-06};
            point_filters_b[193] = point_filter193;
            
            static const uint32_t point_weights194[]={3178207811,3532788941,3785754977,
            };
            static quant_filter_t point_filter194 = {96, 1, point_weights194, 3.153486886731116e-06};
            point_filters_b[194] = point_filter194;
            
            static const uint32_t point_weights195[]={1462893543,2715145393,2965977529,
            };
            static quant_filter_t point_filter195 = {96, 1, point_weights195, 7.080403065629071e-06};
            point_filters_b[195] = point_filter195;
            
            static const uint32_t point_weights196[]={1617169619,91638209,746528383,
            };
            static quant_filter_t point_filter196 = {96, 1, point_weights196, 3.1564502478431677e-06};
            point_filters_b[196] = point_filter196;
            
            static const uint32_t point_weights197[]={1800324375,2939743055,410113335,
            };
            static quant_filter_t point_filter197 = {96, 1, point_weights197, -6.516462462968775e-07};
            point_filters_b[197] = point_filter197;
            
            static const uint32_t point_weights198[]={986043436,4093948440,1932421339,
            };
            static quant_filter_t point_filter198 = {96, 1, point_weights198, -6.748837677150732e-06};
            point_filters_b[198] = point_filter198;
            
            static const uint32_t point_weights199[]={3222051999,3157120745,901250841,
            };
            static quant_filter_t point_filter199 = {96, 1, point_weights199, 4.1188013000237333e-08};
            point_filters_b[199] = point_filter199;
            
            static const uint32_t point_weights200[]={1736730927,2416990939,1116126108,
            };
            static quant_filter_t point_filter200 = {96, 1, point_weights200, 1.4395389371202327e-05};
            point_filters_b[200] = point_filter200;
            
            static const uint32_t point_weights201[]={3355883059,1866607600,1416235647,
            };
            static quant_filter_t point_filter201 = {96, 1, point_weights201, -1.3049641893303487e-05};
            point_filters_b[201] = point_filter201;
            
            static const uint32_t point_weights202[]={2101207151,1865244885,4109617868,
            };
            static quant_filter_t point_filter202 = {96, 1, point_weights202, -2.0572220194026158e-07};
            point_filters_b[202] = point_filter202;
            
            static const uint32_t point_weights203[]={1653781588,1995025459,1843263814,
            };
            static quant_filter_t point_filter203 = {96, 1, point_weights203, 5.476686055772007e-07};
            point_filters_b[203] = point_filter203;
            
            static const uint32_t point_weights204[]={673076541,2748166843,2983226194,
            };
            static quant_filter_t point_filter204 = {96, 1, point_weights204, -3.027397042387747e-06};
            point_filters_b[204] = point_filter204;
            
            static const uint32_t point_weights205[]={57437706,556288243,337363675,
            };
            static quant_filter_t point_filter205 = {96, 1, point_weights205, 4.12029066865216e-06};
            point_filters_b[205] = point_filter205;
            
            static const uint32_t point_weights206[]={1848703857,654038280,2344662665,
            };
            static quant_filter_t point_filter206 = {96, 1, point_weights206, -9.973275155061856e-06};
            point_filters_b[206] = point_filter206;
            
            static const uint32_t point_weights207[]={3854550389,1124081668,1115518816,
            };
            static quant_filter_t point_filter207 = {96, 1, point_weights207, -1.448095986233966e-06};
            point_filters_b[207] = point_filter207;
            
            static const uint32_t point_weights208[]={337024767,3852124848,3027310957,
            };
            static quant_filter_t point_filter208 = {96, 1, point_weights208, -1.1211097444174811e-05};
            point_filters_b[208] = point_filter208;
            
            static const uint32_t point_weights209[]={827111460,97881117,3924498584,
            };
            static quant_filter_t point_filter209 = {96, 1, point_weights209, 1.675569706094393e-06};
            point_filters_b[209] = point_filter209;
            
            static const uint32_t point_weights210[]={1772637721,3835683067,1437330832,
            };
            static quant_filter_t point_filter210 = {96, 1, point_weights210, 8.743004400457721e-06};
            point_filters_b[210] = point_filter210;
            
            static const uint32_t point_weights211[]={643890877,295858058,2686749033,
            };
            static quant_filter_t point_filter211 = {96, 1, point_weights211, 9.32469504277833e-07};
            point_filters_b[211] = point_filter211;
            
            static const uint32_t point_weights212[]={2819856428,2694142313,2468837710,
            };
            static quant_filter_t point_filter212 = {96, 1, point_weights212, 4.7215394261002075e-06};
            point_filters_b[212] = point_filter212;
            
            static const uint32_t point_weights213[]={1900067984,1926735033,2356559399,
            };
            static quant_filter_t point_filter213 = {96, 1, point_weights213, 5.478587809193414e-06};
            point_filters_b[213] = point_filter213;
            
            static const uint32_t point_weights214[]={2969530420,3306251931,2340676836,
            };
            static quant_filter_t point_filter214 = {96, 1, point_weights214, -1.1939169780816883e-05};
            point_filters_b[214] = point_filter214;
            
            static const uint32_t point_weights215[]={1515258707,940402037,3362655554,
            };
            static quant_filter_t point_filter215 = {96, 1, point_weights215, -2.2742536032183125e-07};
            point_filters_b[215] = point_filter215;
            
            static const uint32_t point_weights216[]={1030169124,3862582366,1518216547,
            };
            static quant_filter_t point_filter216 = {96, 1, point_weights216, 5.785015844139707e-08};
            point_filters_b[216] = point_filter216;
            
            static const uint32_t point_weights217[]={2819615202,1692757437,1566581887,
            };
            static quant_filter_t point_filter217 = {96, 1, point_weights217, 1.2456369404389989e-05};
            point_filters_b[217] = point_filter217;
            
            static const uint32_t point_weights218[]={1751625104,1759592747,450088958,
            };
            static quant_filter_t point_filter218 = {96, 1, point_weights218, 4.5795795244885085e-07};
            point_filters_b[218] = point_filter218;
            
            static const uint32_t point_weights219[]={1477888303,856932125,1051742248,
            };
            static quant_filter_t point_filter219 = {96, 1, point_weights219, -6.3359548221342266e-06};
            point_filters_b[219] = point_filter219;
            
            static const uint32_t point_weights220[]={2457884017,2252881050,3598669606,
            };
            static quant_filter_t point_filter220 = {96, 1, point_weights220, 8.019485903787427e-06};
            point_filters_b[220] = point_filter220;
            
            static const uint32_t point_weights221[]={1127799857,2139709975,3960840027,
            };
            static quant_filter_t point_filter221 = {96, 1, point_weights221, -6.390985731741239e-07};
            point_filters_b[221] = point_filter221;
            
            static const uint32_t point_weights222[]={2051672408,2244172328,2517987766,
            };
            static quant_filter_t point_filter222 = {96, 1, point_weights222, 4.006851213489426e-06};
            point_filters_b[222] = point_filter222;
            
            static const uint32_t point_weights223[]={3050235666,1447428576,2506075235,
            };
            static quant_filter_t point_filter223 = {96, 1, point_weights223, -2.004952193601639e-06};
            point_filters_b[223] = point_filter223;
            
            static const uint32_t point_weights224[]={4208609711,2173480569,438693386,
            };
            static quant_filter_t point_filter224 = {96, 1, point_weights224, -2.416528332105372e-06};
            point_filters_b[224] = point_filter224;
            
            static const uint32_t point_weights225[]={112559245,3123929683,2625443468,
            };
            static quant_filter_t point_filter225 = {96, 1, point_weights225, 7.100517791513994e-07};
            point_filters_b[225] = point_filter225;
            
            static const uint32_t point_weights226[]={4188789946,211722295,369835161,
            };
            static quant_filter_t point_filter226 = {96, 1, point_weights226, -7.321498742385302e-06};
            point_filters_b[226] = point_filter226;
            
            static const uint32_t point_weights227[]={3983422908,565889679,3020501767,
            };
            static quant_filter_t point_filter227 = {96, 1, point_weights227, -2.2926669771550223e-06};
            point_filters_b[227] = point_filter227;
            
            static const uint32_t point_weights228[]={3340024378,1204752729,3124188099,
            };
            static quant_filter_t point_filter228 = {96, 1, point_weights228, 4.986947601537395e-07};
            point_filters_b[228] = point_filter228;
            
            static const uint32_t point_weights229[]={1012768790,146137071,4286804123,
            };
            static quant_filter_t point_filter229 = {96, 1, point_weights229, -4.0596123653813265e-06};
            point_filters_b[229] = point_filter229;
            
            static const uint32_t point_weights230[]={1320761749,908998078,3312397359,
            };
            static quant_filter_t point_filter230 = {96, 1, point_weights230, 1.0793582077894825e-05};
            point_filters_b[230] = point_filter230;
            
            static const uint32_t point_weights231[]={1681481186,3806815840,2949798518,
            };
            static quant_filter_t point_filter231 = {96, 1, point_weights231, 2.1189885046624113e-06};
            point_filters_b[231] = point_filter231;
            
            static const uint32_t point_weights232[]={1878849130,2810148656,3858077079,
            };
            static quant_filter_t point_filter232 = {96, 1, point_weights232, 1.135208435698587e-06};
            point_filters_b[232] = point_filter232;
            
            static const uint32_t point_weights233[]={1757684466,481352413,1960780620,
            };
            static quant_filter_t point_filter233 = {96, 1, point_weights233, 1.3164374195184791e-06};
            point_filters_b[233] = point_filter233;
            
            static const uint32_t point_weights234[]={3299616108,15863675,2592060206,
            };
            static quant_filter_t point_filter234 = {96, 1, point_weights234, -6.95986443588481e-07};
            point_filters_b[234] = point_filter234;
            
            static const uint32_t point_weights235[]={2694367728,356706606,541582527,
            };
            static quant_filter_t point_filter235 = {96, 1, point_weights235, 1.6613081754712766e-07};
            point_filters_b[235] = point_filter235;
            
            static const uint32_t point_weights236[]={3137283431,233144553,428671190,
            };
            static quant_filter_t point_filter236 = {96, 1, point_weights236, -5.729681333832559e-07};
            point_filters_b[236] = point_filter236;
            
            static const uint32_t point_weights237[]={122752584,2641941019,717796763,
            };
            static quant_filter_t point_filter237 = {96, 1, point_weights237, -4.187495960650267e-06};
            point_filters_b[237] = point_filter237;
            
            static const uint32_t point_weights238[]={2278240461,4255492386,178851419,
            };
            static quant_filter_t point_filter238 = {96, 1, point_weights238, -4.946250101056648e-06};
            point_filters_b[238] = point_filter238;
            
            static const uint32_t point_weights239[]={2234914448,1753573025,1173175765,
            };
            static quant_filter_t point_filter239 = {96, 1, point_weights239, 1.1976851510553388e-06};
            point_filters_b[239] = point_filter239;
            
            static const uint32_t point_weights240[]={2400414953,394280725,3749043628,
            };
            static quant_filter_t point_filter240 = {96, 1, point_weights240, -4.871843884757254e-06};
            point_filters_b[240] = point_filter240;
            
            static const uint32_t point_weights241[]={895250506,4023001376,3230848207,
            };
            static quant_filter_t point_filter241 = {96, 1, point_weights241, 7.255062882904895e-06};
            point_filters_b[241] = point_filter241;
            
            static const uint32_t point_weights242[]={2496789096,3102701561,2606908873,
            };
            static quant_filter_t point_filter242 = {96, 1, point_weights242, 4.2451617332517344e-07};
            point_filters_b[242] = point_filter242;
            
            static const uint32_t point_weights243[]={602642848,1215020722,264211941,
            };
            static quant_filter_t point_filter243 = {96, 1, point_weights243, 1.2549739949463401e-05};
            point_filters_b[243] = point_filter243;
            
            static const uint32_t point_weights244[]={2453342781,1200498022,3286870521,
            };
            static quant_filter_t point_filter244 = {96, 1, point_weights244, -8.651760253997054e-06};
            point_filters_b[244] = point_filter244;
            
            static const uint32_t point_weights245[]={2623845043,1360912880,4187512122,
            };
            static quant_filter_t point_filter245 = {96, 1, point_weights245, 7.637121939296776e-07};
            point_filters_b[245] = point_filter245;
            
            static const uint32_t point_weights246[]={1099835049,3353668786,1817475926,
            };
            static quant_filter_t point_filter246 = {96, 1, point_weights246, 1.726305413285445e-06};
            point_filters_b[246] = point_filter246;
            
            static const uint32_t point_weights247[]={2000480117,1483602354,3634640031,
            };
            static quant_filter_t point_filter247 = {96, 1, point_weights247, 1.3543509567170986e-06};
            point_filters_b[247] = point_filter247;
            
            static const uint32_t point_weights248[]={3058034659,2199470098,3355192792,
            };
            static quant_filter_t point_filter248 = {96, 1, point_weights248, -8.76692411111435e-06};
            point_filters_b[248] = point_filter248;
            
            static const uint32_t point_weights249[]={886688842,830720091,3219943309,
            };
            static quant_filter_t point_filter249 = {96, 1, point_weights249, 5.012548172089737e-06};
            point_filters_b[249] = point_filter249;
            
            static const uint32_t point_weights250[]={1137716994,679473078,1903171354,
            };
            static quant_filter_t point_filter250 = {96, 1, point_weights250, 3.4819634038285585e-06};
            point_filters_b[250] = point_filter250;
            
            static const uint32_t point_weights251[]={3137353343,1993013247,3423519604,
            };
            static quant_filter_t point_filter251 = {96, 1, point_weights251, -4.992580670659663e-06};
            point_filters_b[251] = point_filter251;
            
            static const uint32_t point_weights252[]={827445109,1713014410,683250008,
            };
            static quant_filter_t point_filter252 = {96, 1, point_weights252, 3.5231009860581253e-06};
            point_filters_b[252] = point_filter252;
            
            static const uint32_t point_weights253[]={1670660018,2605753538,1803780384,
            };
            static quant_filter_t point_filter253 = {96, 1, point_weights253, 1.7634674804867245e-06};
            point_filters_b[253] = point_filter253;
            
            static const uint32_t point_weights254[]={3290476960,1752249943,134554280,
            };
            static quant_filter_t point_filter254 = {96, 1, point_weights254, 1.0212185088676051e-06};
            point_filters_b[254] = point_filter254;
            
            static const uint32_t point_weights255[]={596646390,790212134,3570713322,
            };
            static quant_filter_t point_filter255 = {96, 1, point_weights255, 4.772300144395558e-06};
            point_filters_b[255] = point_filter255;
            
            static const uint32_t point_weights256[]={3048217386,2414504174,1379051095,
            };
            static quant_filter_t point_filter256 = {96, 1, point_weights256, -3.920850758731831e-06};
            point_filters_b[256] = point_filter256;
            
            static const uint32_t point_weights257[]={342495729,2046167224,3026404833,
            };
            static quant_filter_t point_filter257 = {96, 1, point_weights257, 5.00439819006715e-06};
            point_filters_b[257] = point_filter257;
            
            static const uint32_t point_weights258[]={77517569,3536694432,182243814,
            };
            static quant_filter_t point_filter258 = {96, 1, point_weights258, 1.1966083548031747e-05};
            point_filters_b[258] = point_filter258;
            
            static const uint32_t point_weights259[]={2182559110,2386448361,772810008,
            };
            static quant_filter_t point_filter259 = {96, 1, point_weights259, 4.444621936272597e-06};
            point_filters_b[259] = point_filter259;
            
            static const uint32_t point_weights260[]={230691713,1581110569,1701982921,
            };
            static quant_filter_t point_filter260 = {96, 1, point_weights260, -4.985534815205028e-06};
            point_filters_b[260] = point_filter260;
            
            static const uint32_t point_weights261[]={2880578145,377848736,451930435,
            };
            static quant_filter_t point_filter261 = {96, 1, point_weights261, -4.5390011109702755e-06};
            point_filters_b[261] = point_filter261;
            
            static const uint32_t point_weights262[]={1488974074,576225452,3379047855,
            };
            static quant_filter_t point_filter262 = {96, 1, point_weights262, -4.039770828967448e-06};
            point_filters_b[262] = point_filter262;
            
            static const uint32_t point_weights263[]={2864200743,3514203706,849241418,
            };
            static quant_filter_t point_filter263 = {96, 1, point_weights263, -1.1980613123796502e-07};
            point_filters_b[263] = point_filter263;
            
            static const uint32_t point_weights264[]={620124994,127430913,3120287643,
            };
            static quant_filter_t point_filter264 = {96, 1, point_weights264, 1.1146697943331674e-05};
            point_filters_b[264] = point_filter264;
            
            static const uint32_t point_weights265[]={3250519307,3269055943,2255157867,
            };
            static quant_filter_t point_filter265 = {96, 1, point_weights265, 7.93249455455225e-07};
            point_filters_b[265] = point_filter265;
            
            static const uint32_t point_weights266[]={1233056374,404002249,4196847212,
            };
            static quant_filter_t point_filter266 = {96, 1, point_weights266, -8.554578926123213e-06};
            point_filters_b[266] = point_filter266;
            
            static const uint32_t point_weights267[]={2919330762,4036578149,361207797,
            };
            static quant_filter_t point_filter267 = {96, 1, point_weights267, -3.0433800475293538e-06};
            point_filters_b[267] = point_filter267;
            
            static const uint32_t point_weights268[]={1305312499,4263466581,999790692,
            };
            static quant_filter_t point_filter268 = {96, 1, point_weights268, -8.346371942025144e-06};
            point_filters_b[268] = point_filter268;
            
            static const uint32_t point_weights269[]={3071714322,1703416633,1379265131,
            };
            static quant_filter_t point_filter269 = {96, 1, point_weights269, -3.474796130831237e-06};
            point_filters_b[269] = point_filter269;
            
            static const uint32_t point_weights270[]={4050967708,2244480862,1397138394,
            };
            static quant_filter_t point_filter270 = {96, 1, point_weights270, 1.6129272353282431e-06};
            point_filters_b[270] = point_filter270;
            
            static const uint32_t point_weights271[]={3252780596,3648833970,3480404116,
            };
            static quant_filter_t point_filter271 = {96, 1, point_weights271, 5.745803264289862e-07};
            point_filters_b[271] = point_filter271;
            
            static const uint32_t point_weights272[]={1324384402,3695953730,2324839034,
            };
            static quant_filter_t point_filter272 = {96, 1, point_weights272, -4.137684754823567e-06};
            point_filters_b[272] = point_filter272;
            
            static const uint32_t point_weights273[]={1574726286,2995142484,782387049,
            };
            static quant_filter_t point_filter273 = {96, 1, point_weights273, -2.6052293833345175e-06};
            point_filters_b[273] = point_filter273;
            
            static const uint32_t point_weights274[]={2948956070,736385156,3370506589,
            };
            static quant_filter_t point_filter274 = {96, 1, point_weights274, 6.36016466160072e-07};
            point_filters_b[274] = point_filter274;
            
            static const uint32_t point_weights275[]={849611095,1580059137,3090029081,
            };
            static quant_filter_t point_filter275 = {96, 1, point_weights275, -6.7387177296041045e-06};
            point_filters_b[275] = point_filter275;
            
            static const uint32_t point_weights276[]={2525729686,3329961261,2376324609,
            };
            static quant_filter_t point_filter276 = {96, 1, point_weights276, 1.0204838872596156e-05};
            point_filters_b[276] = point_filter276;
            
            static const uint32_t point_weights277[]={2041630579,1995996911,2626854202,
            };
            static quant_filter_t point_filter277 = {96, 1, point_weights277, -9.09980371943675e-06};
            point_filters_b[277] = point_filter277;
            
            static const uint32_t point_weights278[]={2765326740,3492430804,4283649865,
            };
            static quant_filter_t point_filter278 = {96, 1, point_weights278, -4.455721409613034e-06};
            point_filters_b[278] = point_filter278;
            
            static const uint32_t point_weights279[]={269519757,3105387984,2643311700,
            };
            static quant_filter_t point_filter279 = {96, 1, point_weights279, -4.825360520044342e-06};
            point_filters_b[279] = point_filter279;
            
            static const uint32_t point_weights280[]={3840389092,3062100823,3099513808,
            };
            static quant_filter_t point_filter280 = {96, 1, point_weights280, 8.141324542521033e-06};
            point_filters_b[280] = point_filter280;
            
            static const uint32_t point_weights281[]={1163669566,1688652969,1034065205,
            };
            static quant_filter_t point_filter281 = {96, 1, point_weights281, -5.669983693223912e-06};
            point_filters_b[281] = point_filter281;
            
            static const uint32_t point_weights282[]={215596000,3702459755,1766411671,
            };
            static quant_filter_t point_filter282 = {96, 1, point_weights282, -1.0681619642127771e-05};
            point_filters_b[282] = point_filter282;
            
            static const uint32_t point_weights283[]={588653982,4018614159,1335803644,
            };
            static quant_filter_t point_filter283 = {96, 1, point_weights283, -2.5961244318750687e-06};
            point_filters_b[283] = point_filter283;
            
            static const uint32_t point_weights284[]={82709329,4140353539,1007801832,
            };
            static quant_filter_t point_filter284 = {96, 1, point_weights284, 2.4042110453592613e-06};
            point_filters_b[284] = point_filter284;
            
            static const uint32_t point_weights285[]={1151205292,2174249883,398803033,
            };
            static quant_filter_t point_filter285 = {96, 1, point_weights285, 6.785719506297028e-06};
            point_filters_b[285] = point_filter285;
            
            static const uint32_t point_weights286[]={3483362774,1289988177,1956728577,
            };
            static quant_filter_t point_filter286 = {96, 1, point_weights286, -6.512724212370813e-06};
            point_filters_b[286] = point_filter286;
            
            static const uint32_t point_weights287[]={2127225514,3254767611,516304013,
            };
            static quant_filter_t point_filter287 = {96, 1, point_weights287, -5.568229880736908e-06};
            point_filters_b[287] = point_filter287;
            
            static const uint32_t point_weights288[]={1740596916,2530191006,1957479417,
            };
            static quant_filter_t point_filter288 = {96, 1, point_weights288, -4.9948225750995334e-06};
            point_filters_b[288] = point_filter288;
            
            static const uint32_t point_weights289[]={152331079,1143658765,345601621,
            };
            static quant_filter_t point_filter289 = {96, 1, point_weights289, 6.746304279658943e-06};
            point_filters_b[289] = point_filter289;
            
            static const uint32_t point_weights290[]={2717444301,1599885377,3205956892,
            };
            static quant_filter_t point_filter290 = {96, 1, point_weights290, -1.4102319028097554e-06};
            point_filters_b[290] = point_filter290;
            
            static const uint32_t point_weights291[]={1865752099,869621016,759198665,
            };
            static quant_filter_t point_filter291 = {96, 1, point_weights291, -2.784521768717241e-07};
            point_filters_b[291] = point_filter291;
            
            static const uint32_t point_weights292[]={1415227947,4252546285,617227318,
            };
            static quant_filter_t point_filter292 = {96, 1, point_weights292, 6.4710502556408755e-06};
            point_filters_b[292] = point_filter292;
            
            static const uint32_t point_weights293[]={4170100832,3573428366,3924276620,
            };
            static quant_filter_t point_filter293 = {96, 1, point_weights293, 7.689128324273042e-06};
            point_filters_b[293] = point_filter293;
            
            static const uint32_t point_weights294[]={2962281544,635229852,2507000292,
            };
            static quant_filter_t point_filter294 = {96, 1, point_weights294, 4.4759070760846953e-07};
            point_filters_b[294] = point_filter294;
            
            static const uint32_t point_weights295[]={4032060467,3260107950,4086685372,
            };
            static quant_filter_t point_filter295 = {96, 1, point_weights295, 2.955068339360878e-06};
            point_filters_b[295] = point_filter295;
            
            static const uint32_t point_weights296[]={3481910611,141300058,2279896482,
            };
            static quant_filter_t point_filter296 = {96, 1, point_weights296, -2.256815463397288e-07};
            point_filters_b[296] = point_filter296;
            
            static const uint32_t point_weights297[]={132051484,3092688191,689042105,
            };
            static quant_filter_t point_filter297 = {96, 1, point_weights297, 3.3019334750861162e-06};
            point_filters_b[297] = point_filter297;
            
            static const uint32_t point_weights298[]={4289415902,1456884349,3343808653,
            };
            static quant_filter_t point_filter298 = {96, 1, point_weights298, -2.405218992862501e-06};
            point_filters_b[298] = point_filter298;
            
            static const uint32_t point_weights299[]={2476009063,2049393035,103844851,
            };
            static quant_filter_t point_filter299 = {96, 1, point_weights299, -3.1307340577768628e-06};
            point_filters_b[299] = point_filter299;
            
            static const uint32_t point_weights300[]={156655757,1868149863,2267708722,
            };
            static quant_filter_t point_filter300 = {96, 1, point_weights300, 1.3263584719425126e-07};
            point_filters_b[300] = point_filter300;
            
            static const uint32_t point_weights301[]={1948618486,3482306054,1227379170,
            };
            static quant_filter_t point_filter301 = {96, 1, point_weights301, 4.15848262491636e-06};
            point_filters_b[301] = point_filter301;
            
            static const uint32_t point_weights302[]={3401183446,1062027628,3939160053,
            };
            static quant_filter_t point_filter302 = {96, 1, point_weights302, 1.5433655789820477e-05};
            point_filters_b[302] = point_filter302;
            
            static const uint32_t point_weights303[]={3546960670,2345440468,1223981992,
            };
            static quant_filter_t point_filter303 = {96, 1, point_weights303, -3.7427030292747077e-06};
            point_filters_b[303] = point_filter303;
            
            static const uint32_t point_weights304[]={2374516737,1839739060,1371312094,
            };
            static quant_filter_t point_filter304 = {96, 1, point_weights304, 2.269175411129254e-06};
            point_filters_b[304] = point_filter304;
            
            static const uint32_t point_weights305[]={2971561650,1220238845,3868487654,
            };
            static quant_filter_t point_filter305 = {96, 1, point_weights305, -6.840574314992409e-06};
            point_filters_b[305] = point_filter305;
            
            static const uint32_t point_weights306[]={396911485,1522329057,4208590982,
            };
            static quant_filter_t point_filter306 = {96, 1, point_weights306, 6.080973093958164e-07};
            point_filters_b[306] = point_filter306;
            
            static const uint32_t point_weights307[]={1998612914,3601815612,3466731926,
            };
            static quant_filter_t point_filter307 = {96, 1, point_weights307, 2.264913064209395e-06};
            point_filters_b[307] = point_filter307;
            
            static const uint32_t point_weights308[]={2192769008,332258910,653140546,
            };
            static quant_filter_t point_filter308 = {96, 1, point_weights308, -1.4531105989590287e-06};
            point_filters_b[308] = point_filter308;
            
            static const uint32_t point_weights309[]={1191598279,3902644662,3633482424,
            };
            static quant_filter_t point_filter309 = {96, 1, point_weights309, -1.0045201634056866e-05};
            point_filters_b[309] = point_filter309;
            
            static const uint32_t point_weights310[]={1739318873,1568190310,1555192305,
            };
            static quant_filter_t point_filter310 = {96, 1, point_weights310, 6.819085228926269e-06};
            point_filters_b[310] = point_filter310;
            
            static const uint32_t point_weights311[]={943993709,712075087,367490148,
            };
            static quant_filter_t point_filter311 = {96, 1, point_weights311, -2.880329930121661e-06};
            point_filters_b[311] = point_filter311;
            
            static const uint32_t point_weights312[]={957089621,395942892,2838170278,
            };
            static quant_filter_t point_filter312 = {96, 1, point_weights312, -1.6644185052427929e-06};
            point_filters_b[312] = point_filter312;
            
            static const uint32_t point_weights313[]={4238378817,3386940926,4292972693,
            };
            static quant_filter_t point_filter313 = {96, 1, point_weights313, 1.7090447954615229e-06};
            point_filters_b[313] = point_filter313;
            
            static const uint32_t point_weights314[]={86948592,1483164748,4014124661,
            };
            static quant_filter_t point_filter314 = {96, 1, point_weights314, -9.253057214664295e-06};
            point_filters_b[314] = point_filter314;
            
            static const uint32_t point_weights315[]={2349234005,2989078517,2824909970,
            };
            static quant_filter_t point_filter315 = {96, 1, point_weights315, -2.782613783836041e-08};
            point_filters_b[315] = point_filter315;
            
            static const uint32_t point_weights316[]={2808240945,233364103,2467964758,
            };
            static quant_filter_t point_filter316 = {96, 1, point_weights316, 1.0782737263070885e-06};
            point_filters_b[316] = point_filter316;
            
            static const uint32_t point_weights317[]={3999758125,403799091,1196770655,
            };
            static quant_filter_t point_filter317 = {96, 1, point_weights317, 2.0617517293430865e-07};
            point_filters_b[317] = point_filter317;
            
            static const uint32_t point_weights318[]={822357879,2519665995,180317076,
            };
            static quant_filter_t point_filter318 = {96, 1, point_weights318, 5.85769066674402e-06};
            point_filters_b[318] = point_filter318;
            
            static const uint32_t point_weights319[]={164314032,3338770525,114399708,
            };
            static quant_filter_t point_filter319 = {96, 1, point_weights319, 1.4786033943892107e-06};
            point_filters_b[319] = point_filter319;
            
            static const uint32_t point_weights320[]={2448948227,2807155416,1248149634,
            };
            static quant_filter_t point_filter320 = {96, 1, point_weights320, 4.221400104142958e-06};
            point_filters_b[320] = point_filter320;
            
            static const uint32_t point_weights321[]={2109831252,1837746811,1106540812,
            };
            static quant_filter_t point_filter321 = {96, 1, point_weights321, 7.870142439969641e-07};
            point_filters_b[321] = point_filter321;
            
            static const uint32_t point_weights322[]={1273607860,1246665214,1299285508,
            };
            static quant_filter_t point_filter322 = {96, 1, point_weights322, -8.500480362272356e-06};
            point_filters_b[322] = point_filter322;
            
            static const uint32_t point_weights323[]={1294526494,1403930628,1429698409,
            };
            static quant_filter_t point_filter323 = {96, 1, point_weights323, -2.1550302875539273e-08};
            point_filters_b[323] = point_filter323;
            
            static const uint32_t point_weights324[]={87283235,230139874,4077632527,
            };
            static quant_filter_t point_filter324 = {96, 1, point_weights324, -2.3975464955583448e-06};
            point_filters_b[324] = point_filter324;
            
            static const uint32_t point_weights325[]={1738530941,3536030068,1602827035,
            };
            static quant_filter_t point_filter325 = {96, 1, point_weights325, 5.452718028209347e-07};
            point_filters_b[325] = point_filter325;
            
            static const uint32_t point_weights326[]={3365263860,1954423667,3252284549,
            };
            static quant_filter_t point_filter326 = {96, 1, point_weights326, 2.132934014298371e-06};
            point_filters_b[326] = point_filter326;
            
            static const uint32_t point_weights327[]={3297484266,1782683190,4066172364,
            };
            static quant_filter_t point_filter327 = {96, 1, point_weights327, -6.611274784518173e-06};
            point_filters_b[327] = point_filter327;
            
            static const uint32_t point_weights328[]={2382881422,188907407,3754754383,
            };
            static quant_filter_t point_filter328 = {96, 1, point_weights328, 7.832665687601548e-06};
            point_filters_b[328] = point_filter328;
            
            static const uint32_t point_weights329[]={433648754,3281119036,2385382405,
            };
            static quant_filter_t point_filter329 = {96, 1, point_weights329, 2.6757757041195873e-06};
            point_filters_b[329] = point_filter329;
            
            static const uint32_t point_weights330[]={2910760129,1282589595,1664525871,
            };
            static quant_filter_t point_filter330 = {96, 1, point_weights330, -4.374459876999026e-06};
            point_filters_b[330] = point_filter330;
            
            static const uint32_t point_weights331[]={2782435409,3384939098,2648274412,
            };
            static quant_filter_t point_filter331 = {96, 1, point_weights331, 1.6833768086144119e-06};
            point_filters_b[331] = point_filter331;
            
            static const uint32_t point_weights332[]={1975577709,3066547067,2692588992,
            };
            static quant_filter_t point_filter332 = {96, 1, point_weights332, 1.4975114481785567e-06};
            point_filters_b[332] = point_filter332;
            
            static const uint32_t point_weights333[]={2047512929,4292416524,3827843997,
            };
            static quant_filter_t point_filter333 = {96, 1, point_weights333, 9.29005091165891e-06};
            point_filters_b[333] = point_filter333;
            
            static const uint32_t point_weights334[]={3080718965,1374472836,1060801205,
            };
            static quant_filter_t point_filter334 = {96, 1, point_weights334, 4.166691724094562e-06};
            point_filters_b[334] = point_filter334;
            
            static const uint32_t point_weights335[]={1385757394,2382583099,1197327901,
            };
            static quant_filter_t point_filter335 = {96, 1, point_weights335, 1.820573288568994e-06};
            point_filters_b[335] = point_filter335;
            
            static const uint32_t point_weights336[]={1320401121,1590433737,3304807851,
            };
            static quant_filter_t point_filter336 = {96, 1, point_weights336, 5.8759242165251635e-06};
            point_filters_b[336] = point_filter336;
            
            static const uint32_t point_weights337[]={3151344009,3186205398,918003508,
            };
            static quant_filter_t point_filter337 = {96, 1, point_weights337, 3.404860308364732e-06};
            point_filters_b[337] = point_filter337;
            
            static const uint32_t point_weights338[]={670983853,3263990618,3757118210,
            };
            static quant_filter_t point_filter338 = {96, 1, point_weights338, -7.001417088758899e-06};
            point_filters_b[338] = point_filter338;
            
            static const uint32_t point_weights339[]={1739499007,2874972033,3622290963,
            };
            static quant_filter_t point_filter339 = {96, 1, point_weights339, 1.214772419189103e-05};
            point_filters_b[339] = point_filter339;
            
            static const uint32_t point_weights340[]={329248790,538017541,1964429007,
            };
            static quant_filter_t point_filter340 = {96, 1, point_weights340, -3.1110489544516895e-06};
            point_filters_b[340] = point_filter340;
            
            static const uint32_t point_weights341[]={2146170193,2518011456,4067681727,
            };
            static quant_filter_t point_filter341 = {96, 1, point_weights341, 4.221202516418998e-07};
            point_filters_b[341] = point_filter341;
            
            static const uint32_t point_weights342[]={1058451112,575446620,2666232912,
            };
            static quant_filter_t point_filter342 = {96, 1, point_weights342, 2.8026813652104465e-06};
            point_filters_b[342] = point_filter342;
            
            static const uint32_t point_weights343[]={3848643698,224500730,1795337117,
            };
            static quant_filter_t point_filter343 = {96, 1, point_weights343, -3.140825981517992e-07};
            point_filters_b[343] = point_filter343;
            
            static const uint32_t point_weights344[]={891584030,3540622958,152444810,
            };
            static quant_filter_t point_filter344 = {96, 1, point_weights344, 2.8177073545521125e-06};
            point_filters_b[344] = point_filter344;
            
            static const uint32_t point_weights345[]={3404067276,2971316589,2177635949,
            };
            static quant_filter_t point_filter345 = {96, 1, point_weights345, -1.1933690984733403e-05};
            point_filters_b[345] = point_filter345;
            
            static const uint32_t point_weights346[]={870253349,1077250913,3713656331,
            };
            static quant_filter_t point_filter346 = {96, 1, point_weights346, 1.351562787021976e-05};
            point_filters_b[346] = point_filter346;
            
            static const uint32_t point_weights347[]={2254620690,2334663266,1067466980,
            };
            static quant_filter_t point_filter347 = {96, 1, point_weights347, 4.99343059345847e-07};
            point_filters_b[347] = point_filter347;
            
            static const uint32_t point_weights348[]={3049122462,3725962557,2208608742,
            };
            static quant_filter_t point_filter348 = {96, 1, point_weights348, 7.56269537305343e-06};
            point_filters_b[348] = point_filter348;
            
            static const uint32_t point_weights349[]={3694896572,2945498950,4118206098,
            };
            static quant_filter_t point_filter349 = {96, 1, point_weights349, 1.4988787597758346e-06};
            point_filters_b[349] = point_filter349;
            
            static const uint32_t point_weights350[]={1771553072,2090762211,3343465211,
            };
            static quant_filter_t point_filter350 = {96, 1, point_weights350, -4.867562893196009e-06};
            point_filters_b[350] = point_filter350;
            
            static const uint32_t point_weights351[]={4066391685,721772776,2096469516,
            };
            static quant_filter_t point_filter351 = {96, 1, point_weights351, -7.696338798268698e-06};
            point_filters_b[351] = point_filter351;
            
            static const uint32_t point_weights352[]={2600329893,2274090070,3282066205,
            };
            static quant_filter_t point_filter352 = {96, 1, point_weights352, -6.3814677560003474e-06};
            point_filters_b[352] = point_filter352;
            
            static const uint32_t point_weights353[]={1318049059,1070172954,4101206425,
            };
            static quant_filter_t point_filter353 = {96, 1, point_weights353, -1.0315029612684157e-06};
            point_filters_b[353] = point_filter353;
            
            static const uint32_t point_weights354[]={1405010200,1879178360,1918729110,
            };
            static quant_filter_t point_filter354 = {96, 1, point_weights354, 4.147158563228004e-07};
            point_filters_b[354] = point_filter354;
            
            static const uint32_t point_weights355[]={1981836464,740087373,1349743346,
            };
            static quant_filter_t point_filter355 = {96, 1, point_weights355, 2.2812854183484887e-07};
            point_filters_b[355] = point_filter355;
            
            static const uint32_t point_weights356[]={1501393502,618685127,3902218179,
            };
            static quant_filter_t point_filter356 = {96, 1, point_weights356, 3.219602604076499e-06};
            point_filters_b[356] = point_filter356;
            
            static const uint32_t point_weights357[]={3550255159,1384084429,2607417556,
            };
            static quant_filter_t point_filter357 = {96, 1, point_weights357, 4.611487838701578e-06};
            point_filters_b[357] = point_filter357;
            
            static const uint32_t point_weights358[]={1476513815,3133622459,1907995411,
            };
            static quant_filter_t point_filter358 = {96, 1, point_weights358, 5.1275223995617125e-06};
            point_filters_b[358] = point_filter358;
            
            static const uint32_t point_weights359[]={3857018794,1391139576,3983855100,
            };
            static quant_filter_t point_filter359 = {96, 1, point_weights359, 1.0246582178297103e-06};
            point_filters_b[359] = point_filter359;
            
            static const uint32_t point_weights360[]={3603112406,1805034973,404110210,
            };
            static quant_filter_t point_filter360 = {96, 1, point_weights360, -1.119275552241561e-07};
            point_filters_b[360] = point_filter360;
            
            static const uint32_t point_weights361[]={2934158440,2328890614,1882182228,
            };
            static quant_filter_t point_filter361 = {96, 1, point_weights361, 2.1516814285860164e-06};
            point_filters_b[361] = point_filter361;
            
            static const uint32_t point_weights362[]={1512621380,1456126663,3836400657,
            };
            static quant_filter_t point_filter362 = {96, 1, point_weights362, 1.5253650417434983e-05};
            point_filters_b[362] = point_filter362;
            
            static const uint32_t point_weights363[]={3982221114,2939418507,599694318,
            };
            static quant_filter_t point_filter363 = {96, 1, point_weights363, 8.365982466784772e-07};
            point_filters_b[363] = point_filter363;
            
            static const uint32_t point_weights364[]={3222135237,1008338809,1498480174,
            };
            static quant_filter_t point_filter364 = {96, 1, point_weights364, -3.037952637896524e-06};
            point_filters_b[364] = point_filter364;
            
            static const uint32_t point_weights365[]={1020051576,733801015,699802000,
            };
            static quant_filter_t point_filter365 = {96, 1, point_weights365, 1.1352572073519696e-06};
            point_filters_b[365] = point_filter365;
            
            static const uint32_t point_weights366[]={755005918,1726756551,1199156582,
            };
            static quant_filter_t point_filter366 = {96, 1, point_weights366, 1.387265911034774e-06};
            point_filters_b[366] = point_filter366;
            
            static const uint32_t point_weights367[]={4010097110,3243936299,2425864694,
            };
            static quant_filter_t point_filter367 = {96, 1, point_weights367, 1.9625003915280104e-06};
            point_filters_b[367] = point_filter367;
            
            static const uint32_t point_weights368[]={489023758,4177700054,235807317,
            };
            static quant_filter_t point_filter368 = {96, 1, point_weights368, -2.799793492158642e-06};
            point_filters_b[368] = point_filter368;
            
            static const uint32_t point_weights369[]={1350796677,2618308756,52820120,
            };
            static quant_filter_t point_filter369 = {96, 1, point_weights369, -3.833443770417944e-06};
            point_filters_b[369] = point_filter369;
            
            static const uint32_t point_weights370[]={3559621025,28504753,1831112686,
            };
            static quant_filter_t point_filter370 = {96, 1, point_weights370, 3.772823902181699e-06};
            point_filters_b[370] = point_filter370;
            
            static const uint32_t point_weights371[]={1213713587,1636929371,2550071317,
            };
            static quant_filter_t point_filter371 = {96, 1, point_weights371, 7.313047262869077e-06};
            point_filters_b[371] = point_filter371;
            
            static const uint32_t point_weights372[]={3045332652,1914962135,3366717825,
            };
            static quant_filter_t point_filter372 = {96, 1, point_weights372, -1.0128086614713538e-05};
            point_filters_b[372] = point_filter372;
            
            static const uint32_t point_weights373[]={3383581767,1660331768,2207894446,
            };
            static quant_filter_t point_filter373 = {96, 1, point_weights373, 5.5610821618756745e-06};
            point_filters_b[373] = point_filter373;
            
            static const uint32_t point_weights374[]={1952851856,1858976591,2684309777,
            };
            static quant_filter_t point_filter374 = {96, 1, point_weights374, -2.0685226900241105e-06};
            point_filters_b[374] = point_filter374;
            
            static const uint32_t point_weights375[]={394987552,870154195,3618978003,
            };
            static quant_filter_t point_filter375 = {96, 1, point_weights375, -4.4414596231945325e-06};
            point_filters_b[375] = point_filter375;
            
            static const uint32_t point_weights376[]={1669376141,1029146644,3919260780,
            };
            static quant_filter_t point_filter376 = {96, 1, point_weights376, 3.59802561433753e-06};
            point_filters_b[376] = point_filter376;
            
            static const uint32_t point_weights377[]={3918240034,711084340,1319124663,
            };
            static quant_filter_t point_filter377 = {96, 1, point_weights377, 5.157798113941681e-06};
            point_filters_b[377] = point_filter377;
            
            static const uint32_t point_weights378[]={2928534647,4185037033,1693230694,
            };
            static quant_filter_t point_filter378 = {96, 1, point_weights378, -4.366623670648551e-06};
            point_filters_b[378] = point_filter378;
            
            static const uint32_t point_weights379[]={1730256685,2618937438,1501029171,
            };
            static quant_filter_t point_filter379 = {96, 1, point_weights379, -4.443692887434736e-06};
            point_filters_b[379] = point_filter379;
            
            static const uint32_t point_weights380[]={3121272240,270628808,3601289891,
            };
            static quant_filter_t point_filter380 = {96, 1, point_weights380, -9.155362477031304e-07};
            point_filters_b[380] = point_filter380;
            
            static const uint32_t point_weights381[]={3874218399,2934927094,3164284993,
            };
            static quant_filter_t point_filter381 = {96, 1, point_weights381, 5.573902853939217e-06};
            point_filters_b[381] = point_filter381;
            
            static const uint32_t point_weights382[]={1470240451,2656397013,108968574,
            };
            static quant_filter_t point_filter382 = {96, 1, point_weights382, -5.636837840938824e-07};
            point_filters_b[382] = point_filter382;
            
            static const uint32_t point_weights383[]={1756322973,2158596095,3984469821,
            };
            static quant_filter_t point_filter383 = {96, 1, point_weights383, 1.7829551097747753e-06};
            point_filters_b[383] = point_filter383;
            
            static const uint32_t point_weights384[]={4044070205,2073938042,1147921770,
            };
            static quant_filter_t point_filter384 = {96, 1, point_weights384, -3.019919631697121e-06};
            point_filters_b[384] = point_filter384;
            
            static const uint32_t point_weights385[]={1132573093,1362695097,2537551842,
            };
            static quant_filter_t point_filter385 = {96, 1, point_weights385, -1.1343839645405751e-07};
            point_filters_b[385] = point_filter385;
            
            static const uint32_t point_weights386[]={61642009,1357699210,2756606754,
            };
            static quant_filter_t point_filter386 = {96, 1, point_weights386, 4.1481657717667986e-06};
            point_filters_b[386] = point_filter386;
            
            static const uint32_t point_weights387[]={644782158,4067978081,2488656745,
            };
            static quant_filter_t point_filter387 = {96, 1, point_weights387, -4.900531621387927e-06};
            point_filters_b[387] = point_filter387;
            
            static const uint32_t point_weights388[]={577591514,3771092466,2214914940,
            };
            static quant_filter_t point_filter388 = {96, 1, point_weights388, 7.854499017412309e-06};
            point_filters_b[388] = point_filter388;
            
            static const uint32_t point_weights389[]={1635990396,3011505137,3933496381,
            };
            static quant_filter_t point_filter389 = {96, 1, point_weights389, 2.9972540005474e-07};
            point_filters_b[389] = point_filter389;
            
            static const uint32_t point_weights390[]={3612870233,3194083555,4257564823,
            };
            static quant_filter_t point_filter390 = {96, 1, point_weights390, -3.637466306827264e-06};
            point_filters_b[390] = point_filter390;
            
            static const uint32_t point_weights391[]={1774700040,956925594,2522706587,
            };
            static quant_filter_t point_filter391 = {96, 1, point_weights391, -1.4376121271197917e-06};
            point_filters_b[391] = point_filter391;
            
            static const uint32_t point_weights392[]={94757049,2247916020,3488073849,
            };
            static quant_filter_t point_filter392 = {96, 1, point_weights392, 5.577795036515454e-06};
            point_filters_b[392] = point_filter392;
            
            static const uint32_t point_weights393[]={1199169193,2135195248,3068600806,
            };
            static quant_filter_t point_filter393 = {96, 1, point_weights393, -2.7944056455453392e-06};
            point_filters_b[393] = point_filter393;
            
            static const uint32_t point_weights394[]={2727988406,4181268606,4097129133,
            };
            static quant_filter_t point_filter394 = {96, 1, point_weights394, -1.2558148227981292e-06};
            point_filters_b[394] = point_filter394;
            
            static const uint32_t point_weights395[]={4022728925,1712329239,1290311034,
            };
            static quant_filter_t point_filter395 = {96, 1, point_weights395, 1.8102358012583863e-07};
            point_filters_b[395] = point_filter395;
            
            static const uint32_t point_weights396[]={3616455703,481032404,3417703874,
            };
            static quant_filter_t point_filter396 = {96, 1, point_weights396, -1.1948596920774435e-06};
            point_filters_b[396] = point_filter396;
            
            static const uint32_t point_weights397[]={248643832,2739379829,357124264,
            };
            static quant_filter_t point_filter397 = {96, 1, point_weights397, -7.58808346290607e-06};
            point_filters_b[397] = point_filter397;
            
            static const uint32_t point_weights398[]={1139617607,2644676334,1468970522,
            };
            static quant_filter_t point_filter398 = {96, 1, point_weights398, 9.033177775563672e-06};
            point_filters_b[398] = point_filter398;
            
            static const uint32_t point_weights399[]={1153200372,227885540,3087029143,
            };
            static quant_filter_t point_filter399 = {96, 1, point_weights399, -1.4766073945793323e-05};
            point_filters_b[399] = point_filter399;
            
            static const uint32_t point_weights400[]={451831332,664360049,962623417,
            };
            static quant_filter_t point_filter400 = {96, 1, point_weights400, -3.1445244985661702e-06};
            point_filters_b[400] = point_filter400;
            
            static const uint32_t point_weights401[]={3182638484,3545032769,3436479569,
            };
            static quant_filter_t point_filter401 = {96, 1, point_weights401, -2.101989366565249e-06};
            point_filters_b[401] = point_filter401;
            
            static const uint32_t point_weights402[]={1001247894,4294466023,1858760610,
            };
            static quant_filter_t point_filter402 = {96, 1, point_weights402, -5.355375378712779e-06};
            point_filters_b[402] = point_filter402;
            
            static const uint32_t point_weights403[]={2565701216,4167812736,2790789174,
            };
            static quant_filter_t point_filter403 = {96, 1, point_weights403, -2.935276143034571e-06};
            point_filters_b[403] = point_filter403;
            
            static const uint32_t point_weights404[]={826353377,4124021069,2751613262,
            };
            static quant_filter_t point_filter404 = {96, 1, point_weights404, -4.7625334786971507e-07};
            point_filters_b[404] = point_filter404;
            
            static const uint32_t point_weights405[]={3371543284,1887992341,779875597,
            };
            static quant_filter_t point_filter405 = {96, 1, point_weights405, 3.921205916412873e-06};
            point_filters_b[405] = point_filter405;
            
            static const uint32_t point_weights406[]={2885173702,332473235,1355384609,
            };
            static quant_filter_t point_filter406 = {96, 1, point_weights406, -5.258429155219346e-06};
            point_filters_b[406] = point_filter406;
            
            static const uint32_t point_weights407[]={1049588919,139594596,1070065313,
            };
            static quant_filter_t point_filter407 = {96, 1, point_weights407, -1.2296162594793714e-06};
            point_filters_b[407] = point_filter407;
            
            static const uint32_t point_weights408[]={1935742235,2841887920,2399709510,
            };
            static quant_filter_t point_filter408 = {96, 1, point_weights408, -2.043347421931685e-06};
            point_filters_b[408] = point_filter408;
            
            static const uint32_t point_weights409[]={2994879919,1963712440,2836885577,
            };
            static quant_filter_t point_filter409 = {96, 1, point_weights409, 9.722154572955333e-06};
            point_filters_b[409] = point_filter409;
            
            static const uint32_t point_weights410[]={943111510,3250348437,3882180543,
            };
            static quant_filter_t point_filter410 = {96, 1, point_weights410, -3.952179667976452e-06};
            point_filters_b[410] = point_filter410;
            
            static const uint32_t point_weights411[]={4196018084,1295314733,1243436296,
            };
            static quant_filter_t point_filter411 = {96, 1, point_weights411, -1.5621872080373578e-06};
            point_filters_b[411] = point_filter411;
            
            static const uint32_t point_weights412[]={872505318,3038278948,3411898151,
            };
            static quant_filter_t point_filter412 = {96, 1, point_weights412, -9.57741849560989e-06};
            point_filters_b[412] = point_filter412;
            
            static const uint32_t point_weights413[]={678980470,2585902302,3781956966,
            };
            static quant_filter_t point_filter413 = {96, 1, point_weights413, 4.2379733713460155e-06};
            point_filters_b[413] = point_filter413;
            
            static const uint32_t point_weights414[]={2121031805,3770016209,2501971726,
            };
            static quant_filter_t point_filter414 = {96, 1, point_weights414, -5.623973265755922e-06};
            point_filters_b[414] = point_filter414;
            
            static const uint32_t point_weights415[]={2176442530,1450293958,1584095097,
            };
            static quant_filter_t point_filter415 = {96, 1, point_weights415, 1.7956540432351176e-06};
            point_filters_b[415] = point_filter415;
            
            static const uint32_t point_weights416[]={2684102743,2165871790,3267539578,
            };
            static quant_filter_t point_filter416 = {96, 1, point_weights416, 4.29405054092058e-06};
            point_filters_b[416] = point_filter416;
            
            static const uint32_t point_weights417[]={3156483772,2927460628,1832981572,
            };
            static quant_filter_t point_filter417 = {96, 1, point_weights417, 6.403499810403446e-06};
            point_filters_b[417] = point_filter417;
            
            static const uint32_t point_weights418[]={697155196,2774084279,140675943,
            };
            static quant_filter_t point_filter418 = {96, 1, point_weights418, 4.227119916322408e-06};
            point_filters_b[418] = point_filter418;
            
            static const uint32_t point_weights419[]={915016426,2144261581,3439144532,
            };
            static quant_filter_t point_filter419 = {96, 1, point_weights419, -6.560144356626552e-06};
            point_filters_b[419] = point_filter419;
            
            static const uint32_t point_weights420[]={2612028245,3418868208,2696889901,
            };
            static quant_filter_t point_filter420 = {96, 1, point_weights420, 2.3415966552420286e-06};
            point_filters_b[420] = point_filter420;
            
            static const uint32_t point_weights421[]={1733728612,220615114,695803568,
            };
            static quant_filter_t point_filter421 = {96, 1, point_weights421, -1.6516247342224233e-05};
            point_filters_b[421] = point_filter421;
            
            static const uint32_t point_weights422[]={2484157037,3391251121,1293164125,
            };
            static quant_filter_t point_filter422 = {96, 1, point_weights422, 2.78280731436098e-06};
            point_filters_b[422] = point_filter422;
            
            static const uint32_t point_weights423[]={2933315492,1464463613,2442342569,
            };
            static quant_filter_t point_filter423 = {96, 1, point_weights423, 4.3638601709972136e-06};
            point_filters_b[423] = point_filter423;
            
            static const uint32_t point_weights424[]={211296624,4062703524,814070262,
            };
            static quant_filter_t point_filter424 = {96, 1, point_weights424, 1.5139893321247655e-06};
            point_filters_b[424] = point_filter424;
            
            static const uint32_t point_weights425[]={3813042707,4206695028,2494531670,
            };
            static quant_filter_t point_filter425 = {96, 1, point_weights425, -1.1764245755330194e-06};
            point_filters_b[425] = point_filter425;
            
            static const uint32_t point_weights426[]={2924842362,387987591,1068186618,
            };
            static quant_filter_t point_filter426 = {96, 1, point_weights426, -4.751568667415995e-06};
            point_filters_b[426] = point_filter426;
            
            static const uint32_t point_weights427[]={1777554550,1240771924,2100242021,
            };
            static quant_filter_t point_filter427 = {96, 1, point_weights427, 1.5507006537518464e-05};
            point_filters_b[427] = point_filter427;
            
            static const uint32_t point_weights428[]={1737012605,1977157345,2622525407,
            };
            static quant_filter_t point_filter428 = {96, 1, point_weights428, 9.174709703074768e-06};
            point_filters_b[428] = point_filter428;
            
            static const uint32_t point_weights429[]={2773970647,1691533734,3026133934,
            };
            static quant_filter_t point_filter429 = {96, 1, point_weights429, -1.744074893395009e-06};
            point_filters_b[429] = point_filter429;
            
            static const uint32_t point_weights430[]={1779431460,1006811018,2131883952,
            };
            static quant_filter_t point_filter430 = {96, 1, point_weights430, 2.9953666853543837e-06};
            point_filters_b[430] = point_filter430;
            
            static const uint32_t point_weights431[]={461594625,2549953640,2399506638,
            };
            static quant_filter_t point_filter431 = {96, 1, point_weights431, 1.3510405096894829e-06};
            point_filters_b[431] = point_filter431;
            
            static const uint32_t point_weights432[]={1891900208,696496686,485003971,
            };
            static quant_filter_t point_filter432 = {96, 1, point_weights432, 4.438225460035028e-06};
            point_filters_b[432] = point_filter432;
            
            static const uint32_t point_weights433[]={653810779,2467145791,1301343164,
            };
            static quant_filter_t point_filter433 = {96, 1, point_weights433, 2.0952231238879904e-07};
            point_filters_b[433] = point_filter433;
            
            static const uint32_t point_weights434[]={2945656344,3918520867,383019599,
            };
            static quant_filter_t point_filter434 = {96, 1, point_weights434, 4.5200036424830614e-07};
            point_filters_b[434] = point_filter434;
            
            static const uint32_t point_weights435[]={3348239227,829132982,1737438253,
            };
            static quant_filter_t point_filter435 = {96, 1, point_weights435, 9.784334906726144e-06};
            point_filters_b[435] = point_filter435;
            
            static const uint32_t point_weights436[]={4274307354,3290637241,3481895216,
            };
            static quant_filter_t point_filter436 = {96, 1, point_weights436, 1.3232041737865075e-06};
            point_filters_b[436] = point_filter436;
            
            static const uint32_t point_weights437[]={3040540084,2013303330,1825390384,
            };
            static quant_filter_t point_filter437 = {96, 1, point_weights437, 7.533874395448947e-07};
            point_filters_b[437] = point_filter437;
            
            static const uint32_t point_weights438[]={4208635634,3894265128,267894706,
            };
            static quant_filter_t point_filter438 = {96, 1, point_weights438, -1.512593712504895e-06};
            point_filters_b[438] = point_filter438;
            
            static const uint32_t point_weights439[]={2746577606,3090149529,424334045,
            };
            static quant_filter_t point_filter439 = {96, 1, point_weights439, 1.0416991926831543e-06};
            point_filters_b[439] = point_filter439;
            
            static const uint32_t point_weights440[]={4117860476,1537539518,151537846,
            };
            static quant_filter_t point_filter440 = {96, 1, point_weights440, 5.32582771484158e-06};
            point_filters_b[440] = point_filter440;
            
            static const uint32_t point_weights441[]={3593478820,334770311,3252536138,
            };
            static quant_filter_t point_filter441 = {96, 1, point_weights441, 5.473847977555124e-06};
            point_filters_b[441] = point_filter441;
            
            static const uint32_t point_weights442[]={3352651128,799169904,4074778964,
            };
            static quant_filter_t point_filter442 = {96, 1, point_weights442, 7.538047270827519e-07};
            point_filters_b[442] = point_filter442;
            
            static const uint32_t point_weights443[]={1940756154,247227611,3404839178,
            };
            static quant_filter_t point_filter443 = {96, 1, point_weights443, 4.868821633863263e-06};
            point_filters_b[443] = point_filter443;
            
            static const uint32_t point_weights444[]={370102978,2551131454,24538994,
            };
            static quant_filter_t point_filter444 = {96, 1, point_weights444, -5.92932474319241e-06};
            point_filters_b[444] = point_filter444;
            
            static const uint32_t point_weights445[]={1291316451,3368341759,3622371197,
            };
            static quant_filter_t point_filter445 = {96, 1, point_weights445, -1.2533892004285008e-06};
            point_filters_b[445] = point_filter445;
            
            static const uint32_t point_weights446[]={867699542,4182278475,3958413556,
            };
            static quant_filter_t point_filter446 = {96, 1, point_weights446, -9.189313459501136e-06};
            point_filters_b[446] = point_filter446;
            
            static const uint32_t point_weights447[]={1197739392,2367909846,2717762012,
            };
            static quant_filter_t point_filter447 = {96, 1, point_weights447, -2.719051963140373e-06};
            point_filters_b[447] = point_filter447;
            
            static const uint32_t point_weights448[]={305347028,1235549775,3208910675,
            };
            static quant_filter_t point_filter448 = {96, 1, point_weights448, -3.244874051233637e-06};
            point_filters_b[448] = point_filter448;
            
            static const uint32_t point_weights449[]={2127144362,4125413393,1780789792,
            };
            static quant_filter_t point_filter449 = {96, 1, point_weights449, 9.193224741466111e-07};
            point_filters_b[449] = point_filter449;
            
            static const uint32_t point_weights450[]={3758234068,1084365311,1940079640,
            };
            static quant_filter_t point_filter450 = {96, 1, point_weights450, -5.408785909821745e-06};
            point_filters_b[450] = point_filter450;
            
            static const uint32_t point_weights451[]={319472726,2435202995,535250971,
            };
            static quant_filter_t point_filter451 = {96, 1, point_weights451, -4.2841666072490625e-06};
            point_filters_b[451] = point_filter451;
            
            static const uint32_t point_weights452[]={89895612,450591111,2656567304,
            };
            static quant_filter_t point_filter452 = {96, 1, point_weights452, 5.970700385660166e-06};
            point_filters_b[452] = point_filter452;
            
            static const uint32_t point_weights453[]={2399601743,3099315225,3754371341,
            };
            static quant_filter_t point_filter453 = {96, 1, point_weights453, -2.1182415821385803e-06};
            point_filters_b[453] = point_filter453;
            
            static const uint32_t point_weights454[]={315366444,1285676017,3169333527,
            };
            static quant_filter_t point_filter454 = {96, 1, point_weights454, 7.711653779551852e-06};
            point_filters_b[454] = point_filter454;
            
            static const uint32_t point_weights455[]={2631219792,2083443081,3720894573,
            };
            static quant_filter_t point_filter455 = {96, 1, point_weights455, 2.196398554588086e-06};
            point_filters_b[455] = point_filter455;
            
            static const uint32_t point_weights456[]={2963251874,2974657474,251942758,
            };
            static quant_filter_t point_filter456 = {96, 1, point_weights456, 5.007585514249513e-06};
            point_filters_b[456] = point_filter456;
            
            static const uint32_t point_weights457[]={2057730003,1807215232,1126974690,
            };
            static quant_filter_t point_filter457 = {96, 1, point_weights457, -5.742656412621727e-06};
            point_filters_b[457] = point_filter457;
            
            static const uint32_t point_weights458[]={914627519,2370553294,4096922548,
            };
            static quant_filter_t point_filter458 = {96, 1, point_weights458, -1.4906621800037101e-05};
            point_filters_b[458] = point_filter458;
            
            static const uint32_t point_weights459[]={1715424650,566905774,850124163,
            };
            static quant_filter_t point_filter459 = {96, 1, point_weights459, 8.762423931329977e-06};
            point_filters_b[459] = point_filter459;
            
            static const uint32_t point_weights460[]={1687005044,3439007333,3618015420,
            };
            static quant_filter_t point_filter460 = {96, 1, point_weights460, 7.492644726880826e-06};
            point_filters_b[460] = point_filter460;
            
            static const uint32_t point_weights461[]={2150681746,3233619961,3854551233,
            };
            static quant_filter_t point_filter461 = {96, 1, point_weights461, 3.2858079066500068e-06};
            point_filters_b[461] = point_filter461;
            
            static const uint32_t point_weights462[]={3280755420,2088577924,1722460437,
            };
            static quant_filter_t point_filter462 = {96, 1, point_weights462, -1.526696178189013e-05};
            point_filters_b[462] = point_filter462;
            
            static const uint32_t point_weights463[]={2956520836,4216574850,3438232469,
            };
            static quant_filter_t point_filter463 = {96, 1, point_weights463, -8.922199413063936e-06};
            point_filters_b[463] = point_filter463;
            
            static const uint32_t point_weights464[]={1373155757,875256636,2242103079,
            };
            static quant_filter_t point_filter464 = {96, 1, point_weights464, -3.2736284083512146e-06};
            point_filters_b[464] = point_filter464;
            
            static const uint32_t point_weights465[]={3213047711,705520071,4123423050,
            };
            static quant_filter_t point_filter465 = {96, 1, point_weights465, 2.9515632604670827e-07};
            point_filters_b[465] = point_filter465;
            
            static const uint32_t point_weights466[]={1106779299,2276186456,2593087719,
            };
            static quant_filter_t point_filter466 = {96, 1, point_weights466, 2.2815254396846285e-06};
            point_filters_b[466] = point_filter466;
            
            static const uint32_t point_weights467[]={1766241516,2318149649,3742288692,
            };
            static quant_filter_t point_filter467 = {96, 1, point_weights467, -4.361918399808928e-06};
            point_filters_b[467] = point_filter467;
            
            static const uint32_t point_weights468[]={530241301,1532742593,1594062515,
            };
            static quant_filter_t point_filter468 = {96, 1, point_weights468, -2.2231529328564648e-06};
            point_filters_b[468] = point_filter468;
            
            static const uint32_t point_weights469[]={1247082767,3740830804,2562335928,
            };
            static quant_filter_t point_filter469 = {96, 1, point_weights469, 4.768759481521556e-06};
            point_filters_b[469] = point_filter469;
            
            static const uint32_t point_weights470[]={3651908226,4198376915,1349832851,
            };
            static quant_filter_t point_filter470 = {96, 1, point_weights470, -5.539027824852383e-06};
            point_filters_b[470] = point_filter470;
            
            static const uint32_t point_weights471[]={541263189,1262000518,2759695877,
            };
            static quant_filter_t point_filter471 = {96, 1, point_weights471, -5.998641881888034e-06};
            point_filters_b[471] = point_filter471;
            
            static const uint32_t point_weights472[]={1146613432,340320882,1326091362,
            };
            static quant_filter_t point_filter472 = {96, 1, point_weights472, 6.82799100104603e-06};
            point_filters_b[472] = point_filter472;
            
            static const uint32_t point_weights473[]={1496351976,924778671,55362024,
            };
            static quant_filter_t point_filter473 = {96, 1, point_weights473, 3.1456825126952026e-06};
            point_filters_b[473] = point_filter473;
            
            static const uint32_t point_weights474[]={1934013487,685312027,3560208662,
            };
            static quant_filter_t point_filter474 = {96, 1, point_weights474, 7.73775354900863e-07};
            point_filters_b[474] = point_filter474;
            
            static const uint32_t point_weights475[]={4249413369,955736081,3493776372,
            };
            static quant_filter_t point_filter475 = {96, 1, point_weights475, 7.163038844737457e-06};
            point_filters_b[475] = point_filter475;
            
            static const uint32_t point_weights476[]={2297109572,1005730483,625422431,
            };
            static quant_filter_t point_filter476 = {96, 1, point_weights476, 6.1928490140417125e-06};
            point_filters_b[476] = point_filter476;
            
            static const uint32_t point_weights477[]={3407442251,3569941167,3612045271,
            };
            static quant_filter_t point_filter477 = {96, 1, point_weights477, 1.096414621315489e-06};
            point_filters_b[477] = point_filter477;
            
            static const uint32_t point_weights478[]={3613003786,739454446,1244237337,
            };
            static quant_filter_t point_filter478 = {96, 1, point_weights478, -1.5604281315972912e-06};
            point_filters_b[478] = point_filter478;
            
            static const uint32_t point_weights479[]={3730114329,748308376,3415882772,
            };
            static quant_filter_t point_filter479 = {96, 1, point_weights479, -1.9905855879187584e-06};
            point_filters_b[479] = point_filter479;
            
            static const uint32_t point_weights480[]={2105348696,1831201504,1076255348,
            };
            static quant_filter_t point_filter480 = {96, 1, point_weights480, 9.94610400084639e-06};
            point_filters_b[480] = point_filter480;
            
            static const uint32_t point_weights481[]={51022458,1877038084,3457987782,
            };
            static quant_filter_t point_filter481 = {96, 1, point_weights481, -3.7083177630847786e-06};
            point_filters_b[481] = point_filter481;
            
            static const uint32_t point_weights482[]={3565524378,1400820754,4264045421,
            };
            static quant_filter_t point_filter482 = {96, 1, point_weights482, 4.007049938081764e-06};
            point_filters_b[482] = point_filter482;
            
            static const uint32_t point_weights483[]={1840051629,1400371927,698844873,
            };
            static quant_filter_t point_filter483 = {96, 1, point_weights483, 4.978355718776584e-06};
            point_filters_b[483] = point_filter483;
            
            static const uint32_t point_weights484[]={1650486330,543857476,1302747466,
            };
            static quant_filter_t point_filter484 = {96, 1, point_weights484, 8.13842416391708e-06};
            point_filters_b[484] = point_filter484;
            
            static const uint32_t point_weights485[]={3750835322,2321951836,2525068768,
            };
            static quant_filter_t point_filter485 = {96, 1, point_weights485, -2.1557680156547576e-06};
            point_filters_b[485] = point_filter485;
            
            static const uint32_t point_weights486[]={1004271027,4020679638,2412643512,
            };
            static quant_filter_t point_filter486 = {96, 1, point_weights486, -4.174386560862331e-07};
            point_filters_b[486] = point_filter486;
            
            static const uint32_t point_weights487[]={3209087822,3786239589,316073061,
            };
            static quant_filter_t point_filter487 = {96, 1, point_weights487, 2.0113905918606179e-07};
            point_filters_b[487] = point_filter487;
            
            static const uint32_t point_weights488[]={3735828721,3885208548,433793907,
            };
            static quant_filter_t point_filter488 = {96, 1, point_weights488, -4.381993790048e-07};
            point_filters_b[488] = point_filter488;
            
            static const uint32_t point_weights489[]={2879324775,3672679961,545847021,
            };
            static quant_filter_t point_filter489 = {96, 1, point_weights489, 2.322026830370305e-06};
            point_filters_b[489] = point_filter489;
            
            static const uint32_t point_weights490[]={1831121743,2440268028,2762569040,
            };
            static quant_filter_t point_filter490 = {96, 1, point_weights490, 4.649974925996503e-06};
            point_filters_b[490] = point_filter490;
            
            static const uint32_t point_weights491[]={1485474565,166989426,970210261,
            };
            static quant_filter_t point_filter491 = {96, 1, point_weights491, 2.1104290226503508e-07};
            point_filters_b[491] = point_filter491;
            
            static const uint32_t point_weights492[]={183748698,1736270109,387609377,
            };
            static quant_filter_t point_filter492 = {96, 1, point_weights492, -4.194244866084773e-06};
            point_filters_b[492] = point_filter492;
            
            static const uint32_t point_weights493[]={1532894355,1323549853,4104235959,
            };
            static quant_filter_t point_filter493 = {96, 1, point_weights493, 1.899244139735856e-08};
            point_filters_b[493] = point_filter493;
            
            static const uint32_t point_weights494[]={3614590594,1445546156,2350926746,
            };
            static quant_filter_t point_filter494 = {96, 1, point_weights494, -3.0302890081657097e-06};
            point_filters_b[494] = point_filter494;
            
            static const uint32_t point_weights495[]={4169974367,40861464,2465071474,
            };
            static quant_filter_t point_filter495 = {96, 1, point_weights495, 1.3941059933131328e-06};
            point_filters_b[495] = point_filter495;
            
            static const uint32_t point_weights496[]={645027209,2109262521,2832652836,
            };
            static quant_filter_t point_filter496 = {96, 1, point_weights496, -8.775282935857831e-07};
            point_filters_b[496] = point_filter496;
            
            static const uint32_t point_weights497[]={3928441803,853075347,2981077926,
            };
            static quant_filter_t point_filter497 = {96, 1, point_weights497, 1.45818476084969e-05};
            point_filters_b[497] = point_filter497;
            
            static const uint32_t point_weights498[]={3826465928,446613189,1200498377,
            };
            static quant_filter_t point_filter498 = {96, 1, point_weights498, 1.5014805967439315e-06};
            point_filters_b[498] = point_filter498;
            
            static const uint32_t point_weights499[]={1891296281,354787605,1982600841,
            };
            static quant_filter_t point_filter499 = {96, 1, point_weights499, -2.6983057068719063e-06};
            point_filters_b[499] = point_filter499;
            
            static const uint32_t point_weights500[]={1173220786,3499597508,2200338831,
            };
            static quant_filter_t point_filter500 = {96, 1, point_weights500, 2.5035146791196894e-06};
            point_filters_b[500] = point_filter500;
            
            static const uint32_t point_weights501[]={1665598975,1459667027,326532377,
            };
            static quant_filter_t point_filter501 = {96, 1, point_weights501, 3.2228319923888193e-06};
            point_filters_b[501] = point_filter501;
            
            static const uint32_t point_weights502[]={4169317039,1613421161,1127219609,
            };
            static quant_filter_t point_filter502 = {96, 1, point_weights502, -2.529709263399127e-06};
            point_filters_b[502] = point_filter502;
            
            static const uint32_t point_weights503[]={4282690680,2669712040,1785191075,
            };
            static quant_filter_t point_filter503 = {96, 1, point_weights503, -4.0802638068271335e-06};
            point_filters_b[503] = point_filter503;
            
            static const uint32_t point_weights504[]={3344026781,1568608676,4180867020,
            };
            static quant_filter_t point_filter504 = {96, 1, point_weights504, -4.697926215158077e-06};
            point_filters_b[504] = point_filter504;
            
            static const uint32_t point_weights505[]={3687609247,3511810114,911614523,
            };
            static quant_filter_t point_filter505 = {96, 1, point_weights505, -3.2424400160380173e-06};
            point_filters_b[505] = point_filter505;
            
            static const uint32_t point_weights506[]={3364708885,2079300536,2653743837,
            };
            static quant_filter_t point_filter506 = {96, 1, point_weights506, 2.1467469650815474e-06};
            point_filters_b[506] = point_filter506;
            
            static const uint32_t point_weights507[]={2416756685,279932787,2603940124,
            };
            static quant_filter_t point_filter507 = {96, 1, point_weights507, -4.5808405957359355e-06};
            point_filters_b[507] = point_filter507;
            
            static const uint32_t point_weights508[]={3613914097,2790588313,1174032348,
            };
            static quant_filter_t point_filter508 = {96, 1, point_weights508, -1.2564943972392939e-05};
            point_filters_b[508] = point_filter508;
            
            static const uint32_t point_weights509[]={3595708631,2460966846,3657095527,
            };
            static quant_filter_t point_filter509 = {96, 1, point_weights509, -1.2067491752532078e-06};
            point_filters_b[509] = point_filter509;
            
            static const uint32_t point_weights510[]={134812251,3671429072,766496913,
            };
            static quant_filter_t point_filter510 = {96, 1, point_weights510, 7.385515345958993e-06};
            point_filters_b[510] = point_filter510;
            
            static const uint32_t point_weights511[]={2797947935,2285552603,3634819044,
            };
            static quant_filter_t point_filter511 = {96, 1, point_weights511, -1.8454511518939398e-06};
            point_filters_b[511] = point_filter511;
            
            quant_separable_conv2d_layer_t layer = {512, depth_filter_b, point_filters_b};
            return layer;
            }
            
batch_normalization_layer_t init_batch_normalization_397_data(void){

    static const fixed inv_gamma_dev[] ={
    7258, 9041, 5151, 10553, 10227, 6493, 5291, 10457, 5138, 9207, 5853, 9963, 7841, 
    7626, 7295, 10514, 9894, 8805, 8125, 8867, 5685, 7289, 10168, 6293, 6270, 6811, 
    3914, 5031, 7355, 8818, 8555, 10038, 10944, 6831, 6407, 5521, 8220, 7399, 10774, 
    9912, 6245, 9545, 5868, 9405, 7664, 7561, 8691, 10566, 6306, 8728, 7867, 9690, 5088, 
    5696, 7825, 7221, 9112, 8850, 5102, 7951, 10689, 10168, 9180, 5927, 8076, 7724, 
    7794, 8456, 5985, 9069, 4573, 6216, 6820, 7982, 6112, 10558, 10041, 6914, 9025, 
    6299, 8946, 8025, 4613, 7224, 8963, 9174, 7078, 5405, 5939, 5333, 7468, 8441, 6854, 
    7488, 5005, 10522, 8502, 5089, 7853, 4645, 7716, 8840, 8102, 9363, 8830, 7252, 9827, 
    7613, 5591, 9528, 7419, 6414, 6746, 8636, 7309, 6279, 7572, 9229, 7377, 9653, 10046, 
    9996, 9052, 6697, 6672, 9064, 6900, 6972, 5826, 8775, 4102, 6918, 7650, 8901, 9198, 
    7205, 9160, 6925, 8216, 7060, 6364, 8899, 11499, 6767, 9698, 6485, 4619, 7814, 7764, 
    7948, 7931, 8740, 10451, 8126, 7755, 6841, 9080, 6296, 6234, 7748, 6805, 7757, 7675, 
    10546, 6437, 8467, 8798, 5841, 7802, 3601, 9247, 9371, 9507, 3977, 7253, 6027, 7589, 
    7848, 7134, 6483, 8152, 9136, 7183, 8298, 7518, 5238, 5630, 10475, 7641, 6187, 9717, 
    7505, 7792, 6388, 6258, 6758, 6036, 8642, 8826, 9846, 10604, 9517, 6556, 6832, 7808, 
    6109, 7995, 7943, 8907, 10050, 9977, 8701, 7511, 10212, 7199, 8287, 10552, 9935, 
    10429, 11614, 8451, 4864, 7279, 7671, 6570, 8461, 4768, 8818, 9176, 4394, 6697, 
    6062, 6434, 9956, 8910, 7758, 6502, 5340, 10960, 10364, 5910, 4437, 5062, 9288, 
    7610, 8532, 7929, 6403, 9314, 6888, 7619, 8831, 9560, 8683, 7757, 6078, 9841, 5721, 
    8071, 7101, 8343, 9509, 6764, 7151, 9273, 7896, 11802, 9486, 8292, 7797, 8464, 6599, 
    7382, 7268, 8168, 8409, 7635, 7267, 8024, 8136, 9477, 8102, 8970, 8805, 7602, 6019, 
    8815, 8817, 10186, 11334, 6411, 8690, 5249, 6160, 6911, 8266, 6200, 8614, 5446, 
    6788, 12014, 9033, 6935, 5341, 8985, 8623, 7502, 4974, 6976, 8589, 7856, 8634, 6410, 
    5124, 10167, 7140, 7893, 6513, 12094, 6346, 7077, 8729, 11680, 8310, 8384, 5711, 
    10469, 8091, 9261, 8574, 10339, 7562, 8219, 6983, 7377, 7474, 8902, 7277, 11176, 
    9968, 9808, 7375, 7357, 7542, 5077, 7658, 8525, 6384, 7127, 8784, 8354, 7430, 5927, 
    9143, 9223, 6641, 9558, 5815, 7636, 7054, 7254, 5611, 7156, 6145, 7803, 11316, 7506, 
    7450, 7744, 6126, 8703, 7545, 7382, 7172, 9280, 8411, 5928, 5811, 6639, 7605, 9815, 
    7842, 7699, 6375, 8735, 7519, 3952, 8066, 8191, 9888, 4634, 7242, 6080, 7327, 13519, 
    8125, 8577, 7883, 7448, 7492, 9747, 7016, 11803, 9654, 8591, 8953, 7251, 6048, 8033, 
    6108, 8537, 9881, 5683, 9037, 9132, 8238, 7542, 8343, 7076, 10191, 8089, 8778, 8364, 
    8152, 7624, 9025, 6727, 6663, 7541, 10458, 10627, 8400, 7918, 8902, 8716, 9438, 
    7814, 5165, 9262, 7945, 5035, 9280, 9478, 6640, 8347, 7957, 9713, 9989, 6068, 7304, 
    9267, 10454, 8724, 7931, 8137, 10034, 7470, 6015, 9223, 9950, 9179, 6732, 9245, 
    8165, 7345, 9146, 7918, 7015, 7504, 9660, 9342, 7581, 8497, 7044, 8109, 10214, 9265, 
    6968, 9209, 8214, 9098, 8724, 8696, 5529, 7012, 8417, 7602, 5703, 5046, 9481, 8096, 
    7472, 5172, 4776, 8573, 3836, 5538, 9971, 7617, 4092, 8887, 7325, 7579, 7085, 4561, 
    5821, 6849, 9270, 5945, 8928, 8142, 5982
    };
    static const fixed std_beta[] ={
    -54996, 737, -63368, -90474, 140040, 113644, 37552, -22455, -57228, -30335, -19355, 
    -10399, -72596, 12223, 97339, -41579, 47859, 19365, 31793, -118692, -27137, 42683, 
    -43777, -20803, 36594, -39214, -4750, -25333, 15635, -4103, -35506, -157815, 50388, 
    -83892, -27083, -40073, -2225, -127171, 82382, 125501, 11963, 31448, 27432, 117117, 
    -134103, -39488, -88031, -90830, -22293, 44703, -46343, -7537, -41845, 42858, 11091, 
    74980, -67975, -35363, 35817, -13323, -112714, 35296, -34369, 96062, -6346, 71444, 
    -471, 53848, -29282, -13010, 39284, 50057, 63262, 155859, 62098, 77446, 5919, -30517, 
    23488, 16120, -78571, 89839, -62311, 67173, 18613, 15523, -37645, -37246, -3148, 
    2714, -113425, 29651, -26458, -613, 25871, 124924, -11517, 30175, -77789, -5366, 
    -27402, 45730, 9271, 7854, -33300, 35418, 41644, 20811, -31336, 41752, -40006, 89062, 
    43348, -19943, 11783, -12478, 34381, 26711, 64047, 17106, -112282, 56434, -92928, 
    72722, -12391, 32765, 51315, 24841, -12379, 18486, -32257, -3212, 31558, -26120, 
    46937, 8808, -66255, -12162, 54850, 68642, 19500, 64050, 138040, -1645, 69210, 40301, 
    -38138, -10428, -108930, 147544, 3646, 58234, 48401, 50983, -12498, 3150, 141282, 
    -76489, -7822, -2297, -3213, 44225, -22623, -86800, -18081, -43641, -44469, -52038, 
    50939, -18240, 31168, 40307, -59131, 4834, 37787, 7825, 53838, -37275, 80725, -69588, 
    29030, -54014, -13093, 71167, -104207, -110313, -49366, 46359, 41462, 19321, -111870, 
    287, -1037, -50223, 42679, 53681, 4870, -70548, 46177, 147520, 91471, -20200, -93276, 
    -93423, -33814, 17537, 22048, -78445, 60200, -51117, -8084, 105422, -45638, 72519, 
    -3492, -23936, 44582, -104268, -125263, 24087, -17214, -40368, 3652, 18856, -22607, 
    30577, 34237, 24538, -7336, 1296, 85374, 32966, -50953, -12335, -36961, -55868, 
    -28710, 70425, -68135, -47873, 74139, -23940, -40101, 35561, -32841, 49482, -42117, 
    -28834, 13200, -38654, -11708, -58006, 25662, 50782, 1096, -53372, -44860, -11590, 
    14463, 51866, 59836, -105320, 28570, -55797, -23261, -42286, 71854, -76741, 68844, 
    -49339, -20614, -58536, -29712, 35090, 68171, 73328, 49677, 87066, -46267, 102552, 
    -18546, -7034, -19719, -18075, 38755, -13337, -38679, 4938, 22764, 41246, 41565, 
    63134, 63984, -28076, 57530, 47899, -30407, 51852, 27491, -3830, 51983, -53228, 
    -19336, 23887, -74350, -24736, 7405, 29673, -37693, -71175, -65162, -47549, 93277, 
    7351, 68364, 8173, -96277, 6763, 112869, 64909, 5781, -102070, 23877, -56444, -41150, 
    5245, -15819, 23274, -41157, 23664, -55218, 34255, -63737, 44340, 17765, 9742, 108619, 
    52779, -6812, 86334, -48649, -2285, 58337, -28517, 73142, 26089, -76588, -44848, 
    -33057, -38511, 18848, 60325, 9572, 31488, 75419, 29990, -12981, 77651, -67262, 
    32415, -37016, 12185, 85748, 3404, -35559, -143465, -6297, 34728, -1757, 59517, 
    -5618, -7116, 71444, 12736, 41543, -66877, 9913, -41121, 3971, 85955, 40228, -79222, 
    99370, -10089, -113655, 57733, 35511, -55861, -481, -71695, 9408, 78344, 41865, 
    31846, 67983, 10141, -142850, 40037, -72735, 4630, 42378, -48591, 40673, 18513, 
    155410, 60547, -58975, 39731, -73802, 4483, -35289, -89252, 37216, 26922, -91242, 
    -36649, -339, -45199, -62960, 27089, -62575, 65384, -36832, 2429, -38671, -12555, 
    -22503, -55454, -62828, -63694, 139417, 31919, -30149, -68146, -60254, -71582, -13215, 
    12110, 26623, -13326, 10476, -4472, -102356, 50862, 29332, 1793, -52236, 49801, 
    18159, -27712, -20837, 38911, 16669, 15625, -41966, 43140, -30691, -102426, 94726, 
    -4415, 8731, 11612, 39659, 45392, 70689, 89019, -9253, 72392, -17877, -52036, 17297, 
    -23848, -57607, 80319, 11012, -166865, -41221, -17275, 51593, 45073, -63465, -141867, 
    36960, 20168, 17434, 91541, -16674, -97729, -9414, 34312, 90329, -28901, -4106, 
    27887, 113009, -47017, -4869, -53937, 56191, -31976, 25124, 63723, 17079, 79189, 
    18816, 17489, -7911, -77436, 92770, -26747, 48566, -15141
    };

    static const batch_normalization_layer_t norm = { 512, inv_gamma_dev, std_beta  };
    return norm;
}

dense_layer_t init_dense_80_data(void){

    static neuron_t neurons[10];

    /* [-0.05379952  0.03910126  0.04419712  0.09058087 -0.19967821 -0.02909221
  0.01770304  0.01927762 -0.0363671  -0.02242385  0.02385657 -0.00381177
 -0.01010652  0.02189369 -0.02768233 -0.00372944 -0.10048559  0.06434051
  0.01766349 -0.093766    0.00394142  0.12111914 -0.09634526  0.03073879
  0.08253112  0.042069    0.01390119 -0.1097483   0.03593421  0.04233839
 -0.10916369  0.09334242  0.04952634  0.00816104  0.10018469  0.01655191
 -0.04173514 -0.0420214   0.06234884 -0.00833644 -0.1176995   0.00768373
 -0.1523948   0.0720688  -0.03582785  0.07871296  0.03922314 -0.12795243
  0.04914927 -0.09309644 -0.06199722 -0.06985642  0.1073534  -0.1173569
 -0.01775591 -0.02110743  0.00803457 -0.10023607  0.00633598 -0.1482363
  0.02681184  0.10473157 -0.0309733   0.07921828 -0.07110051  0.08952089
 -0.08332182 -0.06404912  0.03200455  0.03586958  0.11408093 -0.11949716
 -0.09010138  0.03675531  0.05221736 -0.03011508  0.02441947  0.04594137
  0.04227134 -0.03258181 -0.12136498 -0.03496027 -0.00033272  0.06435838
 -0.02052037 -0.03028258  0.01977018 -0.00165075 -0.09146971  0.14862394
 -0.11627818 -0.08524115 -0.0403504   0.02686009  0.06420226 -0.05612554
  0.18525659 -0.13279079 -0.05429398  0.11131136  0.0601439   0.17805125
 -0.05027286  0.02090507  0.06370078 -0.03609943 -0.04831625 -0.08118884
  0.00435811  0.07595788  0.03616029 -0.04479796  0.06345151  0.02151775
 -0.00527434 -0.10251864 -0.07079245  0.02435002  0.02865456 -0.00838255
 -0.03127742  0.03195632  0.06356824  0.05997603  0.02768509  0.02645809
 -0.04904288 -0.02977649  0.03623315 -0.05405838 -0.00406332 -0.01761443
  0.01568655 -0.041079   -0.11514264  0.00383633 -0.06416471 -0.10462558
 -0.04248442 -0.12162235 -0.04125299  0.09132919 -0.04663961 -0.06305972
 -0.08637218 -0.06316196  0.02935346  0.00391298  0.00912649 -0.01199752
 -0.15301056 -0.00956901  0.02237257 -0.08625045 -0.03097613  0.11759621
  0.06540973  0.03911997  0.070826   -0.02187214 -0.0573906   0.07883477
 -0.01219606 -0.05372657 -0.02283716 -0.07746185 -0.04704437 -0.02650666
 -0.01826176 -0.04343932 -0.03029933 -0.05578199 -0.01569783  0.0290307
  0.04647596  0.03642979 -0.07860738  0.04554967  0.07548016  0.08068298
  0.05590699  0.04029561  0.03486227 -0.0031704   0.01315434 -0.04208964
  0.07462448 -0.01838886 -0.02875723  0.15058303  0.10620544  0.02572956
  0.04362167  0.03198304  0.08353639  0.09254447 -0.06720535 -0.05435669
  0.06906075  0.07167427  0.02545432 -0.03126317 -0.05240177  0.03227841
 -0.03250176 -0.02531276 -0.08734315  0.05449589 -0.10850828 -0.05618077
  0.02761175  0.00548015  0.00708683 -0.06901712 -0.17803954  0.04361062
  0.06412102  0.03251778 -0.06986981  0.05129137 -0.05447984 -0.07233853
 -0.0048704   0.01241108  0.08318231 -0.02755532 -0.08415311 -0.2014479
  0.03629393 -0.05734291 -0.01678438  0.00146897 -0.11043485 -0.00317348
 -0.03580991 -0.04195785  0.12593083  0.04783874  0.00733944  0.10006612
  0.07292364 -0.03434076  0.03164314  0.05867536 -0.02853841 -0.01702818
  0.05373814 -0.07215641  0.13203833 -0.00903921 -0.0723474   0.03910098
  0.11434858  0.04632769  0.01425323  0.02298469 -0.05898334  0.05660953
  0.0564535   0.03817416  0.08635771  0.03878937 -0.04493501  0.03165312
 -0.1955903  -0.03731012  0.04108761  0.07742586 -0.04636161  0.0658984
  0.10606312  0.04138498 -0.00841423  0.02958982 -0.08502575  0.12076005
 -0.03448091  0.03265567 -0.03741529 -0.06145671  0.17149843  0.09665111
 -0.01905285  0.07042571 -0.06671892 -0.04331588 -0.09697817  0.02167942
 -0.14319782 -0.07672872 -0.01718153  0.04888324  0.08042725 -0.07118201
  0.02552579  0.16406804  0.07932639  0.17326175  0.02352827  0.05252859
  0.05668769  0.04033545 -0.07158573 -0.02521995 -0.11204578  0.06467081
 -0.03561446 -0.04243319 -0.11421578  0.13430169 -0.02651282  0.08486393
  0.04910344  0.03137279 -0.00848982  0.00413897  0.06493032  0.09354047
  0.1043146  -0.0738627   0.01666334 -0.09232482  0.05140849  0.14977632
 -0.05317186 -0.02669357 -0.04245139  0.02261668 -0.15196332 -0.07941148
 -0.12915829  0.11817729 -0.00126592  0.06116085  0.11861971 -0.06701332
 -0.00493709 -0.05882679 -0.04675747  0.05300066 -0.07495116  0.04400251
  0.0261787  -0.020444   -0.12991023 -0.02740909 -0.03288293  0.01043251
 -0.107903   -0.12153945  0.05315594  0.03545275 -0.03491889 -0.09781241
 -0.08163963 -0.01639499  0.0153739  -0.01396664 -0.14763817  0.08389276
  0.05042793 -0.06262559 -0.1257677   0.01988369 -0.04508855 -0.09972034
  0.01635023 -0.05087033  0.06295738 -0.01198631  0.03182583  0.1299854
  0.13889906  0.13770205 -0.09074984  0.04514128 -0.07034607  0.00336756
  0.04468878  0.01839276 -0.09614202 -0.10913891  0.11364654  0.09555522
  0.05459791 -0.02913176  0.02955195  0.04660117  0.04036995  0.10560792
  0.04696112  0.00728822 -0.0064816   0.00996698  0.08237059  0.01204146
  0.11669947  0.00137642 -0.08563107 -0.04971796 -0.05172937  0.03549069
  0.04174259 -0.09056982  0.04408882 -0.15147388  0.06606388 -0.03575198
 -0.07317774 -0.06508516 -0.03054153 -0.01883793 -0.05096353  0.06162586
  0.02198195 -0.17148755  0.00236111 -0.08035818  0.01549845  0.00069647
 -0.04838879  0.1755135  -0.14275031 -0.00385904 -0.15872432 -0.08668998
  0.02029344  0.02582093 -0.16122928  0.02078505 -0.04263682 -0.02708689
  0.02777861 -0.00685254  0.08349283  0.09002253 -0.07140487  0.03441761
 -0.02138837  0.10853574  0.01222243  0.02019874  0.06932499  0.00772134
  0.07653859  0.12651877 -0.1121596   0.04517811 -0.03009611 -0.02787588
 -0.08021753  0.05078245  0.01244874  0.0338876   0.04822955  0.09286035
 -0.01760669  0.05002398  0.01196384 -0.08495972  0.06405354  0.00257809
  0.1235594  -0.12866941  0.05386046  0.09067315 -0.01459739 -0.00265647
 -0.00486695 -0.04450153 -0.01749031 -0.11132762  0.06697381  0.03035737
  0.01154611  0.01024519 -0.01015881  0.01366085  0.08004616 -0.0027245
  0.05796185 -0.05241239  0.07866253 -0.10526323 -0.09865411 -0.05421949
  0.04717548  0.04802566 -0.04592042  0.00349647 -0.12045006  0.05407856
 -0.12164347 -0.03201712 -0.01593416  0.00548537 -0.02185167  0.09215567
  0.10461903 -0.02265627 -0.04827701  0.14765537  0.01660468  0.04085397
  0.00494437 -0.05140127 -0.00436106  0.06024585  0.03430555  0.14764343
 -0.03746707  0.01424323] -0.08304399996995926*/
    static const fixed weights0[] ={
    -7052, 5125, 5793, 11873, -26172, -3813, 2320, 2527, -4767, -2939, 3127, -500, -1325, 
    2870, -3628, -489, -13171, 8433, 2315, -12290, 517, 15875, -12628, 4029, 10818, 
    5514, 1822, -14385, 4710, 5549, -14308, 12235, 6492, 1070, 13131, 2169, -5470, -5508, 
    8172, -1093, -15427, 1007, -19975, 9446, -4696, 10317, 5141, -16771, 6442, -12202, 
    -8126, -9156, 14071, -15382, -2327, -2767, 1053, -13138, 830, -19430, 3514, 13727, 
    -4060, 10383, -9319, 11734, -10921, -8395, 4195, 4701, 14953, -15663, -11810, 4818, 
    6844, -3947, 3201, 6022, 5541, -4271, -15908, -4582, -44, 8436, -2690, -3969, 2591, 
    -216, -11989, 19480, -15241, -11173, -5289, 3521, 8415, -7356, 24282, -17405, -7116, 
    14590, 7883, 23338, -6589, 2740, 8349, -4732, -6333, -10642, 571, 9956, 4740, -5872, 
    8317, 2820, -691, -13437, -9279, 3192, 3756, -1099, -4100, 4189, 8332, 7861, 3629, 
    3468, -6428, -3903, 4749, -7086, -533, -2309, 2056, -5384, -15092, 503, -8410, -13713, 
    -5569, -15941, -5407, 11971, -6113, -8265, -11321, -8279, 3847, 513, 1196, -1573, 
    -20055, -1254, 2932, -11305, -4060, 15414, 8573, 5128, 9283, -2867, -7522, 10333, 
    -1599, -7042, -2993, -10153, -6166, -3474, -2394, -5694, -3971, -7311, -2058, 3805, 
    6092, 4775, -10303, 5970, 9893, 10575, 7328, 5282, 4569, -416, 1724, -5517, 9781, 
    -2410, -3769, 19737, 13921, 3372, 5718, 4192, 10949, 12130, -8809, -7125, 9052, 
    9394, 3336, -4098, -6868, 4231, -4260, -3318, -11448, 7143, -14222, -7364, 3619, 
    718, 929, -9046, -23336, 5716, 8404, 4262, -9158, 6723, -7141, -9482, -638, 1627, 
    10903, -3612, -11030, -26404, 4757, -7516, -2200, 193, -14475, -416, -4694, -5499, 
    16506, 6270, 962, 13116, 9558, -4501, 4148, 7691, -3741, -2232, 7044, -9458, 17307, 
    -1185, -9483, 5125, 14988, 6072, 1868, 3013, -7731, 7420, 7399, 5004, 11319, 5084, 
    -5890, 4149, -25636, -4890, 5385, 10148, -6077, 8637, 13902, 5424, -1103, 3878, 
    -11144, 15828, -4519, 4280, -4904, -8055, 22479, 12668, -2497, 9231, -8745, -5677, 
    -12711, 2842, -18769, -10057, -2252, 6407, 10542, -9330, 3346, 21505, 10397, 22710, 
    3084, 6885, 7430, 5287, -9383, -3306, -14686, 8477, -4668, -5562, -14970, 17603, 
    -3475, 11123, 6436, 4112, -1113, 543, 8511, 12261, 13673, -9681, 2184, -12101, 6738, 
    19631, -6969, -3499, -5564, 2964, -19918, -10409, -16929, 15490, -166, 8016, 15548, 
    -8784, -647, -7711, -6129, 6947, -9824, 5767, 3431, -2680, -17028, -3593, -4310, 
    1367, -14143, -15930, 6967, 4647, -4577, -12820, -10701, -2149, 2015, -1831, -19351, 
    10996, 6610, -8208, -16485, 2606, -5910, -13071, 2143, -6668, 8252, -1571, 4171, 
    17037, 18206, 18049, -11895, 5917, -9220, 441, 5857, 2411, -12602, -14305, 14896, 
    12525, 7156, -3818, 3873, 6108, 5291, 13842, 6155, 955, -850, 1306, 10796, 1578, 
    15296, 180, -11224, -6517, -6780, 4652, 5471, -11871, 5779, -19854, 8659, -4686, 
    -9592, -8531, -4003, -2469, -6680, 8077, 2881, -22477, 309, -10533, 2031, 91, -6342, 
    23005, -18711, -506, -20804, -11363, 2660, 3384, -21133, 2724, -5588, -3550, 3641, 
    -898, 10944, 11799, -9359, 4511, -2803, 14226, 1602, 2647, 9087, 1012, 10032, 16583, 
    -14701, 5922, -3945, -3654, -10514, 6656, 1632, 4442, 6322, 12171, -2308, 6557, 
    1568, -11136, 8396, 338, 16195, -16865, 7060, 11885, -1913, -348, -638, -5833, -2292, 
    -14592, 8778, 3979, 1513, 1343, -1332, 1791, 10492, -357, 7597, -6870, 10310, -13797, 
    -12931, -7107, 6183, 6295, -6019, 458, -15788, 7088, -15944, -4197, -2089, 719, 
    -2864, 12079, 13713, -2970, -6328, 19353, 2176, 5355, 648, -6737, -572, 7897, 4496, 
    19352, -4911, 1867
    };
    
    static const neuron_t neuron0 = {weights0, -10885  };
    neurons[0]=neuron0;

    /* [-9.22798961e-02  1.54562863e-02  8.55072364e-02 -5.02358377e-02
  6.37161806e-02  3.66003625e-02  4.17640805e-02  1.12527674e-02
  1.59589484e-01  6.46049008e-02  2.66921800e-02  2.20413938e-01
 -2.39610691e-02  1.16682544e-01  6.69243140e-03  6.45849109e-02
 -8.82859156e-03  1.07473493e-01 -8.75047669e-02  7.22464845e-02
 -3.50484587e-02  7.83877075e-02  1.28535498e-02 -3.23220491e-02
  1.53571563e-02  5.75598329e-02  1.94372013e-02  3.24522518e-02
  2.22240798e-02  1.61698628e-02 -1.56258401e-02 -2.56566666e-02
  1.51490024e-03  1.27877658e-02  4.71646152e-02  5.18205985e-02
  1.37521923e-01 -1.63917914e-02 -1.18165566e-02  2.96139214e-02
 -2.95031979e-03 -1.80555642e-01  2.65901294e-02 -9.98935103e-02
 -9.25039724e-02  3.02933734e-02  7.83688873e-02 -2.15230770e-02
 -1.12902350e-03  4.95442040e-02  1.99369378e-02 -9.91052017e-03
 -9.70317647e-02 -2.87083592e-02 -4.21135537e-02 -1.38069898e-01
  8.23623538e-02 -5.68641536e-03  2.92195715e-02  1.75746139e-02
 -1.14216772e-03  5.59239760e-02  2.20380798e-02  1.81870386e-01
 -3.94620672e-02  3.68200848e-03 -4.65949699e-02  3.66250388e-02
 -1.20762419e-02 -1.09511435e-01 -2.12274562e-03 -9.00336429e-02
  1.33283651e-02  8.34826007e-02  5.78567833e-02  6.38848469e-02
  3.89317721e-02 -4.04547825e-02 -7.37447441e-02 -1.99618563e-02
  1.42171513e-02 -1.59838609e-02  2.47288477e-02 -9.04133841e-02
 -4.34733257e-02  1.06419198e-01 -2.41810475e-02 -4.63120081e-02
 -3.02219484e-02  9.34520885e-02 -3.29851061e-02  1.71545520e-02
  1.61193348e-02  4.27188277e-02  8.57639909e-02 -1.06719777e-01
 -1.74857657e-02  4.78004031e-02  9.43838283e-02  1.84003804e-02
  7.05886781e-02  5.00966087e-02  4.58267368e-02  6.98864534e-02
  4.08051424e-02 -4.20835093e-02  1.44915044e-01  5.47334105e-02
 -1.41797410e-02  5.30583002e-02  9.89520252e-02  7.78476149e-02
  5.56168631e-02 -7.00983927e-02 -6.89612236e-03  3.44726257e-02
  7.83003401e-03 -7.37732947e-02  5.38958907e-02  5.03572226e-02
 -9.21047702e-02 -2.65398435e-03 -5.91573864e-02 -1.01407915e-01
 -7.68374512e-03 -2.55348459e-02  7.89224505e-02 -9.16191489e-02
  1.06619336e-01 -8.53513256e-02  6.35090470e-02 -5.99390753e-02
 -1.36109099e-01 -5.26688136e-02  9.03645307e-02  8.31965506e-02
  2.70937439e-02 -6.04928657e-02 -9.86490704e-05  8.18930715e-02
  9.27476883e-02  4.54360805e-02  1.62135586e-01 -5.09473905e-02
 -1.64645612e-01  3.04642655e-02 -3.83643508e-02 -9.86910462e-02
  4.10137884e-02  8.86697788e-03 -1.55964270e-01 -4.59129363e-02
  7.56763369e-02  2.89711766e-02 -4.61037382e-02  1.91681609e-02
 -1.51475191e-01 -5.61009049e-02  5.54333329e-02  4.69384007e-02
 -7.23741278e-02 -8.98620859e-02  2.28286348e-03 -1.19231073e-02
 -1.63262407e-03  1.21931463e-01 -5.33757471e-02 -2.65948358e-03
 -9.86808315e-02  3.18681635e-02  7.98323452e-02 -7.21627176e-02
  3.01616527e-02 -1.70162365e-01  9.22005251e-02  9.90925133e-02
 -4.91621904e-02  9.80628058e-02 -5.46083674e-02  1.46529809e-01
 -8.74435082e-02 -7.08278269e-02  7.89587665e-03  7.68925697e-02
  1.46863595e-01 -1.92232651e-03 -1.18129112e-01  2.86807269e-02
  8.72486755e-02 -1.89378113e-02  1.06221564e-01 -1.18328519e-01
 -1.91186052e-02 -5.39560290e-03  3.09081469e-02  2.66355593e-02
  2.21669208e-02  4.87282686e-02 -2.52729421e-03  1.25228211e-01
  1.09233566e-01  2.49658469e-02 -8.02949257e-03  5.22187278e-02
  5.12825176e-02  1.19399190e-01 -6.41567633e-02  5.57878390e-02
  1.02423415e-01  1.53528288e-01 -7.78410658e-02  2.68146068e-01
  2.30378732e-01 -7.47186095e-02  9.93106067e-02 -2.65301391e-02
  1.15742378e-01 -1.40399113e-01  2.29003727e-02  1.95258800e-02
  2.13636249e-01 -9.86548066e-02  2.39827447e-02 -4.06479165e-02
 -1.38781928e-02  5.82537837e-02 -1.38273641e-01  7.62172565e-02
 -4.62851673e-02 -3.33164036e-02  3.27026881e-02  9.08098742e-02
 -1.44723486e-02 -3.47582921e-02  1.12733349e-01  1.45314902e-01
  7.59738088e-02 -2.16395482e-02  1.95336640e-01  1.08510880e-02
  1.40905194e-02  4.46600579e-02 -7.87340328e-02 -2.97446102e-02
  2.79231220e-02  5.15667759e-02 -1.04144700e-02 -1.19726121e-01
  1.27733454e-01  5.91398515e-02 -1.52354389e-01 -8.90434384e-02
 -1.71015263e-02 -3.48638073e-02 -7.52068311e-02 -1.96883045e-02
  8.94332156e-02  5.54534234e-02  1.43667534e-01  1.25918031e-01
 -1.12267740e-01 -1.81898519e-01  9.58122015e-02 -1.00148357e-01
  5.40265115e-04  1.71647474e-01  5.24719395e-02  1.05894625e-01
 -3.79146151e-02  7.89321661e-02 -2.07078800e-01  6.84981868e-02
 -1.42191157e-01  1.43344160e-02 -3.25709023e-02 -1.53984129e-03
 -4.32248451e-02 -1.57702029e-01 -1.38530686e-01  1.60514191e-02
 -4.42902632e-02  1.14592491e-03  3.26144733e-02  9.04159097e-04
  1.11991942e-01  7.84389153e-02  5.64161167e-02 -8.17079097e-02
  1.49639640e-02  4.23023589e-02 -7.44003877e-02 -1.69449657e-01
 -5.41748516e-02  2.00021878e-01  2.05952278e-03  8.52544680e-02
 -1.65391993e-02  7.68574849e-02 -2.48375107e-02  6.70854747e-02
  3.86847779e-02  7.25055262e-02 -8.35901313e-03 -8.55591074e-02
 -1.23002917e-01 -1.41835034e-01 -8.00764859e-02 -4.60416004e-02
  4.04467620e-02  1.14893109e-01 -1.13944195e-01 -1.36611670e-01
  2.36774664e-02 -7.66105875e-02 -6.63270615e-03  2.18089558e-02
  1.26894847e-01 -7.10720420e-02 -2.76014246e-02  5.66894538e-04
 -3.46411462e-03  1.56484265e-02 -1.40017331e-01  8.15865099e-02
  1.84613299e-02 -1.36408418e-01 -1.10407062e-01  7.59095028e-02
 -5.04526235e-02  9.91784781e-02 -1.13130167e-01  7.35440664e-03
 -1.20191406e-02 -4.27299850e-02 -1.13696061e-01 -8.43016133e-02
 -3.09857465e-02  9.48195085e-02 -1.44843617e-02 -9.06405225e-02
  3.69898975e-02 -2.53796820e-02 -9.20856092e-03 -6.62118196e-02
  1.27534634e-02  1.16712414e-01 -6.00068942e-02 -7.47835040e-02
 -1.42165884e-01  1.16071790e-01  1.32457707e-02  3.26995961e-02
 -2.74924450e-02  6.30242005e-02 -1.44042065e-02 -7.67936110e-02
  1.17062010e-01 -3.96118276e-02  7.57127032e-02  5.27583808e-02
  8.52176622e-02  3.19203958e-02  1.07246742e-01 -6.14362694e-02
  1.32420838e-01  5.49355075e-02 -6.05217703e-02  5.02764285e-02
 -4.11536451e-03  3.31461728e-02  7.60785416e-02  6.47497177e-02
  1.25777677e-01 -1.24532282e-02 -5.18808290e-02 -7.19957203e-02
 -1.27613813e-01  3.54449749e-02  7.93754160e-02 -2.84224618e-02
 -3.59051004e-02 -7.50596002e-02 -9.82276397e-05 -3.23743522e-02
 -4.43627946e-02  4.90667224e-02  6.82387203e-02  1.79466844e-01
  8.22702944e-02  2.20350437e-02 -9.39437374e-02 -1.91565882e-02
  1.27933815e-01 -1.39377341e-01  5.18355221e-02 -7.60328770e-02
  6.60301894e-02  1.20425291e-01  3.14102732e-02 -1.21256396e-01
 -5.43689951e-02  7.80621618e-02 -1.02630287e-01 -5.30908294e-02
  8.32450539e-02  8.42292085e-02 -2.91782990e-02 -5.43997623e-02
  4.00848612e-02  9.33149606e-02 -2.39944924e-02 -3.49127948e-02
  1.01735204e-01  3.05363536e-02  6.82531074e-02  1.08359121e-01
  1.84867345e-02 -8.10995474e-02  5.97999021e-02 -9.60007086e-02
 -3.57158594e-02 -2.63398960e-02 -4.39520180e-02  6.83018193e-02
  6.08241484e-02  1.15631305e-01  6.75433353e-02 -5.14713190e-02
 -1.70753181e-01 -1.95071902e-02 -5.02524003e-02 -4.18449566e-02
 -3.27915773e-02 -9.33675468e-02 -1.10890577e-02  1.27184972e-01
 -3.41605283e-02 -2.34784391e-02 -9.89084989e-02 -4.67369147e-02
 -6.78142458e-02  1.82636678e-01  1.58459961e-03 -1.01351656e-01
  6.30828962e-02  1.05791651e-01 -4.54718098e-02  2.11955637e-01
 -1.03217341e-01  6.11857846e-02  1.00861050e-01 -7.58521110e-02
  1.47037059e-01 -4.56862375e-02 -6.07899427e-02 -7.37507492e-02
  1.22680701e-01  7.31446221e-02 -2.74118539e-02 -7.33458847e-02
  4.98398580e-03  8.91346708e-02 -5.67556806e-02 -1.63871124e-02
  7.93641135e-02  2.32268348e-02 -1.32343406e-02  3.79657187e-02
 -9.03163180e-02 -5.13934419e-02 -1.10945664e-01  8.23353305e-02
 -4.34490517e-02  4.44336273e-02 -9.75308046e-02 -1.19963512e-01
  7.50024170e-02  1.28343076e-01 -7.49912709e-02  1.63693409e-02
  5.27306460e-02  9.49387327e-02 -1.92465931e-02 -7.30261877e-02
  1.65676072e-01 -3.43605019e-02 -3.92842218e-02  5.00902534e-03
 -2.59010196e-02  1.04081795e-01 -3.62844840e-02 -3.77600873e-03
 -9.39344466e-02 -1.69132039e-01 -1.68778747e-01 -9.73713920e-02
  6.33775676e-03  1.43895239e-01 -1.93327144e-02 -2.09217351e-02
  1.15852267e-01  1.60535462e-02 -9.26088989e-02 -7.63401240e-02
 -6.80310428e-02  9.82936565e-03  4.19009523e-03 -5.07058250e-03
 -1.21737514e-02 -9.27608982e-02  1.13813169e-01  5.62475547e-02] -0.26595011353492737*/
    static const fixed weights1[] ={
    -12095, 2026, 11208, -6585, 8351, 4797, 5474, 1475, 20918, 8468, 3499, 28890, -3141, 
    15294, 877, 8465, -1157, 14087, -11469, 9469, -4594, 10274, 1685, -4237, 2013, 7544, 
    2548, 4254, 2913, 2119, -2048, -3363, 199, 1676, 6182, 6792, 18025, -2149, -1549, 
    3882, -387, -23666, 3485, -13093, -12125, 3971, 10272, -2821, -148, 6494, 2613, 
    -1299, -12718, -3763, -5520, -18097, 10795, -745, 3830, 2304, -150, 7330, 2889, 
    23838, -5172, 483, -6107, 4801, -1583, -14354, -278, -11801, 1747, 10942, 7583, 
    8374, 5103, -5302, -9666, -2616, 1863, -2095, 3241, -11851, -5698, 13949, -3169, 
    -6070, -3961, 12249, -4323, 2248, 2113, 5599, 11241, -13988, -2292, 6265, 12371, 
    2412, 9252, 6566, 6007, 9160, 5348, -5516, 18994, 7174, -1859, 6954, 12970, 10204, 
    7290, -9188, -904, 4518, 1026, -9670, 7064, 6600, -12072, -348, -7754, -13292, -1007, 
    -3347, 10345, -12009, 13975, -11187, 8324, -7856, -17840, -6903, 11844, 10905, 3551, 
    -7929, -13, 10734, 12157, 5955, 21251, -6678, -21580, 3993, -5028, -12936, 5376, 
    1162, -20443, -6018, 9919, 3797, -6043, 2512, -19854, -7353, 7266, 6152, -9486, 
    -11778, 299, -1563, -214, 15982, -6996, -349, -12934, 4177, 10464, -9459, 3953, 
    -22304, 12085, 12988, -6444, 12853, -7158, 19206, -11461, -9284, 1035, 10078, 19250, 
    -252, -15483, 3759, 11436, -2482, 13923, -15510, -2506, -707, 4051, 3491, 2905, 
    6387, -331, 16414, 14317, 3272, -1052, 6844, 6722, 15650, -8409, 7312, 13425, 20123, 
    -10203, 35146, 30196, -9794, 13017, -3477, 15171, -18402, 3002, 2559, 28002, -12931, 
    3143, -5328, -1819, 7635, -18124, 9990, -6067, -4367, 4286, 11903, -1897, -4556, 
    14776, 19047, 9958, -2836, 25603, 1422, 1847, 5854, -10320, -3899, 3660, 6759, -1365, 
    -15693, 16742, 7752, -19969, -11671, -2242, -4570, -9858, -2581, 11722, 7268, 18831, 
    16504, -14715, -23842, 12558, -13127, 71, 22498, 6878, 13880, -4970, 10346, -27142, 
    8978, -18637, 1879, -4269, -202, -5666, -20670, -18157, 2104, -5805, 150, 4275, 
    119, 14679, 10281, 7395, -10710, 1961, 5545, -9752, -22210, -7101, 26217, 270, 11174, 
    -2168, 10074, -3256, 8793, 5070, 9503, -1096, -11214, -16122, -18591, -10496, -6035, 
    5301, 15059, -14935, -17906, 3103, -10042, -869, 2859, 16632, -9316, -3618, 74, 
    -454, 2051, -18352, 10694, 2420, -17879, -14471, 9950, -6613, 13000, -14828, 964, 
    -1575, -5601, -14902, -11050, -4061, 12428, -1898, -11880, 4848, -3327, -1207, -8679, 
    1672, 15298, -7865, -9802, -18634, 15214, 1736, 4286, -3603, 8261, -1888, -10065, 
    15344, -5192, 9924, 6915, 11170, 4184, 14057, -8053, 17357, 7201, -7933, 6590, -539, 
    4345, 9972, 8487, 16486, -1632, -6800, -9437, -16727, 4646, 10404, -3725, -4706, 
    -9838, -13, -4243, -5815, 6431, 8944, 23523, 10783, 2888, -12313, -2511, 16769, 
    -18268, 6794, -9966, 8655, 15784, 4117, -15893, -7126, 10232, -13452, -6959, 10911, 
    11040, -3824, -7130, 5254, 12231, -3145, -4576, 13335, 4002, 8946, 14203, 2423, 
    -10630, 7838, -12583, -4681, -3452, -5761, 8952, 7972, 15156, 8853, -6746, -22381, 
    -2557, -6587, -5485, -4298, -12238, -1453, 16670, -4477, -3077, -12964, -6126, -8889, 
    23939, 208, -13284, 8268, 13866, -5960, 27781, -13529, 8020, 13220, -9942, 19272, 
    -5988, -7968, -9667, 16080, 9587, -3593, -9614, 653, 11683, -7439, -2148, 10402, 
    3044, -1735, 4976, -11838, -6736, -14542, 10792, -5695, 5824, -12784, -15724, 9831, 
    16822, -9829, 2146, 6912, 12444, -2523, -9572, 21715, -4504, -5149, 657, -3395, 
    13642, -4756, -495, -12312, -22168, -22122, -12763, 831, 18861, -2534, -2742, 15185, 
    2104, -12138, -10006, -8917, 1288, 549, -665, -1596, -12158, 14918, 7372
    };
    
    static const neuron_t neuron1 = {weights1, -34859  };
    neurons[1]=neuron1;

    /* [ 1.58901922e-02  2.50386470e-03 -8.21145251e-02  9.91667137e-02
  4.47797403e-02  2.09414046e-02 -5.43927997e-02 -6.31742030e-02
 -5.53291999e-02  1.55615985e-01 -1.50850797e-02 -3.59931476e-02
  2.08014585e-02  3.22020352e-02  5.99137172e-02  1.02835909e-01
 -1.38776470e-02  1.64034851e-02  4.75455448e-02  2.35446692e-02
 -1.07398301e-01  7.58091882e-02 -4.49311212e-02 -3.27193551e-03
  1.90136060e-02 -1.16816796e-02  2.40503885e-02  1.16381124e-02
 -3.76163004e-03  1.23392045e-02  8.59274715e-03 -9.57091078e-02
  1.09553754e-01  6.57709390e-02  1.56783946e-02  5.08034267e-02
  3.03231813e-02 -8.89722481e-02 -8.12197998e-02 -8.22303742e-02
 -5.82647026e-02  5.55827357e-02  5.70823327e-02  1.25531843e-02
 -8.50940719e-02  1.06534995e-02 -6.80508688e-02 -1.80646852e-02
  8.63500219e-03  1.96293890e-01  1.41132340e-01  4.66696918e-02
  1.02339767e-01  7.04628453e-02 -2.18709046e-03  9.19727907e-02
  1.27273099e-02 -1.53748356e-02  7.75532350e-02 -2.72932518e-02
  6.49115220e-02 -2.59882454e-02  1.54695651e-02  3.44210155e-02
 -3.61002646e-02  1.99438874e-02 -5.79364710e-02 -4.92069731e-03
  9.31244157e-03  1.25017479e-01 -1.03572281e-02 -7.32435426e-03
 -1.91017017e-02  9.95237753e-02  2.21974161e-02 -5.10778874e-02
  1.17982395e-01  3.94914374e-02  5.37664294e-02  1.32396072e-01
 -1.37691386e-02  6.47625029e-02 -3.35330777e-02  6.50158226e-02
  4.07181531e-02 -3.07525322e-02  7.39108473e-02  5.47499917e-02
 -6.56817388e-03  2.30626091e-02 -3.94872017e-02 -1.38950363e-01
  2.13854685e-02 -3.97395045e-02 -9.49412398e-03  7.44480779e-03
  3.98428403e-02 -4.54903431e-02 -8.41106251e-02  6.91727400e-02
  8.42350870e-02  7.51327649e-02  9.08336882e-03 -1.96319502e-02
  8.85421634e-02 -8.36592466e-02  4.56336811e-02 -2.53302436e-02
  5.66875935e-03  2.94967592e-02  5.67150190e-02  5.47473915e-02
 -1.65048819e-02 -8.92242976e-03 -3.36558633e-02  5.20915948e-02
  7.58797452e-02 -1.97397033e-03  3.21751051e-02  1.92930773e-02
 -5.87113351e-02  2.89182402e-02 -7.16097374e-03 -1.81693025e-02
 -1.14205573e-02 -8.12516361e-02  5.61634041e-02  2.41366643e-02
 -7.90769979e-02  2.35322155e-02 -7.33916312e-02  9.49145183e-02
 -2.92049702e-02 -1.07105514e-02  1.14058964e-02  1.99721809e-02
  4.83356155e-02 -4.69506979e-02 -1.75812058e-02 -1.17901988e-01
 -2.00755574e-04  3.08032725e-02 -2.28012707e-02 -1.16926417e-01
  5.50838187e-02  1.00952148e-01 -8.00072849e-02 -8.11641011e-03
 -5.66205457e-02  8.52752104e-02 -1.01325531e-02  1.02576181e-01
 -1.10258490e-01 -9.27378535e-02  8.96994472e-02 -1.92490648e-02
 -5.26540391e-02 -5.67411110e-02 -7.93857500e-02  2.21189149e-02
 -3.89356613e-02  1.41780034e-01  4.44012620e-02  1.11059910e-02
 -7.76444525e-02 -2.54534502e-02 -5.94106950e-02  8.44292250e-03
 -4.39715795e-02  2.62422767e-02  4.78222147e-02  3.81069556e-02
  4.65300307e-02  6.20971881e-02 -4.46868092e-02 -3.48216593e-02
  7.48854950e-02  1.37794502e-02 -5.42079024e-02 -7.55851045e-02
  8.32179487e-02 -3.37400287e-02 -8.57957304e-02  2.63801217e-02
 -2.48985980e-02 -3.15359272e-02  7.04444721e-02  5.05578555e-02
 -4.92559671e-02  1.60025191e-02  1.16476767e-01  2.37418171e-02
 -3.92927714e-02  3.52688953e-02 -2.52172928e-02  2.01660339e-02
  7.16845170e-02  6.69195782e-03  1.54867338e-03  1.61839664e-01
  7.23983627e-03  8.60463530e-02 -1.28737405e-01 -6.12878911e-02
 -7.11199865e-02 -2.82049663e-02 -8.76884088e-02 -1.43734813e-02
 -1.04968846e-01 -3.17359390e-03  1.95345193e-01  1.76138967e-01
 -7.32711256e-02 -1.90171022e-02 -6.93277866e-02 -4.38853540e-02
  7.28310272e-02  2.27708910e-02  4.37809452e-02 -4.44092415e-02
 -7.99143966e-03  1.27416644e-02 -3.94439176e-02  6.55938983e-02
 -3.01616080e-02 -1.31383985e-02  2.58221440e-02 -1.26792461e-01
  9.74993035e-03 -5.53809740e-02  2.36603934e-02 -1.59578457e-01
 -1.13000631e-01  7.29171187e-02 -5.24802366e-03  6.36005476e-02
  3.49009223e-02 -3.71811762e-02 -6.17192127e-02 -2.27594450e-02
  1.52606353e-01 -1.23911098e-01 -5.94351161e-03 -8.10912251e-02
 -2.70164311e-02 -1.13936186e-01 -7.75070637e-02  1.61502045e-03
 -7.42727518e-02 -2.77179535e-02  3.72670032e-03  6.13179430e-03
  2.92817559e-02 -5.41247353e-02 -4.18970287e-02 -1.42977327e-01
 -5.69074415e-04  1.37562500e-02 -7.75861070e-02  5.31843379e-02
  3.81734148e-02 -6.33198842e-02 -7.02760667e-02 -3.97675745e-02
 -1.91724077e-02  8.96562543e-03  2.64869835e-02 -7.10298195e-02
  7.15736486e-03  7.22338334e-02 -1.26553364e-02  2.27549355e-02
  1.37106935e-02 -4.77719046e-02 -1.04261627e-02  3.23706754e-02
 -1.33994356e-01  2.99815573e-02  5.00457212e-02  4.42883857e-02
  2.35567652e-02  2.93109603e-02  3.41022536e-02  7.39082471e-02
 -9.66963172e-02 -7.12975040e-02  4.03122827e-02  2.99025443e-03
  6.58928007e-02  7.86875337e-02 -3.09332404e-02  6.83853179e-02
 -2.81223971e-02  2.74212193e-02  5.83484620e-02  1.34966932e-02
  1.81102734e-02  6.17531165e-02  7.42608905e-02  3.07640266e-02
  5.54533303e-03 -7.80589730e-02  2.87478920e-02 -2.81759519e-02
 -2.64314376e-02 -5.74146062e-02  1.79727122e-01  3.12126186e-02
  8.88535306e-02  4.57629971e-02  3.81300156e-03 -5.32627888e-02
  1.02617398e-01  7.43375644e-02  8.14704522e-02  1.23350769e-02
  8.86831805e-02  9.19636898e-03 -3.33494470e-02 -1.45910243e-02
  1.04448840e-01  9.66443866e-02  4.38838974e-02 -4.15906571e-02
  9.20010731e-03  8.47270191e-02 -2.56946646e-02 -1.10249281e-01
 -6.18762970e-02  1.10080473e-01  4.36625145e-02  2.40537780e-03
  1.39726615e-02 -2.96851918e-02  8.60549808e-02 -6.69689849e-02
  8.62231776e-02 -6.90139756e-02  3.99260409e-02  1.55367246e-02
 -1.83243714e-02  1.98884998e-02  1.22519396e-02 -1.14981104e-02
 -1.51974680e-02 -6.49639145e-02 -8.05808231e-03 -8.89121816e-02
  2.09844392e-03 -1.48145575e-02 -3.52216586e-02  7.86470845e-02
 -6.58078790e-02 -5.42400926e-02  3.21965404e-02 -2.17915382e-02
  3.80260348e-02 -1.94529057e-01 -8.16606451e-03 -2.35963780e-02
  3.45437303e-02 -6.34843903e-03 -6.49116263e-02 -5.92823587e-02
  4.91308868e-02 -4.36099768e-02 -1.06163207e-03 -1.10183023e-01
 -3.32095511e-02 -1.25972251e-03 -7.14215683e-03 -6.81408271e-02
  8.01369622e-02  9.86114051e-03 -4.60428372e-02  6.99292272e-02
  4.51095290e-02  7.14991167e-02  8.32391605e-02 -4.36871201e-02
 -1.32486910e-01  1.39074838e-02  1.13187321e-01 -5.56169562e-02
  3.35837491e-02  1.76497363e-02 -8.77780765e-02 -1.63683593e-02
 -9.54446495e-02 -4.79682460e-02 -2.16327440e-02  2.49976665e-02
 -7.80774578e-02  9.96108055e-02  5.95608689e-02  2.61865295e-02
  6.52997792e-02  1.12790093e-02 -4.66466276e-03 -3.04178298e-02
 -5.61009087e-02  1.24144517e-01 -4.02807519e-02  2.76896302e-02
 -4.67843227e-02 -1.06222376e-01  8.42680335e-02  5.09452149e-02
 -9.70530137e-02  9.11297873e-02 -3.75235192e-02 -1.78022688e-04
  2.73260009e-02 -2.18451545e-02 -1.71285328e-02 -1.12276532e-01
 -1.45044355e-02  1.47782676e-02 -6.63311630e-02  2.38774307e-02
 -8.05908367e-02 -4.60003987e-02 -1.74808204e-02 -3.73735875e-02
 -9.79684666e-02  2.83703320e-02  2.86626555e-02  8.66264403e-02
 -6.99169487e-02 -6.85748756e-02 -8.07566661e-03 -1.29022608e-02
 -5.71067743e-02  5.18084951e-02 -5.57385758e-02 -4.33373563e-02
  3.90930829e-04  8.99920538e-02  8.32370222e-02  5.50379567e-02
 -6.45619184e-02 -5.15051186e-02  3.17146741e-02  2.94616558e-02
  1.05271764e-01 -1.08460698e-03 -3.60248983e-02  1.15631474e-02
  3.20613272e-02  5.35129197e-02 -6.41143620e-02  2.12714206e-02
 -4.33213748e-02  3.06929592e-02 -1.18589520e-01  1.02248199e-01
 -4.69775237e-02  4.32982482e-02 -4.98263091e-02 -5.35658747e-02
 -8.08146130e-03  3.07679828e-02  6.89249113e-02 -1.23662367e-01
  2.48683095e-02  1.67679042e-01  1.04537525e-03 -4.60341945e-02
  1.66418299e-01  9.24589559e-02  1.80594425e-03 -2.30250712e-02
 -5.15032783e-02  2.48073973e-02 -1.55753326e-02 -1.84217598e-02
  4.55333926e-02 -4.09590937e-02  5.27676977e-02  1.31630197e-01
 -1.15135182e-02 -9.33981761e-02 -8.49642307e-02  5.04304208e-02
  2.70958655e-02 -6.73631057e-02  4.75362092e-02  2.03437954e-02
  1.74697873e-03  1.18606970e-01 -6.20911680e-02  8.73139650e-02
  1.12862047e-02 -2.50243060e-02  7.54248500e-02 -4.03900370e-02
  4.15798239e-02 -4.92567047e-02  7.35635757e-02  7.01310635e-02
 -6.99009979e-03 -2.75426097e-02 -3.29829529e-02 -2.46295407e-02
  9.72597227e-02 -1.22567378e-01  6.60555735e-02 -3.42080221e-02
  1.09547645e-01  5.44416197e-02  5.88582680e-02 -3.25810798e-02] 0.004709299188107252*/
    static const fixed weights2[] ={
    2083, 328, -10763, 12998, 5869, 2745, -7129, -8280, -7252, 20397, -1977, -4718, 
    2726, 4221, 7853, 13479, -1819, 2150, 6232, 3086, -14077, 9936, -5889, -429, 2492, 
    -1531, 3152, 1525, -493, 1617, 1126, -12545, 14359, 8621, 2055, 6659, 3975, -11662, 
    -10646, -10778, -7637, 7285, 7482, 1645, -11153, 1396, -8920, -2368, 1132, 25729, 
    18498, 6117, 13414, 9236, -287, 12055, 1668, -2015, 10165, -3577, 8508, -3406, 2028, 
    4512, -4732, 2614, -7594, -645, 1221, 16386, -1358, -960, -2504, 13045, 2909, -6695, 
    15464, 5176, 7047, 17353, -1805, 8489, -4395, 8522, 5337, -4031, 9688, 7176, -861, 
    3023, -5176, -18213, 2803, -5209, -1244, 976, 5222, -5963, -11025, 9067, 11041, 
    9848, 1191, -2573, 11605, -10965, 5981, -3320, 743, 3866, 7434, 7176, -2163, -1169, 
    -4411, 6828, 9946, -259, 4217, 2529, -7695, 3790, -939, -2381, -1497, -10650, 7361, 
    3164, -10365, 3084, -9620, 12441, -3828, -1404, 1495, 2618, 6335, -6154, -2304, 
    -15454, -26, 4037, -2989, -15326, 7220, 13232, -10487, -1064, -7421, 11177, -1328, 
    13445, -14452, -12155, 11757, -2523, -6901, -7437, -10405, 2899, -5103, 18583, 5820, 
    1456, -10177, -3336, -7787, 1107, -5763, 3440, 6268, 4995, 6099, 8139, -5857, -4564, 
    9815, 1806, -7105, -9907, 10908, -4422, -11245, 3458, -3264, -4133, 9233, 6627, 
    -6456, 2097, 15267, 3112, -5150, 4623, -3305, 2643, 9396, 877, 203, 21213, 949, 
    11278, -16874, -8033, -9322, -3697, -11493, -1884, -13758, -416, 25604, 23087, -9604, 
    -2493, -9087, -5752, 9546, 2985, 5738, -5821, -1047, 1670, -5170, 8598, -3953, -1722, 
    3385, -16619, 1278, -7259, 3101, -20916, -14811, 9557, -688, 8336, 4575, -4873, 
    -8090, -2983, 20002, -16241, -779, -10629, -3541, -14934, -10159, 212, -9735, -3633, 
    488, 804, 3838, -7094, -5492, -18740, -75, 1803, -10169, 6971, 5003, -8299, -9211, 
    -5212, -2513, 1175, 3472, -9310, 938, 9468, -1659, 2983, 1797, -6262, -1367, 4243, 
    -17563, 3930, 6560, 5805, 3088, 3842, 4470, 9687, -12674, -9345, 5284, 392, 8637, 
    10314, -4054, 8963, -3686, 3594, 7648, 1769, 2374, 8094, 9734, 4032, 727, -10231, 
    3768, -3693, -3464, -7525, 23557, 4091, 11646, 5998, 500, -6981, 13450, 9744, 10678, 
    1617, 11624, 1205, -4371, -1912, 13690, 12667, 5752, -5451, 1206, 11105, -3368, 
    -14451, -8110, 14428, 5723, 315, 1831, -3891, 11279, -8778, 11301, -9046, 5233, 
    2036, -2402, 2607, 1606, -1507, -1992, -8515, -1056, -11654, 275, -1942, -4617, 
    10308, -8626, -7109, 4220, -2856, 4984, -25497, -1070, -3093, 4528, -832, -8508, 
    -7770, 6440, -5716, -139, -14442, -4353, -165, -936, -8931, 10504, 1293, -6035, 
    9166, 5913, 9372, 10910, -5726, -17365, 1823, 14836, -7290, 4402, 2313, -11505, 
    -2145, -12510, -6287, -2835, 3276, -10234, 13056, 7807, 3432, 8559, 1478, -611, 
    -3987, -7353, 16272, -5280, 3629, -6132, -13923, 11045, 6677, -12721, 11945, -4918, 
    -23, 3582, -2863, -2245, -14716, -1901, 1937, -8694, 3130, -10563, -6029, -2291, 
    -4899, -12841, 3719, 3757, 11354, -9164, -8988, -1058, -1691, -7485, 6791, -7306, 
    -5680, 51, 11795, 10910, 7214, -8462, -6751, 4157, 3862, 13798, -142, -4722, 1516, 
    4202, 7014, -8404, 2788, -5678, 4023, -15544, 13402, -6157, 5675, -6531, -7021, 
    -1059, 4033, 9034, -16209, 3260, 21978, 137, -6034, 21813, 12119, 237, -3018, -6751, 
    3252, -2041, -2415, 5968, -5369, 6916, 17253, -1509, -12242, -11136, 6610, 3552, 
    -8829, 6231, 2667, 229, 15546, -8138, 11444, 1479, -3280, 9886, -5294, 5450, -6456, 
    9642, 9192, -916, -3610, -4323, -3228, 12748, -16065, 8658, -4484, 14359, 7136, 
    7715, -4270
    };
    
    static const neuron_t neuron2 = {weights2, 617  };
    neurons[2]=neuron2;

    /* [-0.00330568 -0.11225482 -0.03659848  0.06057881 -0.07403172  0.0072199
 -0.06254488  0.11540689 -0.10878894 -0.02845326  0.06261259 -0.03798804
  0.07271527  0.08036654  0.00046263 -0.02885656 -0.02049387 -0.00596539
  0.0667998  -0.06725708 -0.06489483 -0.10069928 -0.06166933  0.03820678
 -0.03366483 -0.00171376 -0.02202983 -0.03789707  0.05565003 -0.01556506
  0.04933263 -0.11762446 -0.00494086 -0.02327169 -0.0776969  -0.13598882
  0.04910961 -0.05670342  0.03631305  0.04029381  0.09331568 -0.00332859
  0.04548219  0.09947682  0.02802608  0.04224557 -0.02050784  0.08804414
  0.00927796 -0.01685821 -0.02247508 -0.05934979 -0.01614699 -0.04222655
  0.05226869  0.09574788  0.08671611 -0.04809657  0.03939208  0.08750942
 -0.03865962 -0.09316505  0.06632232 -0.01864146  0.0180575  -0.03355669
 -0.06111162 -0.0072856  -0.02160041  0.05264712  0.01620024  0.11473963
  0.10595838 -0.07879601  0.07670346  0.02438964 -0.06264643 -0.08869541
  0.05202827  0.08822983 -0.00994126  0.12150344 -0.09678355  0.00615557
  0.04332248 -0.07708847 -0.05072729 -0.03121969 -0.01677147 -0.09838536
 -0.08738907  0.0948716   0.02554866  0.03468501 -0.01560167  0.03260599
  0.00785208 -0.01191446 -0.12930046 -0.06908042 -0.01969654  0.02946912
 -0.00256493  0.08334259  0.01088237 -0.03032936 -0.07140389 -0.06895778
  0.02017435 -0.03104353 -0.04628395 -0.00871741 -0.0223729  -0.01344928
  0.01452572  0.02562474 -0.05134394  0.03445302  0.00409173 -0.01034007
  0.01450337 -0.01937206 -0.02411954 -0.02534862  0.00285405  0.01030758
  0.06415479 -0.04126889 -0.0135488   0.08165984 -0.02139134  0.00611122
 -0.01647567 -0.10826448 -0.02709373 -0.14159805  0.03003943  0.11038865
 -0.05227426 -0.06238483  0.00729067 -0.08695053 -0.00587515 -0.05077498
  0.02844108  0.08977507 -0.0263922  -0.07859191 -0.00720069 -0.093476
  0.05572219  0.07189821 -0.05059846 -0.01088973 -0.03793102  0.01766763
  0.05747285 -0.04294126  0.07163448 -0.04341303  0.00100689 -0.00598657
 -0.05316311  0.06471344  0.00891036  0.00742961 -0.05497195 -0.02034833
 -0.04384409  0.02864532  0.02615124  0.01049875 -0.00786967  0.04916506
  0.02062635 -0.02706762 -0.08169806  0.07479643 -0.00313391 -0.058075
  0.04080964  0.03714814  0.09887429  0.02916742 -0.0418956  -0.04623229
 -0.04182184 -0.01505773 -0.01596597 -0.01612663 -0.04177076  0.00511232
  0.03155995 -0.00236911 -0.11641023 -0.0175129  -0.06395628  0.05484476
 -0.00997611 -0.04147131 -0.01918386  0.00023878 -0.00552029  0.11335275
 -0.0043681   0.00299127  0.05970248 -0.07549699 -0.05666197  0.01577521
  0.00772148 -0.01591402  0.04444107 -0.00362921  0.02880224 -0.0640953
 -0.00387661  0.00875532 -0.07576098 -0.04083202  0.01944965  0.09141608
 -0.00920065  0.05100904 -0.02730752 -0.0104159   0.18931268  0.0781905
 -0.01459551 -0.05785516 -0.01506658 -0.05056245 -0.07147456  0.02156265
  0.02886989  0.02370218 -0.04856063  0.11753582 -0.09236965 -0.06279352
 -0.11891527 -0.05481384 -0.0391746   0.0624145  -0.12860222  0.0903367
 -0.01776935  0.01696755 -0.01097804  0.03651695  0.03288753 -0.06602582
 -0.00724608 -0.05777235  0.02535372 -0.01473515  0.00135157 -0.05524336
  0.01766177 -0.00482544  0.00918873  0.0550932   0.04259019  0.11640851
 -0.02316834  0.00944792  0.02960515 -0.01038827 -0.0045789  -0.03999255
 -0.11428232  0.01536721 -0.00148124  0.00092626  0.00472041  0.00488716
  0.02462403 -0.06537991  0.12296829 -0.05609127 -0.0631587  -0.00202334
  0.1149228   0.00275118  0.03566337 -0.10084547  0.04728143 -0.04414728
 -0.11267514 -0.04271506 -0.15821503 -0.00193851  0.11692242 -0.02887572
 -0.04559473 -0.05757416 -0.01674181 -0.08472908  0.10085572 -0.11011165
 -0.00198397 -0.0212279  -0.07239968 -0.06604047  0.03992191  0.04349931
  0.12173293  0.05649029  0.03364522 -0.07991017 -0.00135314 -0.13506694
  0.02562358 -0.00584785 -0.07197224  0.00370583  0.01479643 -0.03998269
 -0.03605347  0.01837223 -0.00033843  0.02197767  0.10246309  0.02594848
 -0.03077438  0.0949275  -0.03938605  0.019254   -0.03022454 -0.07915376
 -0.01962334 -0.04695915 -0.02417021  0.05719808  0.02069264  0.08805726
  0.04597554 -0.04877966  0.06164467 -0.05764184 -0.06524296  0.07689843
  0.03100689  0.02148964  0.03628588 -0.1358663  -0.04485023  0.01084393
  0.09435271 -0.03310756 -0.03767064 -0.00452982 -0.04075801  0.05068669
 -0.1880842  -0.00379505  0.01661725  0.03194683 -0.01602564  0.00979228
 -0.01506041  0.05321439  0.09640129 -0.06350005  0.07706702 -0.04961781
 -0.04283144 -0.05734469  0.03978138  0.09786821  0.05604152 -0.04573713
 -0.09221766  0.02469828  0.02166467 -0.04965413 -0.03253375  0.04041625
 -0.05174769  0.00842379 -0.08469419  0.04150962 -0.0509396   0.05613842
  0.00210449  0.05546801 -0.00257085 -0.1048309  -0.01611804 -0.04276261
 -0.06122749  0.11044758 -0.0224223  -0.00573699  0.01885085 -0.0195052
 -0.05274922 -0.03869166  0.03312742 -0.00565868 -0.03801014  0.0026299
  0.02446603  0.01916378 -0.06606273 -0.05386985  0.04293426  0.06273527
 -0.03008785  0.01877937 -0.06676324 -0.03313066  0.04332236 -0.04396713
 -0.04000245  0.02836711  0.02408467 -0.02167037 -0.01333707  0.10659912
 -0.01626128 -0.08093593  0.02202256 -0.0735969   0.03700199 -0.0020167
 -0.03382351 -0.06192032  0.05788186  0.05339079  0.01349774 -0.09686171
  0.02477736  0.11871447 -0.13121755 -0.08883078  0.06732687  0.10733156
  0.0257797  -0.05878765  0.00131539  0.03714984  0.00375654  0.04610589
  0.06205662 -0.01301981 -0.03394021 -0.04385889  0.03768083  0.07764458
 -0.0365707   0.00253546  0.01402497 -0.00320781 -0.02418352 -0.003801
 -0.04173175 -0.07634718 -0.04178216 -0.00447212 -0.06223955  0.00824534
 -0.05294549  0.0117754   0.03162002 -0.03634311 -0.06726258 -0.0548909
  0.08304366  0.08957981 -0.04350788  0.05294812  0.13432404 -0.04420844
 -0.03668622  0.05919801  0.0189882  -0.03931155  0.11282464  0.01765718
  0.00124447  0.03675801 -0.04800596 -0.0091371   0.04426442 -0.05701721
  0.0147915  -0.12076957  0.006998   -0.05304134  0.03044707  0.09923426
  0.06019907  0.05493988 -0.01314412 -0.10535475  0.01360424  0.01564872
 -0.01675126  0.11557091 -0.00415254 -0.01892621  0.05906697  0.04125835
 -0.07188178 -0.02254372 -0.00398251  0.03622831  0.0227184  -0.00322984
 -0.05761343 -0.04481041] 0.2800866961479187*/
    static const fixed weights3[] ={
    -433, -14713, -4797, 7940, -9703, 946, -8198, 15127, -14259, -3729, 8207, -4979, 
    9531, 10534, 61, -3782, -2686, -782, 8756, -8816, -8506, -13199, -8083, 5008, -4413, 
    -225, -2887, -4967, 7294, -2040, 6466, -15417, -648, -3050, -10184, -17824, 6437, 
    -7432, 4760, 5281, 12231, -436, 5961, 13039, 3673, 5537, -2688, 11540, 1216, -2210, 
    -2946, -7779, -2116, -5535, 6851, 12550, 11366, -6304, 5163, 11470, -5067, -12211, 
    8693, -2443, 2367, -4398, -8010, -955, -2831, 6901, 2123, 15039, 13888, -10328, 
    10054, 3197, -8211, -11625, 6819, 11564, -1303, 15926, -12686, 807, 5678, -10104, 
    -6649, -4092, -2198, -12896, -11454, 12435, 3349, 4546, -2045, 4274, 1029, -1562, 
    -16948, -9055, -2582, 3863, -336, 10924, 1426, -3975, -9359, -9038, 2644, -4069, 
    -6067, -1143, -2932, -1763, 1904, 3359, -6730, 4516, 536, -1355, 1901, -2539, -3161, 
    -3322, 374, 1351, 8409, -5409, -1776, 10703, -2804, 801, -2159, -14190, -3551, -18560, 
    3937, 14469, -6852, -8177, 956, -11397, -770, -6655, 3728, 11767, -3459, -10301, 
    -944, -12252, 7304, 9424, -6632, -1427, -4972, 2316, 7533, -5628, 9389, -5690, 132, 
    -785, -6968, 8482, 1168, 974, -7205, -2667, -5747, 3755, 3428, 1376, -1031, 6444, 
    2704, -3548, -10708, 9804, -411, -7612, 5349, 4869, 12960, 3823, -5491, -6060, -5482, 
    -1974, -2093, -2114, -5475, 670, 4137, -311, -15258, -2295, -8383, 7189, -1308, 
    -5436, -2514, 31, -724, 14857, -573, 392, 7825, -9896, -7427, 2068, 1012, -2086, 
    5825, -476, 3775, -8401, -508, 1148, -9930, -5352, 2549, 11982, -1206, 6686, -3579, 
    -1365, 24814, 10249, -1913, -7583, -1975, -6627, -9368, 2826, 3784, 3107, -6365, 
    15406, -12107, -8230, -15586, -7185, -5135, 8181, -16856, 11841, -2329, 2224, -1439, 
    4786, 4311, -8654, -950, -7572, 3323, -1931, 177, -7241, 2315, -632, 1204, 7221, 
    5582, 15258, -3037, 1238, 3880, -1362, -600, -5242, -14979, 2014, -194, 121, 619, 
    641, 3228, -8569, 16118, -7352, -8278, -265, 15063, 361, 4674, -13218, 6197, -5786, 
    -14769, -5599, -20738, -254, 15325, -3785, -5976, -7546, -2194, -11106, 13219, -14433, 
    -260, -2782, -9490, -8656, 5233, 5702, 15956, 7404, 4410, -10474, -177, -17703, 
    3359, -766, -9434, 486, 1939, -5241, -4726, 2408, -44, 2881, 13430, 3401, -4034, 
    12442, -5162, 2524, -3962, -10375, -2572, -6155, -3168, 7497, 2712, 11542, 6026, 
    -6394, 8080, -7555, -8552, 10079, 4064, 2817, 4756, -17808, -5879, 1421, 12367, 
    -4339, -4938, -594, -5342, 6644, -24653, -497, 2178, 4187, -2101, 1283, -1974, 6975, 
    12636, -8323, 10101, -6504, -5614, -7516, 5214, 12828, 7345, -5995, -12087, 3237, 
    2840, -6508, -4264, 5297, -6783, 1104, -11101, 5441, -6677, 7358, 276, 7270, -337, 
    -13740, -2113, -5605, -8025, 14477, -2939, -752, 2471, -2557, -6914, -5071, 4342, 
    -742, -4982, 345, 3207, 2512, -8659, -7061, 5627, 8223, -3944, 2461, -8751, -4343, 
    5678, -5763, -5243, 3718, 3157, -2840, -1748, 13972, -2131, -10608, 2887, -9646, 
    4850, -264, -4433, -8116, 7587, 6998, 1769, -12696, 3248, 15560, -17199, -11643, 
    8825, 14068, 3379, -7705, 172, 4869, 492, 6043, 8134, -1707, -4449, -5749, 4939, 
    10177, -4793, 332, 1838, -420, -3170, -498, -5470, -10007, -5476, -586, -8158, 1081, 
    -6940, 1543, 4144, -4764, -8816, -7195, 10885, 11741, -5703, 6940, 17606, -5794, 
    -4809, 7759, 2489, -5153, 14788, 2314, 163, 4818, -6292, -1198, 5802, -7473, 1939, 
    -15830, 917, -6952, 3991, 13007, 7890, 7201, -1723, -13809, 1783, 2051, -2196, 15148, 
    -544, -2481, 7742, 5408, -9422, -2955, -522, 4749, 2978, -423, -7552, -5873
    };
    
    static const neuron_t neuron3 = {weights3, 36712  };
    neurons[3]=neuron3;

    /* [ 0.01743422 -0.0564864  -0.04342288 -0.07252723  0.03361699 -0.09844182
 -0.04719643 -0.04274545  0.0051492   0.02862985  0.10954034  0.13596879
 -0.0410146  -0.01788445 -0.04545007 -0.05816762 -0.06340484  0.02123668
 -0.04682928 -0.03215776  0.02332763  0.10739071  0.05287975 -0.0309416
 -0.01038936  0.07530231 -0.04689553  0.05185225 -0.07751677  0.03851887
  0.09503578 -0.00285957  0.00451872  0.12152144  0.01017527 -0.03127884
 -0.07297181  0.01365913 -0.07085864 -0.06974049 -0.02330359 -0.0376967
  0.04291762  0.06638514 -0.05125178  0.02425087 -0.10230713  0.04012919
  0.06914444 -0.03489775  0.0327796   0.00152093 -0.07149992 -0.04789622
  0.04932418  0.11634605 -0.11938545  0.03507997 -0.03610579  0.0885314
 -0.00939048 -0.00597574  0.05722206  0.00888241 -0.00219455 -0.05693366
  0.09880202 -0.00210067 -0.00407291  0.04437183 -0.06031071 -0.0586739
  0.02146559  0.01386567 -0.14176391  0.10533207  0.04861127  0.02400606
  0.10574001  0.02363234 -0.07742926  0.05151151  0.06287142 -0.05741218
  0.01420931  0.01184604 -0.02940234 -0.04785049  0.03410257  0.11701234
  0.06550591  0.0578641  -0.06126517  0.02380093 -0.00571189 -0.06143573
 -0.04473306  0.05108127 -0.03726444 -0.0615409  -0.02048359  0.0332703
 -0.0208675  -0.08344733  0.08148355 -0.06412077  0.0287524  -0.0138108
  0.02494737 -0.04072986 -0.0343879  -0.15307088 -0.02956441  0.03319081
  0.01156104 -0.01026165  0.02006426  0.16649577 -0.0682406   0.05026021
  0.05225655  0.09864236 -0.01933828 -0.01559586  0.0279063   0.02288895
 -0.02226854  0.06739439 -0.05658453  0.09306456  0.02058954 -0.07562942
 -0.03077647 -0.00775539 -0.09249648 -0.03622308 -0.07522456 -0.01745806
 -0.00937268 -0.09459461 -0.07395882 -0.09053133 -0.0462473   0.01922575
  0.12968309  0.12061592  0.00413317  0.09784514  0.03867565  0.02956333
  0.01302678  0.13634838 -0.03989332 -0.10401939  0.06894409  0.04439019
 -0.07136929  0.14755109 -0.08074924  0.0219289   0.01805648 -0.02408426
 -0.09259801 -0.00647358 -0.01291778 -0.11498979  0.00899183 -0.05028238
 -0.08679539  0.0181462   0.02278851  0.03024112  0.01659176  0.04578764
  0.00449574  0.0938563  -0.07594843 -0.04768471 -0.00873458 -0.05326023
 -0.01875419  0.0182007  -0.01274662 -0.05359669  0.03222095  0.13859606
  0.17586792 -0.02986827 -0.04146006  0.01519237 -0.04008051  0.09587315
 -0.03279979  0.03472603 -0.07377221 -0.04918312 -0.00267939 -0.02651867
 -0.07014414 -0.0633401  -0.01162056 -0.0417934  -0.01867642 -0.17191267
 -0.0185818  -0.03484352  0.08403733 -0.00455896 -0.04410209  0.0964383
  0.13755345  0.036793    0.04235616 -0.08851623  0.07555713  0.00997846
 -0.01722242  0.07165732 -0.08732495  0.02283625 -0.00439267 -0.01431894
  0.09541875  0.0040954  -0.0220619   0.03145041 -0.02641176  0.03695204
 -0.0515431  -0.12962945  0.0715067  -0.1490667  -0.04642003 -0.11796767
 -0.09363023 -0.04600293  0.00305109  0.02934813  0.05155597  0.16282542
  0.00842988 -0.01648367  0.00703341  0.06475884  0.01872191 -0.09537765
  0.05109634 -0.00598072 -0.13662432 -0.08761773 -0.0397432   0.05160249
 -0.01144913 -0.05719963  0.02339247 -0.08167689  0.05680874  0.08324134
 -0.18298127  0.00344742  0.07338759 -0.01861253 -0.03772928  0.07903837
 -0.09162139 -0.04841253 -0.02916505  0.00854142  0.04680062 -0.01196001
  0.07523869  0.00573469  0.09727848 -0.07661483 -0.0012712   0.00533119
 -0.03450471  0.06275815  0.04137753  0.0506024  -0.15614781 -0.01220823
 -0.12850514  0.03932445  0.07350037 -0.01702549  0.11055479  0.04587187
  0.06993672 -0.02782249  0.08373371 -0.05192189 -0.05338642 -0.08165769
  0.08708297 -0.06237679 -0.00785256  0.00786619  0.0176512   0.11764818
  0.01807577  0.04621527  0.05495651  0.05924745 -0.04193645 -0.01767101
  0.01522397  0.06064427  0.00313985 -0.02004547  0.15747182  0.02730114
 -0.06006559  0.05565976  0.02815463 -0.02170089 -0.01323927  0.00934345
 -0.0491229   0.00985408  0.0526876  -0.04196514  0.03050279 -0.02496684
 -0.01079334 -0.09345081 -0.02551422 -0.05122847  0.05448945  0.11444413
  0.01201093  0.07806382  0.03768692 -0.03192724  0.12384229 -0.02083037
  0.05719354  0.06385662 -0.01839995  0.05129846  0.05723464 -0.01677311
 -0.01194783 -0.02251678 -0.01643424 -0.21382618 -0.05533383 -0.0627931
  0.09930291 -0.09664762 -0.00950336 -0.08546928 -0.01550452 -0.06762744
 -0.00212175  0.03170335  0.09206297 -0.01719688 -0.11714128 -0.01875153
  0.0882578   0.04673529  0.09871988  0.00275454  0.0099652   0.09548608
 -0.01618234 -0.02905816 -0.04054452 -0.03109415 -0.06856153  0.02470943
  0.01211013  0.04182401 -0.09218075  0.06742707  0.04653989  0.04821169
 -0.05229031  0.07831559  0.02659664  0.04933851 -0.01778753  0.05793078
 -0.03015641 -0.02671939  0.10116637 -0.11512495  0.06602278 -0.07516942
 -0.03856271 -0.03715949 -0.07128905  0.03399809  0.04731314  0.02644195
  0.02014531  0.08758105 -0.03311425  0.0607702   0.06700154  0.00601495
 -0.02268064  0.06520665 -0.01649688  0.02776954  0.0386329  -0.10693163
 -0.11228438  0.00530146  0.06061703 -0.04882222 -0.090346   -0.15097725
  0.09291837  0.00729001 -0.00494615  0.11921867 -0.03907915  0.14028615
 -0.03127493 -0.04062019  0.01629156 -0.09163649 -0.08468261 -0.03161881
 -0.07878613  0.02385595 -0.0156704  -0.04023562  0.06858945 -0.08085899
 -0.04718652  0.05658802  0.01894701 -0.01019846  0.00933302  0.00213969
  0.03610901  0.08234894  0.03714732 -0.00706821  0.05162802 -0.00064478
 -0.09370842 -0.06161878 -0.01227847 -0.01826128  0.01293688  0.08471261
 -0.00395379  0.00145356 -0.05210453  0.09628153 -0.06484933 -0.08412752
  0.01264572 -0.09599241  0.13304907 -0.11201254  0.00800261  0.04040682
 -0.09136143  0.05873661  0.0534451   0.13803342 -0.10344916 -0.02094728
  0.0586671  -0.06597565  0.000914   -0.13889617  0.00290625  0.05786686
 -0.04530937  0.04116733  0.0134227  -0.13188803  0.0348487   0.01548799
 -0.00063198 -0.04429082  0.04003981  0.07618117 -0.01989158  0.02842873
 -0.0231418  -0.09222353 -0.1340214  -0.04785422  0.05879897  0.00047808
  0.01239227  0.12526883  0.01399072  0.05982948  0.13810381 -0.11178321
  0.0417431   0.0964452  -0.08388844  0.05130308 -0.09229396  0.03304983
  0.06215967 -0.07749707 -0.02690163 -0.03289502  0.03347208  0.00666461
 -0.03716804 -0.04944476] 0.0996442586183548*/
    static const fixed weights4[] ={
    2285, -7404, -5692, -9506, 4406, -12903, -6186, -5603, 675, 3753, 14358, 17822, 
    -5376, -2344, -5957, -7624, -8311, 2784, -6138, -4215, 3058, 14076, 6931, -4056, 
    -1362, 9870, -6147, 6796, -10160, 5049, 12457, -375, 592, 15928, 1334, -4100, -9565, 
    1790, -9288, -9141, -3054, -4941, 5625, 8701, -6718, 3179, -13410, 5260, 9063, -4574, 
    4296, 199, -9372, -6278, 6465, 15250, -15648, 4598, -4732, 11604, -1231, -783, 7500, 
    1164, -288, -7462, 12950, -275, -534, 5816, -7905, -7691, 2814, 1817, -18581, 13806, 
    6372, 3147, 13860, 3098, -10149, 6752, 8241, -7525, 1862, 1553, -3854, -6272, 4470, 
    15337, 8586, 7584, -8030, 3120, -749, -8053, -5863, 6695, -4884, -8066, -2685, 4361, 
    -2735, -10938, 10680, -8404, 3769, -1810, 3270, -5339, -4507, -20063, -3875, 4350, 
    1515, -1345, 2630, 21823, -8944, 6588, 6849, 12929, -2535, -2044, 3658, 3000, -2919, 
    8834, -7417, 12198, 2699, -9913, -4034, -1017, -12124, -4748, -9860, -2288, -1228, 
    -12399, -9694, -11866, -6062, 2520, 16998, 15809, 542, 12825, 5069, 3875, 1707, 
    17871, -5229, -13634, 9037, 5818, -9355, 19340, -10584, 2874, 2367, -3157, -12137, 
    -849, -1693, -15072, 1179, -6591, -11376, 2378, 2987, 3964, 2175, 6001, 589, 12302, 
    -9955, -6250, -1145, -6981, -2458, 2386, -1671, -7025, 4223, 18166, 23051, -3915, 
    -5434, 1991, -5253, 12566, -4299, 4552, -9669, -6447, -351, -3476, -9194, -8302, 
    -1523, -5478, -2448, -22533, -2436, -4567, 11015, -598, -5781, 12640, 18029, 4823, 
    5552, -11602, 9903, 1308, -2257, 9392, -11446, 2993, -576, -1877, 12507, 537, -2892, 
    4122, -3462, 4843, -6756, -16991, 9373, -19538, -6084, -15462, -12272, -6030, 400, 
    3847, 6758, 21342, 1105, -2161, 922, 8488, 2454, -12501, 6697, -784, -17908, -11484, 
    -5209, 6764, -1501, -7497, 3066, -10706, 7446, 10911, -23984, 452, 9619, -2440, 
    -4945, 10360, -12009, -6346, -3823, 1120, 6134, -1568, 9862, 752, 12750, -10042, 
    -167, 699, -4523, 8226, 5423, 6633, -20467, -1600, -16843, 5154, 9634, -2232, 14491, 
    6013, 9167, -3647, 10975, -6806, -6997, -10703, 11414, -8176, -1029, 1031, 2314, 
    15420, 2369, 6058, 7203, 7766, -5497, -2316, 1995, 7949, 412, -2627, 20640, 3578, 
    -7873, 7295, 3690, -2844, -1735, 1225, -6439, 1292, 6906, -5500, 3998, -3272, -1415, 
    -12249, -3344, -6715, 7142, 15000, 1574, 10232, 4940, -4185, 16232, -2730, 7496, 
    8370, -2412, 6724, 7502, -2198, -1566, -2951, -2154, -28027, -7253, -8230, 13016, 
    -12668, -1246, -11203, -2032, -8864, -278, 4155, 12067, -2254, -15354, -2458, 11568, 
    6126, 12939, 361, 1306, 12516, -2121, -3809, -5314, -4076, -8986, 3239, 1587, 5482, 
    -12082, 8838, 6100, 6319, -6854, 10265, 3486, 6467, -2331, 7593, -3953, -3502, 13260, 
    -15090, 8654, -9853, -5054, -4871, -9344, 4456, 6201, 3466, 2640, 11479, -4340, 
    7965, 8782, 788, -2973, 8547, -2162, 3640, 5064, -14016, -14717, 695, 7945, -6399, 
    -11842, -19789, 12179, 956, -648, 15626, -5122, 18388, -4099, -5324, 2135, -12011, 
    -11100, -4144, -10327, 3127, -2054, -5274, 8990, -10598, -6185, 7417, 2483, -1337, 
    1223, 280, 4733, 10794, 4869, -926, 6767, -85, -12283, -8076, -1609, -2394, 1696, 
    11103, -518, 191, -6829, 12620, -8500, -11027, 1657, -12582, 17439, -14682, 1049, 
    5296, -11975, 7699, 7005, 18092, -13559, -2746, 7690, -8648, 120, -18205, 381, 7585, 
    -5939, 5396, 1759, -17287, 4568, 2030, -83, -5805, 5248, 9985, -2607, 3726, -3033, 
    -12088, -17566, -6272, 7707, 63, 1624, 16419, 1834, 7842, 18102, -14652, 5471, 12641, 
    -10995, 6724, -12097, 4332, 8147, -10158, -3526, -4312, 4387, 874, -4872, -6481, 
  
    };
    
    static const neuron_t neuron4 = {weights4, 13061  };
    neurons[4]=neuron4;

    /* [-0.09589487 -0.0593698  -0.04828273 -0.05127501  0.04123526  0.0493114
  0.04257531 -0.04192989 -0.06515727 -0.12265882  0.00579592  0.00804975
 -0.04353926  0.08612403 -0.03476084 -0.02150061 -0.11719225 -0.08500215
 -0.07623783 -0.01411657  0.0037563   0.06241248 -0.04323272 -0.04205195
 -0.01156156 -0.10644062  0.02316441 -0.03429053  0.06741362  0.09398336
 -0.03027184 -0.14469366 -0.01930413  0.01343491 -0.0188082  -0.0528457
 -0.00736062  0.04036079  0.00587205 -0.0316083   0.03212801 -0.09555937
  0.04042352  0.01741022  0.0526071  -0.01374918 -0.06133733 -0.00648744
  0.08408493 -0.10017509  0.04754449 -0.0613239  -0.04220182 -0.02334946
  0.07612564  0.06488157  0.01543564 -0.0230653  -0.06837966  0.08232005
 -0.1302342  -0.0268809   0.00104469 -0.06843623 -0.01078568 -0.1156939
 -0.03449291  0.05150262 -0.06628564  0.00899748 -0.06312489  0.10560574
 -0.01183981 -0.03390446  0.07476916 -0.05254219 -0.06264275 -0.02182471
  0.02160327  0.08326717  0.04320722 -0.02768004 -0.15409064  0.01975485
  0.00424004 -0.01331305  0.07695454  0.00423175  0.01493363  0.00967659
 -0.01156979 -0.05728268  0.00660228  0.08559935 -0.01324683  0.19247533
  0.03093197 -0.01674099 -0.08108932 -0.03761844  0.0224206  -0.10711695
 -0.05666951 -0.03250587  0.00342538  0.01800998 -0.03139426 -0.02662337
 -0.01060821  0.1303686  -0.04004588 -0.08345057 -0.04397463 -0.00238762
  0.03230535 -0.03432336  0.04640307  0.04467803  0.02778899 -0.05200008
  0.09510734 -0.03339511 -0.05595246  0.00370586  0.10813392  0.0817124
  0.11493364  0.06823277 -0.04011867  0.05576592  0.0480451  -0.09714999
  0.01466369 -0.03826224 -0.01376018 -0.18882908  0.13071539  0.06515288
  0.04290044  0.00397485 -0.13243373 -0.0626884   0.0178644  -0.02462831
  0.12204943  0.02733451  0.00588785  0.04293474 -0.03406001 -0.03920925
 -0.037735    0.09259502 -0.07360284 -0.07531217  0.00743953  0.05104343
  0.05940958 -0.05236354 -0.02095791 -0.06291164  0.06400023 -0.02299158
 -0.02571259 -0.03194989  0.03461788  0.03548134 -0.08199508  0.04051234
  0.0359157   0.00360841  0.08500288  0.07488914 -0.10643696  0.01936029
 -0.06692757  0.054874    0.02228982  0.01287157  0.03659066 -0.02596931
 -0.00991414  0.03032041  0.06223603 -0.02136177 -0.04065766 -0.09297845
  0.00847921  0.01725903 -0.09266669  0.02959906 -0.06617723 -0.07464533
  0.08204312  0.04096371  0.0377543  -0.00047506 -0.07294132  0.00411936
 -0.04333806 -0.13596456  0.04066416 -0.0431705   0.08362208  0.06968587
  0.0626035  -0.10087311 -0.00732823  0.0027957  -0.1109819  -0.08250239
  0.00532797 -0.02663772 -0.01411508 -0.06014302 -0.04858347  0.01841987
 -0.13912058  0.00757496 -0.04209726  0.02352497 -0.12159852  0.03612845
  0.01465165  0.09723695 -0.03419185  0.06400625  0.11937954  0.09966802
  0.00053246 -0.0417685   0.03130365 -0.00138023 -0.03247584  0.05058114
 -0.04822885  0.02536839 -0.0584316   0.10057152  0.02267649 -0.02322654
  0.00459925 -0.07323758 -0.04189198  0.07136833 -0.14718637 -0.02562269
  0.02895422  0.03326563 -0.00808948 -0.00090426 -0.00790128 -0.0161758
  0.07481994 -0.03379763  0.08010591 -0.02186493  0.03943006 -0.1550946
 -0.03735766  0.03968689  0.03827081 -0.00804011  0.04253384  0.05825679
 -0.08032871 -0.02656596 -0.004345    0.0381458  -0.07428243 -0.09156831
 -0.11415806  0.09863101  0.0264245  -0.06539769 -0.09574846  0.10732315
  0.0250018  -0.04023937  0.13254084 -0.08508862 -0.05014898  0.00546258
  0.05296977  0.03926921  0.02756595 -0.06067674  0.01374627 -0.06461439
 -0.06628061 -0.02509274 -0.01343905  0.02364533  0.00415561 -0.00412977
 -0.13324179 -0.02059344 -0.07124218  0.04790132  0.10617194  0.00782885
 -0.11592745  0.09987051 -0.12096098  0.01871174 -0.06139049  0.02323469
  0.05064858  0.01160651  0.05918611 -0.15315631 -0.03111675 -0.18070242
 -0.06121875  0.01182989  0.09179611 -0.04471653  0.06328763  0.00420621
 -0.00350776  0.02008841 -0.00981681 -0.0211009   0.11605927  0.00830611
  0.02505459  0.01837029 -0.10540015 -0.05903868 -0.0415073   0.01885657
  0.00433212 -0.16591913 -0.01552654 -0.09128339 -0.01615678  0.05504527
  0.08396287 -0.0144536   0.14801073 -0.00141371 -0.09619392  0.11677925
  0.07601462  0.00655604 -0.01272317 -0.08641504  0.01715801 -0.08762102
  0.11333538  0.0633557   0.05602988 -0.00443996 -0.06110007  0.1800708
 -0.07364944  0.0935626   0.06911841 -0.05756433  0.00629352  0.04636458
 -0.08485484 -0.02830896  0.1432262   0.02405895  0.0283429  -0.05216397
 -0.0055509  -0.07429799  0.01846742  0.11680758  0.01958649 -0.07401475
 -0.08509797  0.04224469  0.02120205  0.03607013  0.03262125  0.01908744
 -0.06557089  0.06520796 -0.10795359  0.02723636 -0.06750403  0.01626187
 -0.06082578  0.09037699  0.01774344 -0.11400679 -0.05762378 -0.03155118
  0.0063971  -0.00336269 -0.10182542  0.06290603  0.02445996 -0.01673077
  0.03622671  0.00620946 -0.02119483 -0.01917192  0.10648254  0.07445333
  0.10440005  0.00070381 -0.09723849 -0.11286505  0.09685158  0.00380622
  0.00407051  0.08927596 -0.02718715  0.07284976  0.00434119 -0.10012536
  0.02470519  0.04860623 -0.05250542 -0.05010427 -0.01978075  0.05959387
 -0.0047453  -0.14435044  0.01960936 -0.05018245  0.03535672  0.01845179
  0.04594376 -0.02877977  0.04787475  0.09560591 -0.07644784 -0.01131776
  0.03671128  0.03649254 -0.03879333 -0.10406948  0.02997301  0.02070176
 -0.04281354 -0.06925386  0.06488616 -0.00770528  0.00502762  0.03739528
  0.02186677  0.05895388 -0.09282264  0.03027524 -0.00343768  0.01863729
 -0.09966516 -0.01891949  0.0774895  -0.0535956  -0.03369077  0.06342247
 -0.04593402  0.07629769 -0.1934163  -0.06573679 -0.0891825  -0.01866824
  0.04310774  0.04757277  0.04843906 -0.04306046 -0.13046756 -0.01382263
  0.04441348 -0.02731283  0.02272408  0.13913184  0.04676741 -0.01769838
 -0.06182409  0.08203245  0.08247183  0.03146705 -0.01910425 -0.01754967
 -0.000911    0.06958635 -0.0794974   0.0648194  -0.04043015  0.04329812
 -0.09235388 -0.20047297 -0.01761635 -0.03172015  0.05760843  0.03848357
  0.01422985 -0.02058335  0.02173683 -0.0803907  -0.02406649 -0.01445011
 -0.1090102   0.04720022  0.02848673 -0.08891391  0.03627446  0.01864952
  0.05677791  0.01155838 -0.00200318 -0.01500606 -0.04273034 -0.06995729
 -0.02223687 -0.02081909] -0.04839489236474037*/
    static const fixed weights5[] ={
    -12569, -7782, -6329, -6721, 5405, 6463, 5580, -5496, -8540, -16077, 760, 1055, 
    -5707, 11288, -4556, -2818, -15361, -11141, -9993, -1850, 492, 8181, -5667, -5512, 
    -1515, -13951, 3036, -4495, 8836, 12319, -3968, -18965, -2530, 1761, -2465, -6927, 
    -965, 5290, 770, -4143, 4211, -12525, 5298, 2282, 6895, -1802, -8040, -850, 11021, 
    -13130, 6232, -8038, -5531, -3060, 9978, 8504, 2023, -3023, -8963, 10790, -17070, 
    -3523, 137, -8970, -1414, -15164, -4521, 6751, -8688, 1179, -8274, 13842, -1552, 
    -4444, 9800, -6887, -8211, -2861, 2832, 10914, 5663, -3628, -20197, 2589, 556, -1745, 
    10087, 555, 1957, 1268, -1516, -7508, 865, 11220, -1736, 25228, 4054, -2194, -10629, 
    -4931, 2939, -14040, -7428, -4261, 449, 2361, -4115, -3490, -1390, 17088, -5249, 
    -10938, -5764, -313, 4234, -4499, 6082, 5856, 3642, -6816, 12466, -4377, -7334, 
    486, 14173, 10710, 15065, 8943, -5258, 7309, 6297, -12734, 1922, -5015, -1804, -24750, 
    17133, 8540, 5623, 521, -17358, -8217, 2342, -3228, 15997, 3583, 772, 5628, -4464, 
    -5139, -4946, 12137, -9647, -9871, 975, 6690, 7787, -6863, -2747, -8246, 8389, -3014, 
    -3370, -4188, 4537, 4651, -10747, 5310, 4708, 473, 11141, 9816, -13951, 2538, -8772, 
    7192, 2922, 1687, 4796, -3404, -1299, 3974, 8157, -2800, -5329, -12187, 1111, 2262, 
    -12146, 3880, -8674, -9784, 10754, 5369, 4949, -62, -9561, 540, -5680, -17821, 5330, 
    -5658, 10961, 9134, 8206, -13222, -961, 366, -14547, -10814, 698, -3491, -1850, 
    -7883, -6368, 2414, -18235, 993, -5518, 3083, -15938, 4735, 1920, 12745, -4482, 
    8389, 15647, 13064, 70, -5475, 4103, -181, -4257, 6630, -6321, 3325, -7659, 13182, 
    2972, -3044, 603, -9599, -5491, 9354, -19292, -3358, 3795, 4360, -1060, -119, -1036, 
    -2120, 9807, -4430, 10500, -2866, 5168, -20329, -4897, 5202, 5016, -1054, 5575, 
    7636, -10529, -3482, -570, 5000, -9736, -12002, -14963, 12928, 3464, -8572, -12550, 
    14067, 3277, -5274, 17372, -11153, -6573, 716, 6943, 5147, 3613, -7953, 1802, -8469, 
    -8688, -3289, -1761, 3099, 545, -541, -17464, -2699, -9338, 6279, 13916, 1026, -15195, 
    13090, -15855, 2453, -8047, 3045, 6639, 1521, 7758, -20075, -4079, -23685, -8024, 
    1551, 12032, -5861, 8295, 551, -460, 2633, -1287, -2766, 15212, 1089, 3284, 2408, 
    -13815, -7738, -5440, 2472, 568, -21747, -2035, -11965, -2118, 7215, 11005, -1894, 
    19400, -185, -12608, 15306, 9963, 859, -1668, -11327, 2249, -11485, 14855, 8304, 
    7344, -582, -8009, 23602, -9653, 12263, 9059, -7545, 825, 6077, -11122, -3711, 18773, 
    3153, 3715, -6837, -728, -9738, 2421, 15310, 2567, -9701, -11154, 5537, 2779, 4728, 
    4276, 2502, -8595, 8547, -14150, 3570, -8848, 2131, -7973, 11846, 2326, -14943, 
    -7553, -4135, 838, -441, -13346, 8245, 3206, -2193, 4748, 814, -2778, -2513, 13957, 
    9759, 13684, 92, -12745, -14793, 12695, 499, 534, 11702, -3563, 9549, 569, -13124, 
    3238, 6371, -6882, -6567, -2593, 7811, -622, -18920, 2570, -6578, 4634, 2419, 6022, 
    -3772, 6275, 12531, -10020, -1483, 4812, 4783, -5085, -13641, 3929, 2713, -5612, 
    -9077, 8505, -1010, 659, 4901, 2866, 7727, -12166, 3968, -451, 2443, -13063, -2480, 
    10157, -7025, -4416, 8313, -6021, 10000, -25351, -8616, -11689, -2447, 5650, 6235, 
    6349, -5644, -17101, -1812, 5821, -3580, 2978, 18236, 6130, -2320, -8103, 10752, 
    10810, 4124, -2504, -2300, -119, 9121, -10420, 8496, -5299, 5675, -12105, -26276, 
    -2309, -4158, 7551, 5044, 1865, -2698, 2849, -10537, -3154, -1894, -14288, 6187, 
    3734, -11654, 4755, 2444, 7442, 1515, -263, -1967, -5601, -9169, -2915, -2729
    };
    
    static const neuron_t neuron5 = {weights5, -6343  };
    neurons[5]=neuron5;

    /* [ 1.59834102e-02 -9.74969789e-02  5.05900197e-03  1.25658646e-01
  2.08582561e-02  8.76727402e-02 -7.55646080e-02 -1.79783581e-03
  8.22919309e-02  8.22827443e-02  1.43389031e-01  3.58933955e-03
 -1.26403630e-01 -4.37026806e-02  4.81917560e-02 -1.29629180e-01
 -8.30682069e-02  4.89961728e-02 -4.14083377e-02 -2.56397687e-02
 -8.40480551e-02  3.69910486e-02 -3.94206643e-02  3.87855321e-02
  9.12908763e-02  1.33283630e-01 -7.94292428e-03 -2.24970300e-02
  5.94324060e-02  6.77810162e-02  7.45778829e-02  1.19124942e-01
 -1.13505088e-01 -6.31630700e-03 -9.77729708e-02 -1.00881375e-01
  3.11408192e-02 -1.29351035e-01  8.07327852e-02 -9.27983820e-02
 -1.55260593e-01  8.85366425e-02  2.20038239e-02  4.93280403e-02
  9.14846137e-02  4.64581735e-02  2.82820165e-02  1.38101786e-01
  2.40048356e-02  2.39672586e-02 -1.25261722e-02  5.69629343e-03
  2.73496658e-02 -5.56704169e-03 -1.14058055e-01 -4.90841046e-02
  6.60518743e-03 -1.20912783e-01  1.15167405e-02  1.31717622e-01
 -7.02511519e-02 -8.79555568e-02  1.04467466e-01 -7.14798272e-02
  6.71469867e-02  2.78898478e-02  8.78923666e-03  4.93955575e-02
  1.20024763e-01  3.23863875e-04 -1.23032615e-01 -1.89405344e-02
 -1.03703164e-01 -1.47325203e-01  4.54855599e-02 -6.47526458e-02
  5.77100478e-02 -1.77844875e-02 -8.41294378e-02 -2.31038220e-03
  4.95634601e-02  2.41629817e-02 -8.69348422e-02  1.51522774e-02
 -1.27916291e-01  5.66815026e-02 -1.60091612e-02  9.78194363e-03
  1.00913785e-01 -1.55241191e-01 -7.33147934e-02  5.50853908e-02
  4.07664217e-02 -9.61281359e-03 -2.62074312e-03  4.19357829e-02
 -9.40506011e-02  4.66194302e-02 -2.84651536e-02 -1.67041942e-01
  7.89914131e-02 -7.85825998e-02  8.92903730e-02 -9.65513755e-06
 -6.69471249e-02 -6.87487200e-02  1.22378524e-02 -1.83030784e-01
 -6.48297668e-02 -1.23361133e-01 -4.62822281e-02 -3.64469327e-02
 -1.16994031e-01 -8.25275108e-02  6.42280877e-02 -7.62080252e-02
 -3.14764827e-02  1.20933041e-01 -8.33311081e-02  4.43261154e-02
 -5.20201623e-02 -5.48249930e-02 -7.53568113e-03  1.77093357e-01
  1.43690228e-01  5.95909469e-02  1.23419076e-01 -1.22118130e-01
  4.89255078e-02  3.15494761e-02  4.18062285e-02 -1.96454860e-02
  3.90524901e-02 -7.59962648e-02  3.38497274e-02 -8.94085467e-02
 -8.24941173e-02 -2.88149118e-02 -4.27845195e-02  6.83809519e-02
 -5.45188040e-02 -1.34723395e-01  9.65926945e-02 -6.24167398e-02
  7.77197257e-02  6.13190271e-02  1.61537509e-02  9.93839428e-02
  2.33495701e-02 -1.72506198e-01  2.05721874e-02  3.10215633e-02
  3.08999028e-02 -4.06945273e-02 -6.55942708e-02  2.74472144e-02
  6.06281906e-02  4.90659103e-02 -1.27888665e-01 -1.41715601e-01
  2.20301561e-02  7.49706179e-02 -1.38691396e-01  2.00332999e-02
 -1.05311275e-02 -8.16589221e-02 -5.82144558e-02  1.78361144e-02
  1.43783558e-02  1.29332349e-01 -5.63307256e-02  7.85856619e-02
 -4.23361845e-02 -9.25333649e-02  2.54554972e-02  2.75812577e-02
  1.39393769e-02 -6.74055368e-02 -4.87677902e-02  3.28720883e-02
 -1.48825143e-02  7.95565248e-02 -7.61244772e-03 -6.03449158e-03
 -4.71187457e-02  1.02908909e-01  9.13255587e-02  2.40002982e-02
 -9.98465437e-03  3.58194299e-03 -1.23054899e-01  1.20892853e-01
 -1.25693902e-01 -1.33807942e-01 -6.03822507e-02  5.67714199e-02
  7.39644393e-02  8.47381279e-02 -7.78955668e-02 -1.91624388e-02
 -7.98039734e-02  6.89241067e-02  9.68452245e-02 -9.03706625e-02
  3.18168364e-02  2.23021600e-02 -2.72625051e-02 -2.64612753e-02
 -1.31497845e-01  1.08101584e-01  1.30192369e-01  2.82076234e-03
 -2.63178302e-03  8.15105364e-02  8.25510547e-02 -5.29142749e-03
  1.12535574e-01 -2.81817541e-02 -3.49769890e-02 -3.26910727e-02
  2.70117167e-02  4.43491824e-02  4.54119183e-02  7.71447958e-04
 -1.40942670e-02  1.89439263e-02  1.63696632e-01  6.12091087e-02
  2.07677018e-02 -1.34734977e-02  1.44837677e-01  1.98927850e-01
 -1.62066929e-02  3.69708762e-02  2.60767881e-02  8.42408389e-02
 -3.34022357e-03  5.41581511e-02  2.35788152e-02 -8.20309743e-02
 -9.25677642e-02  4.19023186e-02 -1.60536151e-02 -3.85111868e-02
  6.58460334e-02 -7.82519504e-02  5.89932641e-03 -9.23231617e-02
 -6.56267479e-02  1.09650351e-01 -1.10215701e-01 -9.57533047e-02
  1.36916889e-02  1.66777652e-02 -1.50445551e-02 -3.41635607e-02
  2.12476458e-02  2.25050859e-02 -2.85327397e-02  8.83175582e-02
  1.19496293e-01 -3.73421311e-02 -7.98694864e-02 -2.06784159e-02
  1.03082128e-01 -1.14006966e-01  5.91514586e-03  5.85746504e-02
 -4.17917520e-02 -1.75743401e-02  3.85609455e-02  1.35677180e-03
  3.40963937e-02 -1.30912811e-02 -2.02654094e-01 -2.59258319e-02
  6.85812756e-02 -9.81955603e-02  1.13200121e-01 -8.15896764e-02
 -4.00750153e-03  1.16750024e-01  8.11571553e-02 -1.47250891e-01
 -1.53827175e-01 -3.16733383e-02 -1.13480492e-02 -6.54353481e-03
 -8.56622308e-02  5.64667508e-02  8.64918828e-02 -8.02172571e-02
 -1.78302508e-02 -7.47550949e-02  2.48817448e-02 -9.26159471e-02
 -6.65160939e-02 -3.22885774e-02  6.38244525e-02 -3.68061624e-02
 -9.55494773e-03  7.62791261e-02  1.42779380e-01 -9.87226591e-02
  5.03938925e-03  1.08854704e-01  1.16978073e-02  1.29263222e-01
 -1.16665527e-01  7.68156946e-02 -1.91469789e-02 -7.60378391e-02
 -1.28230616e-03 -6.40284223e-03 -5.07997312e-02  1.29352689e-01
  3.14336233e-02  4.86796685e-02 -5.81413917e-02 -2.77046557e-03
 -8.92187003e-03  9.93227214e-02  1.39948249e-01  4.71312627e-02
  1.39421612e-01  1.55791454e-02 -1.59645583e-02 -1.14635482e-01
 -6.93834759e-03 -2.04508826e-02 -7.94031285e-03 -6.49623647e-02
 -3.56966220e-02  1.13478988e-01  1.49672823e-02  4.47543450e-02
  4.55727018e-02 -5.59232850e-03 -8.12432244e-02 -1.68277938e-02
  5.07512875e-02  2.18345951e-02  5.48607297e-02  1.37669936e-01
 -1.16603710e-02 -1.31195322e-01 -8.32001120e-03 -1.88314766e-02
 -9.28920507e-02 -3.39419618e-02 -4.97509539e-02  3.42749394e-02
  1.15087200e-02 -4.68900464e-02  1.72255170e-02  4.91062365e-02
 -1.26294956e-01 -1.16119221e-01 -7.05457181e-02 -7.69791603e-02
  1.21606849e-01  9.44424644e-02 -3.76481749e-02  4.46146470e-04
  6.88685775e-02 -5.40579334e-02  1.94945950e-02  4.68666255e-02
  4.89979759e-02 -5.50421737e-02  1.40193850e-01  2.06687022e-02
 -5.05806170e-02  2.64015198e-02 -2.70751324e-02  4.93393168e-02
  2.79619023e-02  1.24273092e-01  4.10322733e-02 -2.42503788e-02
  5.36355190e-02  5.82491904e-02 -1.86169352e-02 -4.97758053e-02
  3.82305495e-02  8.83949995e-02 -1.09990299e-01 -1.27872527e-01
  1.62328202e-02 -6.95710108e-02 -1.12109609e-01  2.45343279e-02
 -3.32172103e-02  4.46308851e-02  4.53436635e-02  9.46803167e-02
  2.04508379e-01  6.59960657e-02 -8.55457559e-02  1.36159092e-01
  3.97697426e-02 -1.51332289e-01 -8.84179547e-02 -1.42295673e-01
  4.85447422e-02 -1.02489166e-01  2.12864280e-02 -2.15920809e-04
  6.07473254e-02  7.12642372e-02  1.20498016e-01  2.35236920e-02
  4.65341397e-02 -1.15860561e-02  3.09949555e-02  9.87084806e-02
  2.01050714e-02 -8.71415138e-02 -2.68112905e-02  4.85606231e-02
  4.23255041e-02 -5.65125681e-02  6.96241334e-02 -5.04110800e-03
 -3.10991108e-02  2.50603631e-02 -5.57153895e-02  5.84541447e-02
  5.01840264e-02  8.75021294e-02 -8.79337490e-02  4.74691354e-02
 -1.06380023e-02  1.49722576e-01 -5.48785031e-02 -3.15447822e-02
 -3.05182114e-02  4.82333899e-02  9.63184163e-02  1.41192734e-01
  9.24918577e-02 -4.97339852e-02  3.59668657e-02  2.27814168e-01
 -9.98633131e-02 -7.85902366e-02  8.73009861e-02 -6.10231347e-02
 -6.53909286e-03  1.43395523e-02 -2.08989345e-03 -1.69630498e-02
  1.10784720e-04 -2.85806414e-02  1.77995339e-02 -7.99400657e-02
  2.52462714e-03 -1.01680629e-01  1.09150046e-02  1.05881738e-02
 -6.96353242e-02  9.12718847e-02 -2.94062048e-02  1.75213903e-01
  1.12140104e-01 -1.36947423e-01  1.30412161e-01 -1.57606512e-01
  1.53791690e-02  7.31830671e-02 -4.06322302e-03  2.67778840e-02
  1.83950961e-01 -1.46974847e-01 -1.49729490e-01 -1.11997202e-01
  4.79002930e-02  6.41639084e-02  3.88073958e-02  1.19493064e-02
 -1.62487879e-01 -1.62897054e-02 -5.41357696e-02  5.60343564e-02
  8.69663656e-02 -7.64248520e-02 -3.45940329e-02 -9.92913619e-02
  2.75891572e-02 -2.05682553e-02  7.39582777e-02 -2.18994580e-02
  2.32577566e-02  3.31335217e-02  3.63941267e-02 -6.29819185e-02
  7.68783018e-02  3.84464636e-02 -7.58519694e-02  5.88595569e-02
 -9.67094526e-02  9.37381461e-02  7.81407803e-02 -3.94679978e-02
  2.63018347e-02 -1.23166777e-01  6.20565563e-02 -2.05049142e-02
 -1.58963129e-01  7.94543549e-02  3.99686815e-03  3.57174613e-02] 0.22412531077861786*/
    static const fixed weights6[] ={
    2095, -12779, 663, 16470, 2734, 11491, -9904, -236, 10786, 10785, 18794, 470, -16568, 
    -5728, 6317, -16991, -10888, 6422, -5427, -3361, -11016, 4848, -5167, 5084, 11966, 
    17470, -1041, -2949, 7790, 8884, 9775, 15614, -14877, -828, -12815, -13223, 4082, 
    -16954, 10582, -12163, -20350, 11605, 2884, 6466, 11991, 6089, 3707, 18101, 3146, 
    3141, -1642, 747, 3585, -730, -14950, -6434, 866, -15848, 1510, 17264, -9208, -11529, 
    13693, -9369, 8801, 3656, 1152, 6474, 15732, 42, -16126, -2483, -13593, -19310, 
    5962, -8487, 7564, -2331, -11027, -303, 6496, 3167, -11395, 1986, -16766, 7429, 
    -2098, 1282, 13227, -20348, -9610, 7220, 5343, -1260, -344, 5497, -12327, 6111, 
    -3731, -21895, 10354, -10300, 11703, -1, -8775, -9011, 1604, -23990, -8497, -16169, 
    -6066, -4777, -15335, -10817, 8419, -9989, -4126, 15851, -10922, 5810, -6818, -7186, 
    -988, 23212, 18834, 7811, 16177, -16006, 6413, 4135, 5480, -2575, 5119, -9961, 4437, 
    -11719, -10813, -3777, -5608, 8963, -7146, -17658, 12661, -8181, 10187, 8037, 2117, 
    13026, 3060, -22611, 2696, 4066, 4050, -5334, -8598, 3598, 7947, 6431, -16763, -18575, 
    2888, 9827, -18179, 2626, -1380, -10703, -7630, 2338, 1885, 16952, -7383, 10300, 
    -5549, -12129, 3337, 3615, 1827, -8835, -6392, 4309, -1951, 10428, -998, -791, -6176, 
    13488, 11970, 3146, -1309, 469, -16129, 15846, -16475, -17538, -7914, 7441, 9695, 
    11107, -10210, -2512, -10460, 9034, 12694, -11845, 4170, 2923, -3573, -3468, -17236, 
    14169, 17065, 370, -345, 10684, 10820, -694, 14750, -3694, -4585, -4285, 3540, 5813, 
    5952, 101, -1847, 2483, 21456, 8023, 2722, -1766, 18984, 26074, -2124, 4846, 3418, 
    11042, -438, 7099, 3091, -10752, -12133, 5492, -2104, -5048, 8631, -10257, 773, 
    -12101, -8602, 14372, -14446, -12551, 1795, 2186, -1972, -4478, 2785, 2950, -3740, 
    11576, 15663, -4895, -10469, -2710, 13511, -14943, 775, 7677, -5478, -2304, 5054, 
    178, 4469, -1716, -26562, -3398, 8989, -12871, 14837, -10694, -525, 15303, 10637, 
    -19300, -20162, -4151, -1487, -858, -11228, 7401, 11337, -10514, -2337, -9798, 3261, 
    -12139, -8718, -4232, 8366, -4824, -1252, 9998, 18714, -12940, 661, 14268, 1533, 
    16943, -15292, 10068, -2510, -9966, -168, -839, -6658, 16955, 4120, 6381, -7621, 
    -363, -1169, 13018, 18343, 6178, 18274, 2042, -2093, -15026, -909, -2681, -1041, 
    -8515, -4679, 14874, 1962, 5866, 5973, -733, -10649, -2206, 6652, 2862, 7191, 18045, 
    -1528, -17196, -1091, -2468, -12176, -4449, -6521, 4492, 1508, -6146, 2258, 6436, 
    -16554, -15220, -9247, -10090, 15939, 12379, -4935, 58, 9027, -7085, 2555, 6143, 
    6422, -7214, 18375, 2709, -6630, 3460, -3549, 6467, 3665, 16289, 5378, -3179, 7030, 
    7635, -2440, -6524, 5011, 11586, -14417, -16761, 2128, -9119, -14694, 3216, -4354, 
    5850, 5943, 12410, 26805, 8650, -11213, 17847, 5213, -19835, -11589, -18651, 6363, 
    -13433, 2790, -28, 7962, 9341, 15794, 3083, 6099, -1519, 4063, 12938, 2635, -11422, 
    -3514, 6365, 5548, -7407, 9126, -661, -4076, 3285, -7303, 7662, 6578, 11469, -11526, 
    6222, -1394, 19624, -7193, -4135, -4000, 6322, 12625, 18506, 12123, -6519, 4714, 
    29860, -13089, -10301, 11443, -7998, -857, 1880, -274, -2223, 15, -3746, 2333, -10478, 
    331, -13327, 1431, 1388, -9127, 11963, -3854, 22966, 14698, -17950, 17093, -20658, 
    2016, 9592, -533, 3510, 24111, -19264, -19625, -14680, 6278, 8410, 5087, 1566, -21298, 
    -2135, -7096, 7345, 11399, -10017, -4534, -13014, 3616, -2696, 9694, -2870, 3048, 
    4343, 4770, -8255, 10077, 5039, -9942, 7715, -12676, 12286, 10242, -5173, 3447, 
    -16144, 8134, -2688, -20836, 10414, 524, 4682
    };
    
    static const neuron_t neuron6 = {weights6, 29377  };
    neurons[6]=neuron6;

    /* [ 6.53651133e-02  3.23382579e-02  9.12770722e-03 -3.12591419e-02
 -2.05284599e-02 -1.34829879e-01  3.63206193e-02 -8.75635073e-02
 -5.39998412e-02 -6.55087978e-02 -2.33342848e-03 -6.78263232e-02
 -1.22266464e-01  1.06142588e-01  7.42332563e-02  1.08211912e-01
  7.65699297e-02  9.28291082e-02  4.33510207e-02  6.16719574e-02
  1.65574439e-02  6.73910007e-02  3.46259288e-02  1.32365506e-02
 -9.18747857e-02 -1.32011339e-01 -4.68409015e-03  1.93257127e-02
 -3.47912125e-02 -5.88151105e-02 -4.30716798e-02  1.26386955e-01
 -4.55849208e-02 -3.73779656e-03 -4.28773947e-02  8.35695714e-02
 -1.06033579e-01  6.59870803e-02  3.80622782e-02  1.28989862e-02
  1.82000056e-01 -1.01398543e-01  7.19747916e-02  8.65010396e-02
 -2.55491100e-02  1.24459798e-02 -1.66186735e-01 -4.03599031e-02
  8.60338733e-02 -5.90404943e-02 -2.78254915e-02  6.37329742e-02
 -4.89780121e-02 -6.84961975e-02 -3.16846706e-02  1.08045295e-01
 -5.17669134e-02 -3.96023951e-02 -1.03958927e-01  4.31716219e-02
 -1.53696224e-01  1.04491100e-01  6.08838946e-02  1.02168895e-01
  1.52816074e-02 -1.01780213e-01 -1.66883040e-02 -6.31784424e-02
  1.09571740e-01  9.39543471e-02 -7.37453997e-02  1.02217495e-01
  6.29032357e-03 -1.52246896e-02 -1.00611802e-02  8.35973397e-03
  1.37478029e-02 -6.00898527e-02  4.68522348e-02  1.00058116e-01
 -1.53096735e-01 -3.52774523e-02 -4.80199195e-02 -4.97328267e-02
 -6.93660006e-02 -5.15463054e-02  7.44598582e-02 -3.83291394e-02
  3.24496068e-02  6.18624389e-02  1.73339099e-01  6.21101353e-03
 -1.10521033e-01  1.25283180e-02 -2.61079893e-02  1.49848133e-01
  7.95215741e-03  1.04662798e-01  4.15110402e-02  1.95908602e-02
 -9.96018294e-03  5.43916151e-02 -1.54300723e-02 -7.73510560e-02
  6.68418705e-02  7.37971067e-02  3.16869328e-03  8.73452574e-02
  6.09521195e-02  5.90126663e-02 -3.64085585e-02  9.06605157e-04
 -7.62440339e-02  1.15585551e-01 -9.36786160e-02  5.13236150e-02
  1.54434830e-01  6.85162423e-03 -8.01437870e-02  1.04178101e-01
  1.13515690e-01  1.16316460e-01 -9.65815336e-02 -7.69015998e-02
  6.29838929e-02 -5.55851981e-02  1.47749901e-01  1.83593214e-01
 -2.80793831e-02 -2.98992116e-02 -7.55632203e-03 -1.65406298e-02
  1.42527133e-01  9.38797072e-02  6.22014748e-03  2.16304921e-02
 -6.92651374e-03 -1.01433612e-01  2.82262396e-02  1.35055017e-02
 -6.52394369e-02 -9.10218339e-04  1.13035426e-01 -9.85345468e-02
  6.63589165e-02  1.95853151e-02  7.22577795e-02 -7.87090585e-02
 -1.07550614e-01 -1.21705867e-01  6.46027476e-02 -2.36639418e-02
 -9.42689851e-02 -7.30785280e-02  3.55141014e-02 -1.98845956e-02
 -1.75598555e-03 -3.66060920e-02 -1.05093867e-01  1.74696118e-01
  3.85180376e-02  3.98855563e-03  1.77577231e-02 -1.36477008e-01
  3.94578986e-02 -9.71513055e-03 -3.08409333e-02  1.43109551e-02
  8.70217010e-02  1.26678184e-01 -2.00488009e-02 -5.04851192e-02
 -1.02409720e-02 -2.12261491e-02 -3.72582749e-02 -2.79124565e-02
 -1.16340883e-01  9.83350631e-03 -1.89394280e-01 -8.69942904e-02
 -2.24222988e-02  1.02267526e-01  5.29754721e-02 -7.03812093e-02
 -3.31075676e-02  1.92998387e-02  1.50456821e-04  4.71341200e-02
 -5.57478182e-02  1.17537826e-01  2.74956264e-02  7.35342503e-02
  1.73948910e-02 -4.61375676e-02 -1.59989186e-02  1.25120252e-01
 -1.61088184e-01 -1.16909601e-01  8.56704172e-03 -8.71714503e-02
 -5.03365435e-02 -1.15078807e-01  1.26050651e-01 -4.66425307e-02
  2.45666895e-02 -8.73911083e-02  2.31873561e-02 -2.25352421e-02
  3.24559398e-02 -6.21858984e-03 -7.87479952e-02 -5.41635752e-02
 -3.84132825e-02  1.16888344e-01 -6.25559464e-02 -1.94007214e-02
 -4.42838445e-02  1.21900029e-01 -7.35463798e-02 -2.08751187e-02
 -9.31646526e-02  2.51307362e-03 -4.50172611e-02  4.85527255e-02
  6.87694969e-03  5.74544780e-02 -3.88013758e-02  1.05420172e-01
 -1.21416435e-01  1.51751405e-02  8.79854560e-02 -1.06935557e-02
  1.62743121e-01 -5.75385354e-02 -8.14587325e-02 -2.59153787e-02
 -6.10780157e-02 -1.07817158e-01  1.10328533e-01  2.30807997e-02
 -9.34015363e-02  8.84865820e-02 -6.22146204e-02  6.42906055e-02
  1.17828893e-02 -3.97169143e-02  6.29309639e-02  1.46112785e-01
  1.23831704e-02 -7.90540129e-02  1.55147528e-02 -1.99081555e-01
  1.12237260e-01  6.38771057e-03  7.67138228e-02  2.11769287e-02
  4.43413481e-02 -3.89307067e-02 -1.11403912e-01 -5.76715097e-02
  4.87879217e-02  3.37882154e-02  7.92590007e-02 -2.38264818e-02
 -1.80468455e-01  8.31390321e-02 -1.18316308e-01  1.40644079e-02
  1.38984784e-01 -1.47092706e-02 -4.87387329e-02 -4.51928638e-02
  4.20771055e-02 -5.21351434e-02 -2.53592804e-02 -8.47377330e-02
 -1.01447366e-02  9.63867605e-02 -1.57616828e-02 -6.87519163e-02
 -8.45594183e-02 -1.55496295e-03 -1.10319495e-01  2.57089175e-03
  9.82354730e-02  9.80337430e-03  4.14996892e-02 -4.04364690e-02
  3.99806947e-02  6.23942316e-02  7.58068636e-02  2.10234169e-02
 -2.13613123e-01 -1.63101684e-03  3.20460647e-02 -7.93648697e-03
 -7.99901336e-02 -8.86191204e-02 -8.66938289e-03 -4.52485345e-02
 -5.43520041e-02 -4.12037633e-02 -4.29249592e-02 -6.75143301e-02
 -1.49033517e-01 -8.53587966e-03 -9.33025628e-02 -3.73122618e-02
  1.94339659e-02  5.42350952e-03 -9.11210012e-03  3.08612548e-02
 -2.92798672e-02  3.36472318e-02  1.04304992e-01 -5.98472580e-02
 -9.20403600e-02 -2.63111237e-02 -3.84243838e-02  2.87017711e-02
  3.66125740e-02  5.14808148e-02 -4.94347252e-02  5.76828457e-02
  1.10526748e-01 -1.40649825e-01  1.10672802e-01 -1.26294836e-01
  1.91111505e-01  6.31615566e-03 -1.60609651e-02 -2.11661216e-02
 -3.03933416e-02 -7.35369474e-02 -3.57281603e-02 -3.09800692e-02
 -4.78313342e-02 -1.65302418e-02  4.87099364e-02  1.54311374e-01
 -3.94891165e-02  1.18663862e-01 -6.04091138e-02  4.38222177e-02
  1.68076009e-02 -9.91558135e-02 -4.06656004e-02 -3.67272459e-02
  6.85944557e-02 -5.03002070e-02  7.15945214e-02  6.98152035e-02
  4.81491573e-02  1.62961811e-01  7.48650730e-02 -3.51940282e-02
  1.10529244e-01  1.35922404e-02 -7.46810064e-02 -2.78267451e-02
 -4.86416891e-02 -1.59839585e-01  6.18973337e-02  3.24521400e-02
 -2.00395554e-01  2.62733568e-02 -2.06521191e-02  1.21395022e-01
 -1.19253971e-01 -3.07997074e-02 -1.09052542e-03 -1.04424125e-02
 -6.70396164e-02 -6.83619305e-02  2.62974910e-02  5.48607064e-03
  5.04454598e-02 -4.53230627e-02 -2.87159309e-02 -7.71241710e-02
 -2.03136336e-02  1.84919730e-01 -7.51096904e-02 -2.38855369e-02
  2.23192964e-02 -3.10490141e-03  3.54643241e-02 -4.38134409e-02
  3.55078862e-03  2.42919065e-02  4.10712175e-02  6.50636181e-02
  1.13008693e-01  2.86459904e-02  1.22935712e-01  4.06288877e-02
 -7.62147009e-02 -9.08692777e-02  5.47159426e-02  8.61944407e-02
  2.90544070e-02  2.75789406e-02 -4.14161347e-02  7.28954449e-02
 -4.27088700e-02  1.16618916e-01 -3.48795205e-02 -1.24697499e-01
 -3.01205665e-02  1.68551028e-01  9.80794653e-02 -8.78244266e-02
 -5.81959896e-02 -4.30092886e-02  4.60270010e-02 -3.85555290e-02
  6.94543123e-02  4.65309843e-02  1.48747563e-01 -1.83705203e-02
  1.44973278e-01  3.42527516e-02 -7.24123865e-02 -7.08654076e-02
  6.80217817e-02 -6.03604130e-02  3.99514847e-02 -1.26193061e-01
 -1.46731199e-03 -6.61848625e-03  1.09595634e-01 -1.05968013e-01
 -2.72865389e-02  1.25323802e-01  1.09282322e-01 -1.35057569e-01
  3.03102396e-02  1.65455103e-01  5.23108989e-02 -6.85145631e-02
 -4.35515791e-02  4.10081744e-02 -1.11619987e-01  3.60189527e-02
 -9.93473008e-02 -8.85912478e-02  3.19817215e-02  4.27192897e-02
  6.86921924e-02 -9.50041860e-02 -4.00058962e-02  9.08816382e-02
  6.54507577e-02  7.42387995e-02 -6.66577816e-02  1.67536247e-03
 -1.83319926e-01  6.97301701e-02 -7.14337006e-02 -2.10500043e-02
 -1.17744608e-02 -7.25112557e-02  2.22171545e-02  2.12893658e-03
  1.61696076e-01 -1.22730680e-01 -5.40773422e-02  1.91965681e-02
  7.47401118e-02 -7.70266056e-02  3.21892947e-02  2.49445031e-04
 -6.17437512e-02  1.00795686e-01  4.12474535e-02  1.19015537e-01
 -2.45584957e-02  4.43248823e-02 -2.41068490e-02  1.18081681e-02
  7.95577690e-02  1.55865818e-01 -2.91589629e-02  9.90392342e-02
  4.37080860e-02  7.32988864e-02 -6.33364543e-02 -3.50159290e-03
  2.65116543e-02  6.80180714e-02  5.00694625e-02 -3.21754552e-02
 -1.00283585e-01  5.79758128e-03  1.91315729e-02 -4.94525302e-03
 -2.67199278e-02 -3.26586589e-02  1.83798254e-01 -4.57084589e-02
  4.89911102e-02 -3.74229206e-03  1.03198718e-02 -9.46173444e-03
  3.38584967e-02  1.97567772e-02 -2.82750987e-02 -5.79103455e-02
 -1.95093174e-02 -7.17440024e-02 -4.84532788e-02  2.04686411e-02] -0.32285812497138977*/
    static const fixed weights7[] ={
    8568, 4239, 1196, -4097, -2691, -17672, 4761, -11477, -7078, -8586, -306, -8890, 
    -16026, 13912, 9730, 14184, 10036, 12167, 5682, 8083, 2170, 8833, 4538, 1735, -12042, 
    -17303, -614, 2533, -4560, -7709, -5645, 16566, -5975, -490, -5620, 10954, -13898, 
    8649, 4989, 1691, 23855, -13291, 9434, 11338, -3349, 1631, -21782, -5290, 11277, 
    -7739, -3647, 8354, -6420, -8978, -4153, 14162, -6785, -5191, -13626, 5659, -20145, 
    13696, 7980, 13391, 2003, -13341, -2187, -8281, 14362, 12315, -9666, 13398, 824, 
    -1996, -1319, 1096, 1802, -7876, 6141, 13115, -20067, -4624, -6294, -6519, -9092, 
    -6756, 9760, -5024, 4253, 8108, 22720, 814, -14486, 1642, -3422, 19641, 1042, 13718, 
    5441, 2568, -1306, 7129, -2022, -10139, 8761, 9673, 415, 11449, 7989, 7735, -4772, 
    119, -9993, 15150, -12279, 6727, 20242, 898, -10505, 13655, 14879, 15246, -12659, 
    -10080, 8255, -7286, 19366, 24064, -3680, -3919, -990, -2168, 18681, 12305, 815, 
    2835, -908, -13295, 3700, 1770, -8551, -119, 14816, -12915, 8698, 2567, 9471, -10317, 
    -14097, -15952, 8468, -3102, -12356, -9579, 4655, -2606, -230, -4798, -13775, 22898, 
    5049, 523, 2328, -17888, 5172, -1273, -4042, 1876, 11406, 16604, -2628, -6617, -1342, 
    -2782, -4884, -3659, -15249, 1289, -24824, -11403, -2939, 13404, 6944, -9225, -4339, 
    2530, 20, 6178, -7307, 15406, 3604, 9638, 2280, -6047, -2097, 16400, -21114, -15324, 
    1123, -11426, -6598, -15084, 16522, -6114, 3220, -11455, 3039, -2954, 4254, -815, 
    -10322, -7099, -5035, 15321, -8199, -2543, -5804, 15978, -9640, -2736, -12211, 329, 
    -5901, 6364, 901, 7531, -5086, 13818, -15914, 1989, 11532, -1402, 21331, -7542, 
    -10677, -3397, -8006, -14132, 14461, 3025, -12242, 11598, -8155, 8427, 1544, -5206, 
    8248, 19151, 1623, -10362, 2034, -26094, 14711, 837, 10055, 2776, 5812, -5103, -14602, 
    -7559, 6395, 4429, 10389, -3123, -23654, 10897, -15508, 1843, 18217, -1928, -6388, 
    -5924, 5515, -6833, -3324, -11107, -1330, 12634, -2066, -9011, -11083, -204, -14460, 
    337, 12876, 1285, 5439, -5300, 5240, 8178, 9936, 2756, -27999, -214, 4200, -1040, 
    -10484, -11615, -1136, -5931, -7124, -5401, -5626, -8849, -19534, -1119, -12229, 
    -4891, 2547, 711, -1194, 4045, -3838, 4410, 13671, -7844, -12064, -3449, -5036, 
    3762, 4799, 6748, -6480, 7561, 14487, -18435, 14506, -16554, 25049, 828, -2105, 
    -2774, -3984, -9639, -4683, -4061, -6269, -2167, 6385, 20226, -5176, 15554, -7918, 
    5744, 2203, -12997, -5330, -4814, 8991, -6593, 9384, 9151, 6311, 21360, 9813, -4613, 
    14487, 1782, -9789, -3647, -6376, -20950, 8113, 4254, -26266, 3444, -2707, 15911, 
    -15631, -4037, -143, -1369, -8787, -8960, 3447, 719, 6612, -5941, -3764, -10109, 
    -2663, 24238, -9845, -3131, 2925, -407, 4648, -5743, 465, 3184, 5383, 8528, 14812, 
    3755, 16113, 5325, -9990, -11910, 7172, 11298, 3808, 3615, -5428, 9555, -5598, 15285, 
    -4572, -16344, -3948, 22092, 12855, -11511, -7628, -5637, 6033, -5054, 9104, 6099, 
    19497, -2408, 19002, 4490, -9491, -9288, 8916, -7912, 5237, -16540, -192, -867, 
    14365, -13889, -3577, 16426, 14324, -17702, 3973, 21687, 6856, -8980, -5708, 5375, 
    -14630, 4721, -13022, -11612, 4192, 5599, 9004, -12452, -5244, 11912, 8579, 9731, 
    -8737, 220, -24028, 9140, -9363, -2759, -1543, -9504, 2912, 279, 21194, -16087, 
    -7088, 2516, 9796, -10096, 4219, 33, -8093, 13211, 5406, 15600, -3219, 5810, -3160, 
    1548, 10428, 20430, -3822, 12981, 5729, 9607, -8302, -459, 3475, 8915, 6563, -4217, 
    -13144, 760, 2508, -648, -3502, -4281, 24091, -5991, 6421, -491, 1353, -1240, 4438, 
    2590, -3706, -7590, -2557, -9404, -6351, 2683
    };
    
    static const neuron_t neuron7 = {weights7, -42318  };
    neurons[7]=neuron7;

    /* [ 1.14292242e-02  9.46215540e-02 -8.51765648e-02  2.61332750e-01
 -2.59804595e-02 -1.63694173e-01 -7.09828213e-02 -9.90748778e-02
  1.70003821e-03  7.35247582e-02 -1.60785809e-01  5.21946661e-02
  1.10521592e-01  2.64475700e-02  1.13518760e-01  6.27863184e-02
 -8.99752229e-02  4.80283350e-02 -5.37185706e-02 -2.75259949e-02
 -6.24440387e-02 -7.48423636e-02 -4.34741341e-02  2.93288436e-02
  6.01805039e-02  1.31129414e-01  4.69454490e-02 -8.76207054e-02
 -9.67959967e-03  3.40340026e-02  9.86388978e-03 -4.56287563e-02
 -3.18338238e-02 -4.19252813e-02  3.91487144e-02  8.04813672e-03
  4.70366254e-02  5.54944854e-03  2.44155210e-02 -5.67129403e-02
  3.71006615e-02  1.18226498e-01 -9.08576250e-02 -1.91558842e-02
 -7.80015290e-02 -5.62677346e-02  7.02480972e-02  2.15997193e-02
  4.89375405e-02  5.75372279e-02  9.58856754e-03 -5.09343483e-03
  9.22606662e-02 -7.43792653e-02 -1.13857917e-01 -3.94568257e-02
  1.54399961e-01  6.41693175e-02  8.45830590e-02 -8.56464654e-02
  1.52343541e-01 -5.44170737e-02 -4.78922483e-03 -4.49555069e-02
  8.76834318e-02  5.58450446e-02  1.24124996e-02 -9.11098048e-02
 -2.33529974e-02  1.10960910e-02  1.22201197e-01 -5.56729548e-03
  1.27087921e-01  8.55979770e-02  1.06968768e-01 -1.06879182e-01
 -3.62698622e-02 -6.03333823e-02 -7.01406896e-02  4.95226569e-02
 -6.94471076e-02 -4.39189188e-02  1.51321307e-01  5.16362563e-02
 -1.39727937e-02  3.86546948e-03 -5.86851686e-02  1.51197687e-01
 -1.11204684e-01 -1.06797904e-01  1.48574740e-01  2.17474867e-02
 -5.99910831e-03 -4.48363349e-02  1.48860097e-01 -1.99170634e-02
 -4.21366096e-02 -1.33914426e-01 -9.02508125e-02  1.10192776e-01
 -2.94638239e-02  8.63127932e-02  8.00941736e-02  2.39757225e-02
  9.96555462e-02  1.27284169e-01 -1.97666511e-02 -8.17182362e-02
 -1.04601726e-01  2.48972028e-02  2.15318967e-02 -8.96974057e-02
  4.87152003e-02 -1.05250739e-01  4.66629453e-02 -5.75227253e-02
 -8.65691379e-02 -7.51374066e-02 -9.91668925e-02 -9.16197617e-03
 -4.05115709e-02  2.70463228e-02 -5.90981543e-02  5.98515496e-02
  6.55147880e-02 -2.60117557e-02 -4.95657958e-02  3.80560453e-03
 -5.18495440e-02 -9.31747109e-02 -3.96469682e-02 -8.47868621e-02
 -2.18654564e-03  1.28395036e-01 -8.96381214e-02  1.22090593e-01
 -1.38486668e-01 -3.40825990e-02 -1.84168458e-01  1.14643283e-01
  3.32726873e-02  7.75977224e-02 -1.27639011e-01 -3.56206745e-02
 -1.04228206e-01 -3.62679362e-02 -4.83934619e-02  2.38602329e-03
 -1.27454802e-01  4.61270548e-02 -2.31929519e-03 -6.55927062e-02
  1.03839427e-01  1.10971279e-01 -1.07073128e-01 -1.29754841e-01
 -2.42577586e-02  6.30945638e-02  9.62820649e-02 -7.75508210e-02
 -4.95792590e-02 -1.11862190e-01  6.55310452e-02 -5.79095446e-02
  1.59139127e-01  2.83163460e-03 -1.38189895e-02 -9.64322239e-02
  1.29166856e-01 -1.00580215e-01  8.10227096e-02  1.82721037e-02
  7.16527924e-02  1.95387006e-01  4.44879420e-02 -1.56995561e-03
 -1.35761006e-02  6.83685904e-03  2.83704847e-02 -1.09772803e-02
 -2.98879296e-02 -2.92640701e-02 -1.02138892e-01  3.43508050e-02
  9.55058262e-02 -1.17495455e-01  1.36223674e-01 -6.02400787e-02
  2.39485428e-02  4.89039458e-02 -4.48360182e-02 -1.65399656e-01
  1.51023149e-01  1.44684628e-01  4.33391146e-03  7.09907860e-02
 -4.62527312e-02  9.70185250e-02  9.47233140e-02 -9.27774701e-03
  1.86656583e-02  1.21202022e-01 -1.05664335e-01  5.80614284e-02
 -1.27823660e-02  8.15534294e-02 -4.60474975e-02  4.12345380e-02
 -1.61902100e-01 -5.07201515e-02 -6.55000657e-02 -6.47835841e-05
  4.73816954e-02 -1.36357039e-01 -1.34926826e-01  7.60101154e-02
  1.50753722e-01  3.10643390e-02  4.02423553e-02 -1.58798084e-01
  8.61889273e-02 -2.65108794e-02 -2.93004140e-02  6.05299138e-02
  6.14501685e-02 -1.04931541e-01 -1.82086110e-01  2.91981455e-02
 -1.32188216e-01 -1.05567183e-03 -1.66547634e-02  4.03861143e-02
 -5.23455590e-02 -7.99031332e-02  1.55224383e-01 -2.96682157e-02
 -5.77036245e-03  8.50108489e-02  3.50996405e-02  2.06640307e-02
 -3.98405567e-02  9.68165230e-03  1.03930987e-01 -1.61254462e-02
 -2.70319488e-02  1.11645274e-02 -5.57784252e-02 -2.89121866e-02
  3.40228039e-03  2.59911958e-02  3.80086675e-02 -6.46972284e-02
 -2.58809403e-02  5.41363470e-02 -3.34843174e-02 -9.50673372e-02
  6.38790429e-02 -5.98789528e-02 -3.32920253e-02 -1.53023545e-02
 -3.50345857e-02  9.82946008e-02 -3.46218944e-02 -2.83190981e-02
  7.22721443e-02  8.17933679e-02 -1.18285622e-02 -3.70123498e-02
 -5.58945388e-02  2.85271015e-02 -4.46614577e-03 -9.15683061e-02
 -1.26540810e-01 -4.56748493e-02  1.22984378e-02 -1.59614339e-01
 -1.44321630e-02 -5.16508073e-02 -3.00339237e-02  5.45198433e-02
  2.06370831e-01 -1.07692899e-02  1.47988915e-01  1.36647195e-01
  7.67429620e-02 -2.58031189e-02 -7.00253099e-02  3.55409011e-02
  1.52877821e-02 -6.93142489e-02 -1.11553492e-02 -4.53938618e-02
  9.98255312e-02 -1.07737795e-01  5.99381216e-02  4.97043915e-02
  3.72108631e-02  1.60535872e-01 -1.37289047e-01  1.51787400e-01
 -3.96546200e-02  2.74160299e-02 -3.88624519e-02 -1.44515140e-02
  1.07032582e-01  2.19383538e-02  9.91523117e-02 -2.05580235e-01
  4.98453192e-02  2.29339991e-02  9.95775778e-03 -3.55729945e-02
  4.20673043e-02  6.55974671e-02 -4.47074126e-04  7.94053171e-03
 -1.19936518e-01 -5.75400181e-02 -1.54671624e-01  1.20411433e-01
 -5.73946834e-02 -1.03694750e-02  1.59848947e-02  5.80689423e-02
 -8.87583941e-02  1.29918575e-01 -7.73616880e-02  7.24910898e-03
  5.99882714e-02  6.60878569e-02  3.46930884e-02  1.79158226e-01
  1.22930862e-01  2.60090213e-02  5.86658390e-03  2.44996720e-03
  5.43663129e-02  1.08738802e-01 -3.27085815e-02 -1.63273796e-01
  1.12034138e-02 -6.02269843e-02  4.36783731e-02 -1.26133054e-01
  4.93370593e-02  1.44509092e-01 -3.83983627e-02 -5.88294342e-02
 -7.58292503e-04 -1.83128286e-02 -1.05608597e-01 -4.85027097e-02
  5.43738715e-03  2.03345604e-02 -1.97136938e-03 -1.51560903e-01
 -4.04611789e-02  4.07715999e-02 -4.49725650e-02  2.03164686e-02
  1.66362926e-01  9.31017622e-02 -1.71661571e-01  1.60184324e-01
  5.59316799e-02  6.59538880e-02 -6.07409067e-02 -8.76692962e-03
  4.43301126e-02 -2.30944306e-02  2.09296662e-02  1.34350844e-02
 -7.14817941e-02  1.20751455e-01 -1.22584747e-02 -1.64075736e-02
  9.58700702e-02 -4.55711596e-02  7.90935755e-02 -1.28795821e-02
  3.15731280e-02 -2.06163019e-01  1.42132536e-01  3.46677527e-02
  3.41369025e-02  2.01060753e-02 -6.22233935e-03  2.34973263e-02
 -8.18863213e-02 -1.76900756e-02  1.42293107e-02  2.30401643e-02
 -1.26737386e-01  2.18582917e-02 -1.32475391e-01 -3.54650654e-02
 -3.12611759e-02  1.15667313e-01  2.22608298e-01 -2.11807325e-01
  1.29826307e-01 -1.88503917e-02  3.46156172e-02  1.82040304e-01
 -7.41444007e-02  9.04405266e-02 -8.25686976e-02 -5.80248721e-02
 -8.53500050e-03 -8.24081898e-02  1.01395827e-02  1.58193797e-01
  7.87232742e-02  1.50068775e-02  7.40565062e-02 -7.28526339e-02
 -5.76150268e-02 -8.73559788e-02 -1.29602253e-01 -9.65594426e-02
  5.57069629e-02  1.12527624e-01 -4.32201251e-02  6.19110875e-02
  1.70620561e-01 -6.98750541e-02  4.34034988e-02  7.48455971e-02
 -1.25472397e-01  9.72286463e-02 -1.92149577e-03  1.00206219e-01
 -5.76156043e-02  3.88324931e-02 -8.89520869e-02  8.95053297e-02
  2.76843440e-02 -1.30754799e-01 -7.19989464e-02  8.07444528e-02
  3.85621749e-02 -9.04078223e-03  1.16169468e-01 -1.85046464e-01
  6.89834030e-03  1.24984197e-01  1.53381258e-01  2.74770278e-02
  6.35211840e-02  3.42832431e-02 -5.72537035e-02 -1.29570439e-01
 -2.35107411e-02 -1.50633588e-01 -3.94180045e-03 -1.55635281e-02
  1.91622928e-01 -1.95569620e-02  1.20227434e-01 -1.74118411e-02
 -7.69201443e-02 -4.44150195e-02 -6.55689314e-02 -1.32775575e-01
 -3.76074910e-02  5.38797267e-02  6.53239116e-02  1.62745461e-01
 -1.86898336e-02 -1.01958252e-02 -4.78289425e-02 -1.79364413e-01
  1.04455203e-01 -6.69082394e-03  9.07526389e-02 -7.48523558e-03
  3.09978426e-03  2.10866015e-02 -1.36742353e-01 -1.07545398e-01
 -1.02243781e-01  3.01643480e-02 -1.26472011e-01 -6.08762018e-02
 -1.99170150e-02  2.31094547e-02  1.16529509e-01  1.42359570e-01
  7.98728541e-02 -1.13031164e-01  3.15535627e-02  4.48025391e-02
 -2.30494980e-02  3.38950977e-02  1.40374219e-02 -3.45689394e-02
 -6.04420863e-02  1.20326653e-01 -2.57971771e-02 -6.03926964e-02
  3.48576494e-02 -1.40506089e-01 -3.30173559e-02  1.05480187e-01
 -1.01540200e-01  1.29816756e-01 -5.85582890e-02  2.33968161e-02
  1.71460390e-01  4.84713539e-02 -1.05050832e-01 -1.51200956e-02] 0.08081910014152527*/
    static const fixed weights8[] ={
    1498, 12402, -11164, 34253, -3405, -21456, -9304, -12986, 223, 9637, -21075, 6841, 
    14486, 3467, 14879, 8230, -11793, 6295, -7041, -3608, -8185, -9810, -5698, 3844, 
    7888, 17187, 6153, -11485, -1269, 4461, 1293, -5981, -4173, -5495, 5131, 1055, 6165, 
    727, 3200, -7433, 4863, 15496, -11909, -2511, -10224, -7375, 9208, 2831, 6414, 7542, 
    1257, -668, 12093, -9749, -14924, -5172, 20238, 8411, 11086, -11226, 19968, -7133, 
    -628, -5892, 11493, 7320, 1627, -11942, -3061, 1454, 16017, -730, 16658, 11219, 
    14021, -14009, -4754, -7908, -9193, 6491, -9103, -5757, 19834, 6768, -1831, 507, 
    -7692, 19818, -14576, -13998, 19474, 2850, -786, -5877, 19511, -2611, -5523, -17552, 
    -11829, 14443, -3862, 11313, 10498, 3143, 13062, 16683, -2591, -10711, -13710, 3263, 
    2822, -11757, 6385, -13795, 6116, -7540, -11347, -9848, -12998, -1201, -5310, 3545, 
    -7746, 7845, 8587, -3409, -6497, 499, -6796, -12213, -5197, -11113, -287, 16829, 
    -11749, 16003, -18152, -4467, -24139, 15027, 4361, 10171, -16730, -4669, -13661, 
    -4754, -6343, 313, -16706, 6046, -304, -8597, 13610, 14545, -14034, -17007, -3180, 
    8270, 12620, -10165, -6498, -14662, 8589, -7590, 20859, 371, -1811, -12640, 16930, 
    -13183, 10620, 2395, 9392, 25610, 5831, -206, -1779, 896, 3719, -1439, -3917, -3836, 
    -13388, 4502, 12518, -15400, 17855, -7896, 3139, 6410, -5877, -21679, 19795, 18964, 
    568, 9305, -6062, 12716, 12416, -1216, 2447, 15886, -13850, 7610, -1675, 10689, 
    -6036, 5405, -21221, -6648, -8585, -8, 6210, -17873, -17685, 9963, 19760, 4072, 
    5275, -20814, 11297, -3475, -3840, 7934, 8054, -13754, -23866, 3827, -17326, -138, 
    -2183, 5293, -6861, -10473, 20346, -3889, -756, 11143, 4601, 2708, -5222, 1269, 
    13622, -2114, -3543, 1463, -7311, -3790, 446, 3407, 4982, -8480, -3392, 7096, -4389, 
    -12461, 8373, -7848, -4364, -2006, -4592, 12884, -4538, -3712, 9473, 10721, -1550, 
    -4851, -7326, 3739, -585, -12002, -16586, -5987, 1612, -20921, -1892, -6770, -3937, 
    7146, 27049, -1412, 19397, 17911, 10059, -3382, -9178, 4658, 2004, -9085, -1462, 
    -5950, 13084, -14121, 7856, 6515, 4877, 21042, -17995, 19895, -5198, 3593, -5094, 
    -1894, 14029, 2876, 12996, -26946, 6533, 3006, 1305, -4663, 5514, 8598, -59, 1041, 
    -15720, -7542, -20273, 15783, -7523, -1359, 2095, 7611, -11634, 17029, -10140, 950, 
    7863, 8662, 4547, 23483, 16113, 3409, 769, 321, 7126, 14253, -4287, -21401, 1468, 
    -7894, 5725, -16533, 6467, 18941, -5033, -7711, -99, -2400, -13842, -6357, 713, 
    2665, -258, -19865, -5303, 5344, -5895, 2663, 21806, 12203, -22500, 20996, 7331, 
    8645, -7961, -1149, 5810, -3027, 2743, 1761, -9369, 15827, -1607, -2151, 12566, 
    -5973, 10367, -1688, 4138, -27022, 18630, 4544, 4474, 2635, -816, 3080, -10733, 
    -2319, 1865, 3020, -16612, 2865, -17364, -4648, -4097, 15161, 29178, -27762, 17017, 
    -2471, 4537, 23860, -9718, 11854, -10822, -7605, -1119, -10801, 1329, 20735, 10318, 
    1967, 9707, -9549, -7552, -11450, -16987, -12656, 7302, 14749, -5665, 8115, 22364, 
    -9159, 5689, 9810, -16446, 12744, -252, 13134, -7552, 5090, -11659, 11732, 3629, 
    -17138, -9437, 10583, 5054, -1185, 15227, -24254, 904, 16382, 20104, 3601, 8326, 
    4494, -7504, -16983, -3082, -19744, -517, -2040, 25116, -2563, 15758, -2282, -10082, 
    -5822, -8594, -17403, -4929, 7062, 8562, 21331, -2450, -1336, -6269, -23510, 13691, 
    -877, 11895, -981, 406, 2764, -17923, -14096, -13401, 3954, -16577, -7979, -2611, 
    3029, 15274, 18659, 10469, -14815, 4136, 5872, -3021, 4443, 1840, -4531, -7922, 
    15771, -3381, -7916, 4569, -18416, -4328, 13825, -13309, 17015, -7675, 3067, 22474, 
    6353, -13769, -1982
    };
    
    static const neuron_t neuron8 = {weights8, 10593  };
    neurons[8]=neuron8;

    /* [-6.54483363e-02  7.43586719e-02  1.28728803e-02 -1.50537580e-01
  2.68312339e-02  2.76496019e-02 -7.80983791e-02  8.86134729e-02
  1.16537273e-01 -5.16597740e-03 -6.73068017e-02 -2.17226315e-02
 -1.03044054e-02  2.98327468e-02 -7.58453533e-02  1.11670732e-01
 -1.47694379e-01  7.62791187e-02 -9.21982303e-02  1.53917119e-01
  6.15745895e-02 -6.19391985e-02  8.73766541e-02  9.62227061e-02
  2.04048008e-02 -1.04551479e-01  2.02745162e-02 -2.68437415e-02
  1.85423195e-02  2.50474904e-02  4.08193208e-02 -4.92374599e-02
 -2.24686763e-03 -8.12603254e-03 -1.99281499e-02 -5.37107885e-02
 -1.37202665e-02  9.17145759e-02  3.71331610e-02  1.85745358e-01
  4.54665236e-02 -4.86552231e-02  2.73980740e-02 -1.05374381e-01
  6.58139810e-02  2.85980664e-02 -3.29074860e-02 -7.86990896e-02
  5.08737378e-02 -1.21291623e-01 -5.24889976e-02  2.88371630e-02
 -8.94371122e-02  2.64380639e-03 -9.94730368e-02 -4.30190563e-02
  1.25141129e-01  8.02337751e-02  4.35778461e-02  4.54858691e-02
 -4.02210206e-02  1.48626804e-01  5.52314781e-02 -4.46175970e-02
  5.17525449e-02 -1.18949926e-02  3.56420130e-02  1.32847026e-01
  8.86159763e-02 -1.65867507e-01  2.41760164e-02 -4.62980866e-02
  1.06642775e-01  1.64400153e-02  4.44872789e-02  1.20470643e-01
  2.39927322e-02 -4.23640460e-02 -4.77520637e-02  1.48576414e-02
  1.29925892e-01 -4.06011641e-02  1.33945450e-01 -6.33783638e-02
 -4.80099805e-02  3.90375294e-02  4.82532121e-02 -1.19779311e-01
 -2.19478067e-02 -2.39939578e-02  8.84124860e-02  1.29555225e-01
  1.01905257e-01 -1.67270482e-01  1.55779183e-01  1.50834592e-02
 -5.95726110e-02  4.10786271e-02  7.67726824e-02  3.99374068e-02
 -1.42165855e-01  1.34530440e-01  4.02352475e-02 -2.60383468e-02
 -7.77837709e-02 -1.32889494e-01 -6.11527003e-02 -5.00596575e-02
 -1.64448546e-04  6.65343627e-02  2.50828024e-02  6.67069182e-02
 -6.54948875e-02 -3.61906625e-02  2.20914576e-02 -9.33486521e-02
 -6.61637308e-03 -1.10148840e-01 -3.98964398e-02  2.83496827e-02
 -1.79707512e-01 -1.77837703e-02 -2.23715883e-03 -1.95269838e-01
  1.09101534e-01 -1.55071571e-01 -1.97006613e-02 -3.41109047e-03
  5.99149913e-02  1.44535741e-02  1.00604836e-02 -5.02984785e-02
 -2.49860454e-02 -7.69825652e-02 -2.40298267e-03  5.49916923e-02
  2.14682352e-02 -1.12269744e-02 -1.64551049e-01  5.86275384e-02
  5.75940683e-02  6.86112046e-02  3.52460034e-02  1.88158825e-02
 -9.30118188e-02 -5.04681244e-02 -3.55922431e-02 -3.94707546e-02
  2.16609556e-02  2.08053254e-02 -2.81538963e-02 -8.95043761e-02
  2.55766418e-02  1.65936664e-01  3.48514970e-03 -1.02983676e-01
 -9.84346122e-02 -1.12593360e-01 -3.65233980e-02 -3.40735912e-02
  2.35375594e-02 -8.23310316e-02  6.92319348e-02 -7.51390979e-02
 -2.74318047e-02  6.85368851e-02  1.14001170e-01 -5.08090407e-02
 -5.22154756e-03  6.69580996e-02  7.65897185e-02 -7.35075027e-02
  8.01176652e-02 -6.84578344e-02  5.95649742e-02  5.34058511e-02
 -1.77753419e-02  1.28194794e-01 -4.45011295e-02 -3.12772878e-02
  1.51431024e-01 -3.52604389e-02  1.30964108e-02  8.26122165e-02
  4.31566611e-02  3.88340764e-02 -8.43245462e-02 -7.52452612e-02
  3.31332199e-02  4.33022045e-02 -3.28164026e-02 -1.06577277e-01
  3.62418629e-02 -1.27465993e-01 -1.67124141e-02  2.01588757e-02
 -4.00230773e-02  1.27450051e-02  4.06759530e-02 -1.75866969e-02
 -2.14555740e-01 -6.14312105e-02  4.85256687e-02  1.02002760e-02
  3.78127396e-02  1.12890691e-01  1.40390947e-01 -5.17150946e-02
  1.26714393e-01  1.22118993e-02 -8.51294845e-02  1.17747366e-01
  7.98749253e-02 -4.34411354e-02  1.10495418e-01 -1.44177796e-02
 -2.44054794e-02 -1.82342827e-02  5.05948253e-02 -3.53974178e-02
 -4.08564173e-02 -8.39918256e-02  6.59298450e-02 -3.44205685e-02
 -3.45795304e-02 -1.29075095e-01 -7.11046755e-02  4.94652763e-02
  7.30482489e-02  8.44008699e-02  4.99322917e-03  2.45344657e-02
  7.90555775e-02 -6.00011386e-02  1.04565419e-01  1.42842799e-01
  6.63411543e-02  1.12169133e-02  1.26959560e-02 -3.32001341e-03
 -3.70077416e-02  1.48985133e-01  4.09789532e-02  6.81129321e-02
  1.06576875e-01  2.90486179e-02  1.03303418e-02  4.42403480e-02
  1.60422459e-01  1.08785398e-01 -1.16071530e-01 -7.39017352e-02
 -9.16830823e-02  4.67526019e-02 -9.42187905e-02  1.41882086e-02
  3.65939215e-02  2.55576856e-02  1.74583167e-01 -2.60764938e-02
 -1.08357050e-01 -2.21067872e-02  1.18891291e-01 -8.38539302e-02
 -4.09753025e-02  8.84109512e-02 -9.83958393e-02 -3.88259627e-02
 -6.95171207e-02 -4.29604249e-03 -1.78848177e-01 -1.32270120e-02
 -8.40203092e-02 -1.57257169e-01 -6.36130050e-02 -2.37166658e-02
  6.20458163e-02 -1.30406111e-01 -1.42392749e-03  5.11275046e-03
  4.21610959e-02 -2.23902781e-02  3.54996175e-02 -4.23699059e-03
  1.07864678e-01  4.27560285e-02 -1.92628186e-02  6.71403529e-03
 -1.29661942e-02  1.59608070e-02 -9.92072374e-02 -1.66518837e-01
 -4.30334769e-02  2.41260499e-01 -1.95211004e-02  1.01451762e-01
 -5.87693341e-02 -1.75290611e-02 -5.10223024e-03  4.99715433e-02
 -6.80732131e-02  9.67179798e-03 -4.48451005e-03 -8.89444202e-02
 -1.61587764e-02 -1.58938803e-02 -8.57989267e-02 -9.82914940e-02
  2.05866322e-02 -8.88282061e-02 -1.41905263e-01 -5.90459257e-03
 -5.76694608e-02  1.56185016e-01  6.61016554e-02  2.58354358e-02
  7.09705474e-03 -6.54518306e-02 -1.03638254e-01  4.22078446e-02
 -1.89340375e-02 -1.98276024e-02 -2.09485173e-01 -2.28458680e-02
 -2.95775430e-03 -7.89784715e-02  3.69052961e-02  8.44660699e-02
  1.65875629e-01  4.92332727e-02 -1.21382251e-01 -8.93395394e-02
 -4.83017936e-02  1.91725232e-02 -5.22497594e-02  5.89288957e-02
 -7.84720182e-02  3.43723036e-02  7.44791925e-02  1.49287423e-02
  5.19641973e-02  3.12708504e-02  3.07409605e-03 -5.82547998e-03
 -4.13697399e-02  1.19210243e-01  7.25358129e-02  7.38207484e-03
  6.24930952e-04 -1.41088832e-02 -3.52271385e-02  4.74457629e-02
  1.57178968e-01  1.01646259e-01 -1.56783462e-01 -9.80562270e-02
  6.85268864e-02  4.72886041e-02  2.71759205e-03 -8.26038048e-02
  3.26704532e-02  3.02669574e-02  2.15292305e-01  4.48301807e-02
 -1.84148010e-02  6.86564371e-02 -4.78563495e-02  1.41112953e-01
 -5.33050969e-02 -2.00322434e-01  9.25491191e-03 -2.63740681e-02
  7.52089405e-03 -6.49726391e-03 -3.17846015e-02  7.96168856e-03
 -3.17551102e-03 -1.18274473e-01  6.72250316e-02 -1.80878520e-01
 -2.53478740e-03 -9.16684717e-02 -1.28532067e-01 -4.98883463e-02
 -8.78223404e-02  1.73029341e-02  8.57014284e-02  5.87440617e-02
  9.11767632e-02 -2.22047009e-02  2.81626871e-03  7.57482722e-02
  2.02265792e-02  2.45067431e-03  4.17120680e-02  3.03359777e-02
  5.56584634e-02  2.05400400e-02 -6.69827610e-02 -2.58696582e-02
 -7.80017376e-02  3.65280136e-02 -1.09219760e-01  8.86721611e-02
  1.01439342e-01  1.10119715e-01 -8.49159807e-02 -2.23953035e-02
  1.28659219e-01  7.73781165e-02  7.34828189e-02  3.42110097e-02
  1.12373158e-01  1.08749419e-02 -8.38230923e-02 -2.03474462e-02
  9.94945392e-02 -2.35054865e-02  8.49424824e-02  2.07061355e-04
 -8.25120360e-02 -8.15195367e-02 -3.56253199e-02  1.24940481e-02
  1.59498565e-02 -5.70170209e-02  1.63081348e-01 -1.51877657e-01
  6.93791658e-02  2.80901268e-02 -4.81114164e-03 -6.55802116e-02
  1.25078887e-01  4.45755608e-02  6.74918666e-02  5.20172827e-02
 -1.44290626e-02 -4.40678336e-02 -1.70433056e-02  6.01466419e-03
  2.54607387e-03  7.65959397e-02 -5.75389937e-02  9.35104489e-02
 -3.27809565e-02  1.53054586e-02 -6.38684481e-02  4.33334634e-02
  1.34816736e-01  7.04989806e-02  6.21627383e-02 -9.23625603e-02
 -4.36861180e-02  1.28850769e-02  3.69526856e-02  2.01967638e-02
  1.48333326e-01  3.99120264e-02 -6.11384474e-02  6.02534525e-02
  1.65283516e-01  4.40622866e-02 -6.70554340e-02  3.34828645e-02
 -1.10570915e-01 -1.55261457e-02 -1.72788501e-01 -1.53390672e-02
 -1.15860172e-01 -9.97335911e-02 -2.02095024e-02  2.20849320e-01
  8.10656846e-02 -6.71872646e-02 -9.23486054e-02  2.54407171e-02
 -7.37773720e-03  8.34261551e-02  1.14816930e-02 -7.64328521e-03
  7.17964349e-03  1.58797428e-01 -5.00258431e-02  8.26863125e-02
 -1.36415614e-02 -4.44956459e-02  4.23459429e-03  7.66425580e-02
 -5.22034289e-03 -1.29082143e-01  6.16075769e-02 -3.11165210e-02
 -6.97804838e-02 -6.00154176e-02 -1.62590548e-01 -3.66058424e-02
 -5.83861116e-03  1.82054061e-02  1.52599495e-02 -9.03652981e-02
  7.20818788e-02  2.55369432e-02 -6.83421046e-02 -1.23561453e-03
 -1.06804125e-01  4.79965024e-02  3.05465534e-02 -9.54027128e-05
 -3.78115959e-02 -3.29099819e-02 -3.13159451e-02  5.83872758e-02] 0.027055202051997185*/
    static const fixed weights9[] ={
    -8578, 9746, 1687, -19731, 3517, 3624, -10237, 11615, 15275, -677, -8822, -2847, 
    -1351, 3910, -9941, 14637, -19359, 9998, -12085, 20174, 8071, -8118, 11453, 12612, 
    2674, -13704, 2657, -3518, 2430, 3283, 5350, -6454, -295, -1065, -2612, -7040, -1798, 
    12021, 4867, 24346, 5959, -6377, 3591, -13812, 8626, 3748, -4313, -10315, 6668, 
    -15898, -6880, 3780, -11723, 347, -13038, -5639, 16402, 10516, 5712, 5962, -5272, 
    19481, 7239, -5848, 6783, -1559, 4672, 17413, 11615, -21741, 3169, -6068, 13978, 
    2155, 5831, 15790, 3145, -5553, -6259, 1947, 17030, -5322, 17556, -8307, -6293, 
    5117, 6325, -15700, -2877, -3145, 11588, 16981, 13357, -21924, 20418, 1977, -7808, 
    5384, 10063, 5235, -18634, 17633, 5274, -3413, -10195, -17418, -8015, -6561, -22, 
    8721, 3288, 8743, -8585, -4744, 2896, -12235, -867, -14437, -5229, 3716, -23555, 
    -2331, -293, -25594, 14300, -20326, -2582, -447, 7853, 1894, 1319, -6593, -3275, 
    -10090, -315, 7208, 2814, -1472, -21568, 7684, 7549, 8993, 4620, 2466, -12191, -6615, 
    -4665, -5174, 2839, 2727, -3690, -11732, 3352, 21750, 457, -13498, -12902, -14758, 
    -4787, -4466, 3085, -10791, 9074, -9849, -3596, 8983, 14942, -6660, -684, 8776, 
    10039, -9635, 10501, -8973, 7807, 7000, -2330, 16803, -5833, -4100, 19848, -4622, 
    1717, 10828, 5657, 5090, -11053, -9863, 4343, 5676, -4301, -13969, 4750, -16707, 
    -2191, 2642, -5246, 1671, 5331, -2305, -28122, -8052, 6360, 1337, 4956, 14797, 18401, 
    -6778, 16609, 1601, -11158, 15433, 10469, -5694, 14483, -1890, -3199, -2390, 6632, 
    -4640, -5355, -11009, 8642, -4512, -4532, -16918, -9320, 6484, 9575, 11063, 654, 
    3216, 10362, -7864, 13706, 18723, 8695, 1470, 1664, -435, -4851, 19528, 5371, 8928, 
    13969, 3807, 1354, 5799, 21027, 14259, -15214, -9686, -12017, 6128, -12349, 1860, 
    4796, 3350, 22883, -3418, -14203, -2898, 15583, -10991, -5371, 11588, -12897, -5089, 
    -9112, -563, -23442, -1734, -11013, -20612, -8338, -3109, 8132, -17093, -187, 670, 
    5526, -2935, 4653, -555, 14138, 5604, -2525, 880, -1700, 2092, -13003, -21826, -5640, 
    31622, -2559, 13297, -7703, -2298, -669, 6550, -8922, 1268, -588, -11658, -2118, 
    -2083, -11246, -12883, 2698, -11643, -18600, -774, -7559, 20471, 8664, 3386, 930, 
    -8579, -13584, 5532, -2482, -2599, -27458, -2994, -388, -10352, 4837, 11071, 21742, 
    6453, -15910, -11710, -6331, 2513, -6848, 7724, -10285, 4505, 9762, 1957, 6811, 
    4099, 403, -764, -5422, 15625, 9507, 968, 82, -1849, -4617, 6219, 20602, 13323, 
    -20550, -12852, 8982, 6198, 356, -10827, 4282, 3967, 28219, 5876, -2414, 8999, -6273, 
    18496, -6987, -26257, 1213, -3457, 986, -852, -4166, 1044, -416, -15502, 8811, -23708, 
    -332, -12015, -16847, -6539, -11511, 2268, 11233, 7700, 11951, -2910, 369, 9928, 
    2651, 321, 5467, 3976, 7295, 2692, -8780, -3391, -10224, 4788, -14316, 11622, 13296, 
    14434, -11130, -2935, 16864, 10142, 9632, 4484, 14729, 1425, -10987, -2667, 13041, 
    -3081, 11134, 27, -10815, -10685, -4669, 1638, 2091, -7473, 21375, -19907, 9094, 
    3682, -631, -8596, 16394, 5843, 8846, 6818, -1891, -5776, -2234, 788, 334, 10040, 
    -7542, 12257, -4297, 2006, -8371, 5680, 17671, 9240, 8148, -12106, -5726, 1689, 
    4843, 2647, 19442, 5231, -8014, 7898, 21664, 5775, -8789, 4389, -14493, -2035, -22648, 
    -2011, -15186, -13072, -2649, 28947, 10625, -8806, -12104, 3335, -967, 10935, 1505, 
    -1002, 941, 20814, -6557, 10838, -1788, -5832, 555, 10046, -684, -16919, 8075, -4079, 
    -9146, -7866, -21311, -4798, -765, 2386, 2000, -11844, 9448, 3347, -8958, -162, 
    -13999, 6291, 4004, -13, -4956, -4314, -4105, 7653
    };
    
    static const neuron_t neuron9 = {weights9, 3546  };
    neurons[9]=neuron9;

    dense_layer_t layer= { 10, neurons};
    return layer;
}

