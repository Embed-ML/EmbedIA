#include "fashionmnist_model.h"

// Initialization function prototypes
quantconv2d_layer_t init_quant_conv2d_3_data(void);
batch_normalization_layer_t init_batch_normalization_12_data(void);
quantconv2d_layer_t init_quant_conv2d_4_data(void);
batch_normalization_layer_t init_batch_normalization_13_data(void);
quantconv2d_layer_t init_quant_conv2d_5_data(void);
batch_normalization_layer_t init_batch_normalization_14_data(void);
quantdense_layer_t init_quant_dense_1_data(void);
batch_normalization_layer_t init_batch_normalization_15_data(void);
dense_layer_t init_dense_5_data(void);


// Global Variables
quantconv2d_layer_t quant_conv2d_3_data;
batch_normalization_layer_t batch_normalization_12_data;
quantconv2d_layer_t quant_conv2d_4_data;
batch_normalization_layer_t batch_normalization_13_data;
quantconv2d_layer_t quant_conv2d_5_data;
batch_normalization_layer_t batch_normalization_14_data;
quantdense_layer_t quant_dense_1_data;
batch_normalization_layer_t batch_normalization_15_data;
dense_layer_t dense_5_data;


void model_init(){
    quant_conv2d_3_data = init_quant_conv2d_3_data();
    batch_normalization_12_data = init_batch_normalization_12_data();
    quant_conv2d_4_data = init_quant_conv2d_4_data();
    batch_normalization_13_data = init_batch_normalization_13_data();
    quant_conv2d_5_data = init_quant_conv2d_5_data();
    batch_normalization_14_data = init_batch_normalization_14_data();
    quant_dense_1_data = init_quant_dense_1_data();
    batch_normalization_15_data = init_batch_normalization_15_data();
    dense_5_data = init_dense_5_data();

}

void model_predict(data3d_t input, data1d_t * output){
  
    prepare_buffers();
    
    //*************** LAYER 0 **************//
    // Layer name: quant_conv2d_3
    data3d_t output0;
        // convert image for first EmbedIA Conv2d layer
        image_adapt_layer(input, &output0);
        input = output0;
    
     quantconv2d_input_not_binary_layer(quant_conv2d_3_data, input, &output0);
    
    // Activation layer for quant_conv2d_3
    relu_activation(output0.data, 10816);
    
    //*************** LAYER 1 **************//
    // Layer name: max_pooling2d_6
    input = output0;
    static const pooling2d_layer_t max_pooling2d_6_data = { 2, 2 };
    max_pooling2d_layer(max_pooling2d_6_data, input, &output0);
    
    //*************** LAYER 2 **************//
    // Layer name: batch_normalization_12
    batch_normalization3d_layer(batch_normalization_12_data, &output0);
    
    //*************** LAYER 3 **************//
    // Layer name: quant_conv2d_4
    input = output0;
    quantconv2d_layer(quant_conv2d_4_data, input, &output0);
    // Activation layer for quant_conv2d_4
    relu_activation(output0.data, 7744);
    
    //*************** LAYER 4 **************//
    // Layer name: max_pooling2d_7
    input = output0;
    static const pooling2d_layer_t max_pooling2d_7_data = { 2, 2 };
    max_pooling2d_layer(max_pooling2d_7_data, input, &output0);
    
    //*************** LAYER 5 **************//
    // Layer name: batch_normalization_13
    batch_normalization3d_layer(batch_normalization_13_data, &output0);
    
    //*************** LAYER 6 **************//
    // Layer name: quant_conv2d_5
    input = output0;
    quantconv2d_layer(quant_conv2d_5_data, input, &output0);
    // Activation layer for quant_conv2d_5
    relu_activation(output0.data, 576);
    
    //*************** LAYER 7 **************//
    // Layer name: batch_normalization_14
    batch_normalization3d_layer(batch_normalization_14_data, &output0);
    
    //*************** LAYER 8 **************//
    // Layer name: flatten_3
    input = output0;
    data1d_t output1;
    flatten3d_layer(input, &output1);
    
    //*************** LAYER 9 **************//
    // Layer name: dropout_3
    data1d_t input1;
    input1 = output1;
    
    
    //*************** LAYER 10 **************//
    // Layer name: quant_dense_1
    input1 = output1;
    quantdense_layer(quant_dense_1_data, input1, &output1);
    // Activation layer for quant_dense_1
    relu_activation(output1.data, 64);
    
    //*************** LAYER 11 **************//
    // Layer name: batch_normalization_15
    batch_normalization1d_layer(batch_normalization_15_data, &output1);
    
    //*************** LAYER 12 **************//
    // Layer name: dense_5
    input1 = output1;
    dense_layer(dense_5_data, input1, &output1);
    
    // Activation layer for dense_5
    softmax_activation(output1.data, 10);
    

    *output = output1;

}

int model_predict_class(data3d_t input, data1d_t * results){
  
   
    model_predict(input, results);
    
    return argmax(*results);
    //return argmax(data1d_t);

}

// Implementation of initialization functions



          quantconv2d_layer_t init_quant_conv2d_3_data(void){

            static quant_filter_t filtros_b[16];
            
            static const uint32_t weights0[]={578813952
            };
            static quant_filter_t filter0 = {1, 3, weights0, FL2FX(0.4061693549156189)};
            filtros_b[0]=filter0;
              
            static const uint32_t weights1[]={41943040
            };
            static quant_filter_t filter1 = {1, 3, weights1, FL2FX(-1.0408833026885986)};
            filtros_b[1]=filter1;
              
            static const uint32_t weights2[]={109051904
            };
            static quant_filter_t filter2 = {1, 3, weights2, FL2FX(0.04880460724234581)};
            filtros_b[2]=filter2;
              
            static const uint32_t weights3[]={4253024256
            };
            static quant_filter_t filter3 = {1, 3, weights3, FL2FX(0.3285691738128662)};
            filtros_b[3]=filter3;
              
            static const uint32_t weights4[]={973078528
            };
            static quant_filter_t filter4 = {1, 3, weights4, FL2FX(0.25663718581199646)};
            filtros_b[4]=filter4;
              
            static const uint32_t weights5[]={2583691264
            };
            static quant_filter_t filter5 = {1, 3, weights5, FL2FX(0.6880521774291992)};
            filtros_b[5]=filter5;
              
            static const uint32_t weights6[]={1233125376
            };
            static quant_filter_t filter6 = {1, 3, weights6, FL2FX(0.8006962537765503)};
            filtros_b[6]=filter6;
              
            static const uint32_t weights7[]={780140544
            };
            static quant_filter_t filter7 = {1, 3, weights7, FL2FX(0.7163945436477661)};
            filtros_b[7]=filter7;
              
            static const uint32_t weights8[]={2600468480
            };
            static quant_filter_t filter8 = {1, 3, weights8, FL2FX(0.7777426242828369)};
            filtros_b[8]=filter8;
              
            static const uint32_t weights9[]={0
            };
            static quant_filter_t filter9 = {1, 3, weights9, FL2FX(-2.3068253993988037)};
            filtros_b[9]=filter9;
              
            static const uint32_t weights10[]={3758096384
            };
            static quant_filter_t filter10 = {1, 3, weights10, FL2FX(0.02544320933520794)};
            filtros_b[10]=filter10;
              
            static const uint32_t weights11[]={2466250752
            };
            static quant_filter_t filter11 = {1, 3, weights11, FL2FX(0.8907575011253357)};
            filtros_b[11]=filter11;
              
            static const uint32_t weights12[]={3623878656
            };
            static quant_filter_t filter12 = {1, 3, weights12, FL2FX(0.2859678566455841)};
            filtros_b[12]=filter12;
              
            static const uint32_t weights13[]={1258291200
            };
            static quant_filter_t filter13 = {1, 3, weights13, FL2FX(0.5072418451309204)};
            filtros_b[13]=filter13;
              
            static const uint32_t weights14[]={33554432
            };
            static quant_filter_t filter14 = {1, 3, weights14, FL2FX(-1.1004241704940796)};
            filtros_b[14]=filter14;
              
            static const uint32_t weights15[]={1837105152
            };
            static quant_filter_t filter15 = {1, 3, weights15, FL2FX(0.990117609500885)};
            filtros_b[15]=filter15;
              
            quantconv2d_layer_t layer = {16,filtros_b};
            return layer;
          }
            
batch_normalization_layer_t init_batch_normalization_12_data(void){

    static const fixed inv_gamma_dev[] ={
    FL2FX(0.859475/sqrt(5.261692+0.001000)), FL2FX(1.518206/sqrt(8.264037+0.001000)), 
    FL2FX(0.921601/sqrt(3.901204+0.001000)), FL2FX(0.703234/sqrt(4.482805+0.001000)), 
    FL2FX(0.504828/sqrt(0.621109+0.001000)), FL2FX(0.753004/sqrt(1.853972+0.001000)), 
    FL2FX(1.050289/sqrt(1.691703+0.001000)), FL2FX(0.606456/sqrt(2.450305+0.001000)), 
    FL2FX(0.599561/sqrt(2.377116+0.001000)), FL2FX(1.915929/sqrt(6.010066+0.001000)), 
    FL2FX(0.927584/sqrt(2.952105+0.001000)), FL2FX(0.821255/sqrt(2.989714+0.001000)), 
    FL2FX(1.129678/sqrt(1.633579+0.001000)), FL2FX(0.925428/sqrt(1.594417+0.001000)), 
    FL2FX(1.571090/sqrt(6.837259+0.001000)), FL2FX(0.785975/sqrt(3.248444+0.001000)), 
  
    };
    static const fixed std_beta[] ={
    FL2FX(0.143211-3.085560*0.859475/sqrt(5.261692+0.001000)), FL2FX(-0.490970-3.420699*1.518206/sqrt(8.264037+0.001000)), 
    FL2FX(0.016219-2.587269*0.921601/sqrt(3.901204+0.001000)), FL2FX(-0.537282-1.504546*0.703234/sqrt(4.482805+0.001000)), 
    FL2FX(-0.299716-0.895746*0.504828/sqrt(0.621109+0.001000)), FL2FX(-0.090223-1.525813*0.753004/sqrt(1.853972+0.001000)), 
    FL2FX(-0.161680-1.488675*1.050289/sqrt(1.691703+0.001000)), FL2FX(-0.235841-1.816636*0.606456/sqrt(2.450305+0.001000)), 
    FL2FX(-0.160454-1.413434*0.599561/sqrt(2.377116+0.001000)), FL2FX(-0.984760-2.690658*1.915929/sqrt(6.010066+0.001000)), 
    FL2FX(0.067720-2.022292*0.927584/sqrt(2.952105+0.001000)), FL2FX(0.018062-2.272368*0.821255/sqrt(2.989714+0.001000)), 
    FL2FX(-0.219593-1.200172*1.129678/sqrt(1.633579+0.001000)), FL2FX(-0.315465-1.217730*0.925428/sqrt(1.594417+0.001000)), 
    FL2FX(-0.548320-3.115503*1.571090/sqrt(6.837259+0.001000)), FL2FX(-0.389025-1.529717*0.785975/sqrt(3.248444+0.001000)), 
  
    };

    static const batch_normalization_layer_t norm = { 16, inv_gamma_dev, std_beta };
    return norm;
}


          quantconv2d_layer_t init_quant_conv2d_4_data(void){

            static quant_filter_t filtros_b[64];
            
            static const uint32_t weights0[]={1685224192,1735322016,3909087947,496229295,2456027136
            };
            static quant_filter_t filter0 = {16, 3, weights0, FL2FX(0.2172354906797409)};
            filtros_b[0]=filter0;
              
            static const uint32_t weights1[]={431933965,3515444534,2471301824,1303679561,3066626048
            };
            static quant_filter_t filter1 = {16, 3, weights1, FL2FX(-0.004691683687269688)};
            filtros_b[1]=filter1;
              
            static const uint32_t weights2[]={857312600,2453385527,1677073292,3435471572,2291335168
            };
            static quant_filter_t filter2 = {16, 3, weights2, FL2FX(0.2971287965774536)};
            filtros_b[2]=filter2;
              
            static const uint32_t weights3[]={1932983121,3003640654,1052145355,3698142869,2813722624
            };
            static quant_filter_t filter3 = {16, 3, weights3, FL2FX(-0.004930376540869474)};
            filtros_b[3]=filter3;
              
            static const uint32_t weights4[]={1924791347,2714512990,563917385,3096461345,2267873280
            };
            static quant_filter_t filter4 = {16, 3, weights4, FL2FX(0.4806590676307678)};
            filtros_b[4]=filter4;
              
            static const uint32_t weights5[]={2708513539,1910695537,678212488,3700465383,2300706816
            };
            static quant_filter_t filter5 = {16, 3, weights5, FL2FX(-0.07803566753864288)};
            filtros_b[5]=filter5;
              
            static const uint32_t weights6[]={4101912832,2138765107,3231480377,411386599,283377664
            };
            static quant_filter_t filter6 = {16, 3, weights6, FL2FX(0.20833918452262878)};
            filtros_b[6]=filter6;
              
            static const uint32_t weights7[]={1261801050,437629114,2528523666,3335218612,2896035840
            };
            static quant_filter_t filter7 = {16, 3, weights7, FL2FX(0.2852506637573242)};
            filtros_b[7]=filter7;
              
            static const uint32_t weights8[]={247980396,2716359703,2333736897,3786455624,1041039360
            };
            static quant_filter_t filter8 = {16, 3, weights8, FL2FX(0.27122369408607483)};
            filtros_b[8]=filter8;
              
            static const uint32_t weights9[]={590896272,2589400282,1961439759,1023316809,2792161280
            };
            static quant_filter_t filter9 = {16, 3, weights9, FL2FX(0.6271102428436279)};
            filtros_b[9]=filter9;
              
            static const uint32_t weights10[]={1304879743,3566295511,2603183827,863686860,1993932800
            };
            static quant_filter_t filter10 = {16, 3, weights10, FL2FX(0.5403072834014893)};
            filtros_b[10]=filter10;
              
            static const uint32_t weights11[]={3446649670,3638020306,3525181182,1369699619,1952382976
            };
            static quant_filter_t filter11 = {16, 3, weights11, FL2FX(0.3941102921962738)};
            filtros_b[11]=filter11;
              
            static const uint32_t weights12[]={2457279666,2175429199,1821184269,2846444724,187498496
            };
            static quant_filter_t filter12 = {16, 3, weights12, FL2FX(0.49369537830352783)};
            filtros_b[12]=filter12;
              
            static const uint32_t weights13[]={444079760,806919220,117569840,2450524174,3222142976
            };
            static quant_filter_t filter13 = {16, 3, weights13, FL2FX(-0.020097479224205017)};
            filtros_b[13]=filter13;
              
            static const uint32_t weights14[]={2454521566,3171814927,642420024,2202929812,2030501888
            };
            static quant_filter_t filter14 = {16, 3, weights14, FL2FX(0.030757801607251167)};
            filtros_b[14]=filter14;
              
            static const uint32_t weights15[]={3925637123,1314139084,3834903597,412708431,2699427840
            };
            static quant_filter_t filter15 = {16, 3, weights15, FL2FX(0.33992648124694824)};
            filtros_b[15]=filter15;
              
            static const uint32_t weights16[]={747834590,265513447,427628366,2243334728,3081830400
            };
            static quant_filter_t filter16 = {16, 3, weights16, FL2FX(0.49427589774131775)};
            filtros_b[16]=filter16;
              
            static const uint32_t weights17[]={428394109,3232783739,2433158886,4078991546,2123956224
            };
            static quant_filter_t filter17 = {16, 3, weights17, FL2FX(-0.03440918028354645)};
            filtros_b[17]=filter17;
              
            static const uint32_t weights18[]={2050841234,2751341548,2094825311,2621032235,3393060864
            };
            static quant_filter_t filter18 = {16, 3, weights18, FL2FX(0.035640764981508255)};
            filtros_b[18]=filter18;
              
            static const uint32_t weights19[]={4075944101,2940683055,3585562694,1030342495,2262564864
            };
            static quant_filter_t filter19 = {16, 3, weights19, FL2FX(0.5377961993217468)};
            filtros_b[19]=filter19;
              
            static const uint32_t weights20[]={2860802664,2174586222,397509715,859403502,1721171968
            };
            static quant_filter_t filter20 = {16, 3, weights20, FL2FX(0.6183759570121765)};
            filtros_b[20]=filter20;
              
            static const uint32_t weights21[]={1626084881,3518774482,1770430514,1511553207,1684602880
            };
            static quant_filter_t filter21 = {16, 3, weights21, FL2FX(0.02450157143175602)};
            filtros_b[21]=filter21;
              
            static const uint32_t weights22[]={2933615531,1726702880,1768218603,2766594471,2661351424
            };
            static quant_filter_t filter22 = {16, 3, weights22, FL2FX(0.26457905769348145)};
            filtros_b[22]=filter22;
              
            static const uint32_t weights23[]={1941539025,2892468463,617222430,3712853553,2302017536
            };
            static quant_filter_t filter23 = {16, 3, weights23, FL2FX(0.21199898421764374)};
            filtros_b[23]=filter23;
              
            static const uint32_t weights24[]={722569945,2665706229,863969158,3310347124,2377318400
            };
            static quant_filter_t filter24 = {16, 3, weights24, FL2FX(0.2038869708776474)};
            filtros_b[24]=filter24;
              
            static const uint32_t weights25[]={2371299661,3897141028,2653264140,3335834218,3683778560
            };
            static quant_filter_t filter25 = {16, 3, weights25, FL2FX(0.46581414341926575)};
            filtros_b[25]=filter25;
              
            static const uint32_t weights26[]={2244206719,2382742401,3236185059,4037443330,532676608
            };
            static quant_filter_t filter26 = {16, 3, weights26, FL2FX(-0.7322949767112732)};
            filtros_b[26]=filter26;
              
            static const uint32_t weights27[]={607287602,1898895966,215691849,4253103251,2701459456
            };
            static quant_filter_t filter27 = {16, 3, weights27, FL2FX(0.07587706297636032)};
            filtros_b[27]=filter27;
              
            static const uint32_t weights28[]={3681391253,2551293236,2519068038,3330429226,3365535744
            };
            static quant_filter_t filter28 = {16, 3, weights28, FL2FX(-0.054228223860263824)};
            filtros_b[28]=filter28;
              
            static const uint32_t weights29[]={1533384399,917583028,2452228532,3846615889,160563200
            };
            static quant_filter_t filter29 = {16, 3, weights29, FL2FX(0.04020422697067261)};
            filtros_b[29]=filter29;
              
            static const uint32_t weights30[]={113995196,2388375131,2236154479,632444972,860684288
            };
            static quant_filter_t filter30 = {16, 3, weights30, FL2FX(0.43529200553894043)};
            filtros_b[30]=filter30;
              
            static const uint32_t weights31[]={792565977,896349842,1263279502,3427578413,2907373568
            };
            static quant_filter_t filter31 = {16, 3, weights31, FL2FX(-0.010750201530754566)};
            filtros_b[31]=filter31;
              
            static const uint32_t weights32[]={4163782566,3923044714,3164372853,429593835,3233021952
            };
            static quant_filter_t filter32 = {16, 3, weights32, FL2FX(0.2646222710609436)};
            filtros_b[32]=filter32;
              
            static const uint32_t weights33[]={3404558503,1816149545,3852855621,814958895,1391198208
            };
            static quant_filter_t filter33 = {16, 3, weights33, FL2FX(-0.27041569352149963)};
            filtros_b[33]=filter33;
              
            static const uint32_t weights34[]={3443958601,690245779,448528338,3579918918,2890399744
            };
            static quant_filter_t filter34 = {16, 3, weights34, FL2FX(0.017410416156053543)};
            filtros_b[34]=filter34;
              
            static const uint32_t weights35[]={1685420114,2064024200,3436216282,2554340871,2940403712
            };
            static quant_filter_t filter35 = {16, 3, weights35, FL2FX(0.017645863816142082)};
            filtros_b[35]=filter35;
              
            static const uint32_t weights36[]={3813783745,1060574085,3001160130,4249468885,2914189312
            };
            static quant_filter_t filter36 = {16, 3, weights36, FL2FX(0.2701622545719147)};
            filtros_b[36]=filter36;
              
            static const uint32_t weights37[]={3381193800,1538644762,2445088482,3886773698,3433627648
            };
            static quant_filter_t filter37 = {16, 3, weights37, FL2FX(0.5433737635612488)};
            filtros_b[37]=filter37;
              
            static const uint32_t weights38[]={3446154894,2633095322,3132047414,52616635,1617756160
            };
            static quant_filter_t filter38 = {16, 3, weights38, FL2FX(0.0062707350589334965)};
            filtros_b[38]=filter38;
              
            static const uint32_t weights39[]={4033886211,2961154808,1883431000,4195368483,190316544
            };
            static quant_filter_t filter39 = {16, 3, weights39, FL2FX(-0.20639140903949738)};
            filtros_b[39]=filter39;
              
            static const uint32_t weights40[]={3680875084,3579402549,3669116310,1197941138,751501312
            };
            static quant_filter_t filter40 = {16, 3, weights40, FL2FX(0.05175134539604187)};
            filtros_b[40]=filter40;
              
            static const uint32_t weights41[]={2757453242,1078601061,845077097,2178191960,2551316480
            };
            static quant_filter_t filter41 = {16, 3, weights41, FL2FX(0.29498955607414246)};
            filtros_b[41]=filter41;
              
            static const uint32_t weights42[]={3286364391,4123060033,2307703266,4051958626,265355264
            };
            static quant_filter_t filter42 = {16, 3, weights42, FL2FX(-0.01525147631764412)};
            filtros_b[42]=filter42;
              
            static const uint32_t weights43[]={1924233649,2735518444,2794893580,2585355605,3402760192
            };
            static quant_filter_t filter43 = {16, 3, weights43, FL2FX(0.7568378448486328)};
            filtros_b[43]=filter43;
              
            static const uint32_t weights44[]={305580601,580275821,910767848,2335020105,3073769472
            };
            static quant_filter_t filter44 = {16, 3, weights44, FL2FX(0.24471940100193024)};
            filtros_b[44]=filter44;
              
            static const uint32_t weights45[]={641419673,1019848037,845454189,211126692,2612133888
            };
            static quant_filter_t filter45 = {16, 3, weights45, FL2FX(0.06612448394298553)};
            filtros_b[45]=filter45;
              
            static const uint32_t weights46[]={2380587646,3578577555,1766653958,1931591413,770244608
            };
            static quant_filter_t filter46 = {16, 3, weights46, FL2FX(0.31617116928100586)};
            filtros_b[46]=filter46;
              
            static const uint32_t weights47[]={1680914443,4190397152,3906531387,479055079,2178416640
            };
            static quant_filter_t filter47 = {16, 3, weights47, FL2FX(0.2529347836971283)};
            filtros_b[47]=filter47;
              
            static const uint32_t weights48[]={2067045970,3080678346,2766779090,3183064031,3844931584
            };
            static quant_filter_t filter48 = {16, 3, weights48, FL2FX(0.4407086968421936)};
            filtros_b[48]=filter48;
              
            static const uint32_t weights49[]={3497010817,2985843836,503544120,4030740499,1211564032
            };
            static quant_filter_t filter49 = {16, 3, weights49, FL2FX(-0.2572692036628723)};
            filtros_b[49]=filter49;
              
            static const uint32_t weights50[]={3706156806,1837756541,3697049599,59150695,2062417920
            };
            static quant_filter_t filter50 = {16, 3, weights50, FL2FX(0.19543607532978058)};
            filtros_b[50]=filter50;
              
            static const uint32_t weights51[]={160417277,3347514241,3504574403,3791636224,1073217536
            };
            static quant_filter_t filter51 = {16, 3, weights51, FL2FX(-0.1395343840122223)};
            filtros_b[51]=filter51;
              
            static const uint32_t weights52[]={4218763777,2655954342,3604150550,2584323919,3369271296
            };
            static quant_filter_t filter52 = {16, 3, weights52, FL2FX(-0.059187229722738266)};
            filtros_b[52]=filter52;
              
            static const uint32_t weights53[]={2714892527,2387817267,2313553347,3966565154,2376663040
            };
            static quant_filter_t filter53 = {16, 3, weights53, FL2FX(-0.25735968351364136)};
            filtros_b[53]=filter53;
              
            static const uint32_t weights54[]={3496113414,1958502336,3568101496,4127866434,1358168064
            };
            static quant_filter_t filter54 = {16, 3, weights54, FL2FX(-0.20527705550193787)};
            filtros_b[54]=filter54;
              
            static const uint32_t weights55[]={2210915662,1563767761,2686601160,3848682002,1067974656
            };
            static quant_filter_t filter55 = {16, 3, weights55, FL2FX(-0.19603197276592255)};
            filtros_b[55]=filter55;
              
            static const uint32_t weights56[]={260555260,260112917,187439055,3251078660,3215654912
            };
            static quant_filter_t filter56 = {16, 3, weights56, FL2FX(-0.2952280640602112)};
            filtros_b[56]=filter56;
              
            static const uint32_t weights57[]={464621273,2587300121,1130502378,4266539792,3481600000
            };
            static quant_filter_t filter57 = {16, 3, weights57, FL2FX(0.5028913617134094)};
            filtros_b[57]=filter57;
              
            static const uint32_t weights58[]={1842535290,3587434866,4234336219,765899768,3061579776
            };
            static quant_filter_t filter58 = {16, 3, weights58, FL2FX(0.018407966941595078)};
            filtros_b[58]=filter58;
              
            static const uint32_t weights59[]={2097071923,771526663,3981017083,428738295,4159242240
            };
            static quant_filter_t filter59 = {16, 3, weights59, FL2FX(-0.3476545810699463)};
            filtros_b[59]=filter59;
              
            static const uint32_t weights60[]={1227247698,2562120268,3583190739,153695313,2003107840
            };
            static quant_filter_t filter60 = {16, 3, weights60, FL2FX(0.6274764537811279)};
            filtros_b[60]=filter60;
              
            static const uint32_t weights61[]={3536331821,2582704105,3164448853,842389864,1173749760
            };
            static quant_filter_t filter61 = {16, 3, weights61, FL2FX(0.10084597766399384)};
            filtros_b[61]=filter61;
              
            static const uint32_t weights62[]={1689525089,4159083464,1824667163,952705553,2808872960
            };
            static quant_filter_t filter62 = {16, 3, weights62, FL2FX(0.23237518966197968)};
            filtros_b[62]=filter62;
              
            static const uint32_t weights63[]={2529191341,2940253001,3171271237,852743578,334233600
            };
            static quant_filter_t filter63 = {16, 3, weights63, FL2FX(0.02211354859173298)};
            filtros_b[63]=filter63;
              
            quantconv2d_layer_t layer = {64,filtros_b};
            return layer;
          }
            
batch_normalization_layer_t init_batch_normalization_13_data(void){

    static const fixed inv_gamma_dev[] ={
    FL2FX(1.295200/sqrt(263.078888+0.001000)), FL2FX(0.999835/sqrt(108.300644+0.001000)), 
    FL2FX(0.928370/sqrt(110.086494+0.001000)), FL2FX(1.100259/sqrt(110.728729+0.001000)), 
    FL2FX(0.628929/sqrt(82.908058+0.001000)), FL2FX(0.954274/sqrt(178.912460+0.001000)), 
    FL2FX(1.248912/sqrt(184.492691+0.001000)), FL2FX(1.001489/sqrt(121.794418+0.001000)), 
    FL2FX(0.983482/sqrt(109.038193+0.001000)), FL2FX(0.981253/sqrt(120.617516+0.001000)), 
    FL2FX(0.980167/sqrt(84.543633+0.001000)), FL2FX(1.273850/sqrt(127.494148+0.001000)), 
    FL2FX(0.742787/sqrt(126.554062+0.001000)), FL2FX(0.964583/sqrt(269.108826+0.001000)), 
    FL2FX(1.010168/sqrt(166.777878+0.001000)), FL2FX(0.954731/sqrt(117.137779+0.001000)), 
    FL2FX(1.043302/sqrt(183.258667+0.001000)), FL2FX(0.994408/sqrt(89.626068+0.001000)), 
    FL2FX(1.261690/sqrt(100.348488+0.001000)), FL2FX(0.914788/sqrt(75.082848+0.001000)), 
    FL2FX(0.913166/sqrt(104.193855+0.001000)), FL2FX(0.862743/sqrt(136.748276+0.001000)), 
    FL2FX(1.211244/sqrt(162.954422+0.001000)), FL2FX(0.761744/sqrt(121.508141+0.001000)), 
    FL2FX(0.928655/sqrt(105.975197+0.001000)), FL2FX(1.181275/sqrt(172.891769+0.001000)), 
    FL2FX(1.313107/sqrt(172.821014+0.001000)), FL2FX(1.164546/sqrt(274.117188+0.001000)), 
    FL2FX(1.448877/sqrt(359.582153+0.001000)), FL2FX(1.018143/sqrt(182.246429+0.001000)), 
    FL2FX(0.949223/sqrt(148.197311+0.001000)), FL2FX(0.928975/sqrt(103.434967+0.001000)), 
    FL2FX(1.331212/sqrt(110.334663+0.001000)), FL2FX(0.951596/sqrt(128.906631+0.001000)), 
    FL2FX(0.934261/sqrt(154.258759+0.001000)), FL2FX(0.974286/sqrt(176.088882+0.001000)), 
    FL2FX(0.900589/sqrt(101.396660+0.001000)), FL2FX(0.870046/sqrt(89.842018+0.001000)), 
    FL2FX(0.901445/sqrt(143.257553+0.001000)), FL2FX(1.026217/sqrt(177.710754+0.001000)), 
    FL2FX(1.313180/sqrt(209.622787+0.001000)), FL2FX(0.790559/sqrt(138.351288+0.001000)), 
    FL2FX(1.065324/sqrt(136.877670+0.001000)), FL2FX(0.716899/sqrt(95.163307+0.001000)), 
    FL2FX(1.156002/sqrt(168.703751+0.001000)), FL2FX(1.165472/sqrt(222.955536+0.001000)), 
    FL2FX(0.823618/sqrt(132.890594+0.001000)), FL2FX(0.986002/sqrt(204.590515+0.001000)), 
    FL2FX(1.094223/sqrt(112.443161+0.001000)), FL2FX(1.174787/sqrt(221.298584+0.001000)), 
    FL2FX(1.197138/sqrt(106.328888+0.001000)), FL2FX(1.267323/sqrt(196.683319+0.001000)), 
    FL2FX(1.213215/sqrt(221.121994+0.001000)), FL2FX(1.067222/sqrt(127.764091+0.001000)), 
    FL2FX(1.012151/sqrt(144.901215+0.001000)), FL2FX(1.131773/sqrt(99.541771+0.001000)), 
    FL2FX(1.392649/sqrt(234.369019+0.001000)), FL2FX(0.862833/sqrt(84.016113+0.001000)), 
    FL2FX(1.288343/sqrt(161.779037+0.001000)), FL2FX(1.138568/sqrt(130.305267+0.001000)), 
    FL2FX(1.174519/sqrt(234.478470+0.001000)), FL2FX(0.894942/sqrt(207.696442+0.001000)), 
    FL2FX(0.954437/sqrt(180.808304+0.001000)), FL2FX(0.893377/sqrt(170.323486+0.001000)), 
  
    };
    static const fixed std_beta[] ={
    FL2FX(-0.211499-10.696534*1.295200/sqrt(263.078888+0.001000)), FL2FX(-0.384732-11.479220*0.999835/sqrt(108.300644+0.001000)), 
    FL2FX(-0.124590-12.379833*0.928370/sqrt(110.086494+0.001000)), FL2FX(-0.513247-5.646174*1.100259/sqrt(110.728729+0.001000)), 
    FL2FX(-0.068103-17.579025*0.628929/sqrt(82.908058+0.001000)), FL2FX(-0.154945-14.926071*0.954274/sqrt(178.912460+0.001000)), 
    FL2FX(-0.134194-12.072823*1.248912/sqrt(184.492691+0.001000)), FL2FX(-0.101517-12.135567*1.001489/sqrt(121.794418+0.001000)), 
    FL2FX(-0.025662-10.771869*0.983482/sqrt(109.038193+0.001000)), FL2FX(-0.085382-10.558100*0.981253/sqrt(120.617516+0.001000)), 
    FL2FX(-0.435959-8.415052*0.980167/sqrt(84.543633+0.001000)), FL2FX(-0.055141-9.640590*1.273850/sqrt(127.494148+0.001000)), 
    FL2FX(-0.067280-15.139292*0.742787/sqrt(126.554062+0.001000)), FL2FX(-0.097742-29.442995*0.964583/sqrt(269.108826+0.001000)), 
    FL2FX(-0.017732-16.550129*1.010168/sqrt(166.777878+0.001000)), FL2FX(-0.037225-8.960438*0.954731/sqrt(117.137779+0.001000)), 
    FL2FX(-0.249117-10.165729*1.043302/sqrt(183.258667+0.001000)), FL2FX(-0.486413-10.517459*0.994408/sqrt(89.626068+0.001000)), 
    FL2FX(-0.029417-8.387836*1.261690/sqrt(100.348488+0.001000)), FL2FX(-0.183406-7.695740*0.914788/sqrt(75.082848+0.001000)), 
    FL2FX(-0.107977-10.833195*0.913166/sqrt(104.193855+0.001000)), FL2FX(-0.132531-15.953290*0.862743/sqrt(136.748276+0.001000)), 
    FL2FX(-0.016316-12.086588*1.211244/sqrt(162.954422+0.001000)), FL2FX(-0.256528-12.456125*0.761744/sqrt(121.508141+0.001000)), 
    FL2FX(-0.491272-7.685490*0.928655/sqrt(105.975197+0.001000)), FL2FX(0.213623-16.792463*1.181275/sqrt(172.891769+0.001000)), 
    FL2FX(-0.726308-13.831896*1.313107/sqrt(172.821014+0.001000)), FL2FX(-0.178738-17.384661*1.164546/sqrt(274.117188+0.001000)), 
    FL2FX(0.065202-21.421450*1.448877/sqrt(359.582153+0.001000)), FL2FX(-0.134839-15.433146*1.018143/sqrt(182.246429+0.001000)), 
    FL2FX(-0.068502-14.430947*0.949223/sqrt(148.197311+0.001000)), FL2FX(-0.232500-10.492564*0.928975/sqrt(103.434967+0.001000)), 
    FL2FX(-0.073822-7.832242*1.331212/sqrt(110.334663+0.001000)), FL2FX(-0.494005-13.327338*0.951596/sqrt(128.906631+0.001000)), 
    FL2FX(-0.108617-14.069035*0.934261/sqrt(154.258759+0.001000)), FL2FX(-0.127635-17.732241*0.974286/sqrt(176.088882+0.001000)), 
    FL2FX(-0.479671-7.675731*0.900589/sqrt(101.396660+0.001000)), FL2FX(-0.199342-7.978439*0.870046/sqrt(89.842018+0.001000)), 
    FL2FX(-0.276780-14.251707*0.901445/sqrt(143.257553+0.001000)), FL2FX(-0.205403-17.654732*1.026217/sqrt(177.710754+0.001000)), 
    FL2FX(-0.079633-14.469330*1.313180/sqrt(209.622787+0.001000)), FL2FX(-0.129782-17.768154*0.790559/sqrt(138.351288+0.001000)), 
    FL2FX(-0.391060-11.309340*1.065324/sqrt(136.877670+0.001000)), FL2FX(-0.185058-12.762525*0.716899/sqrt(95.163307+0.001000)), 
    FL2FX(-0.124943-12.772783*1.156002/sqrt(168.703751+0.001000)), FL2FX(-0.040870-16.211384*1.165472/sqrt(222.955536+0.001000)), 
    FL2FX(-0.269777-12.321926*0.823618/sqrt(132.890594+0.001000)), FL2FX(-0.063980-15.112982*0.986002/sqrt(204.590515+0.001000)), 
    FL2FX(-0.371016-6.814685*1.094223/sqrt(112.443161+0.001000)), FL2FX(-0.247785-20.065584*1.174787/sqrt(221.298584+0.001000)), 
    FL2FX(-0.149328-8.665216*1.197138/sqrt(106.328888+0.001000)), FL2FX(-0.406990-9.008690*1.267323/sqrt(196.683319+0.001000)), 
    FL2FX(-0.263530-12.709206*1.213215/sqrt(221.121994+0.001000)), FL2FX(-0.635054-10.412727*1.067222/sqrt(127.764091+0.001000)), 
    FL2FX(-0.332755-14.301723*1.012151/sqrt(144.901215+0.001000)), FL2FX(-0.305167-11.455599*1.131773/sqrt(99.541771+0.001000)), 
    FL2FX(-0.258473-15.572607*1.392649/sqrt(234.369019+0.001000)), FL2FX(-0.148282-11.226542*0.862833/sqrt(84.016113+0.001000)), 
    FL2FX(-0.402583-6.506679*1.288343/sqrt(161.779037+0.001000)), FL2FX(-0.388083-8.691804*1.138568/sqrt(130.305267+0.001000)), 
    FL2FX(0.046876-14.146768*1.174519/sqrt(234.478470+0.001000)), FL2FX(-0.323194-14.757297*0.894942/sqrt(207.696442+0.001000)), 
    FL2FX(-0.228616-13.752377*0.954437/sqrt(180.808304+0.001000)), FL2FX(-0.318303-14.120730*0.893377/sqrt(170.323486+0.001000)), 
  
    };

    static const batch_normalization_layer_t norm = { 64, inv_gamma_dev, std_beta };
    return norm;
}


          quantconv2d_layer_t init_quant_conv2d_5_data(void){

            static quant_filter_t filtros_b[64];
            
            static const uint32_t weights0[]={67112841,1263678770,2698173288,466060652,3021036699,2037039913,3084975826,537077444,2457363273,1705274551,1147351960,3739106069,2934407810,3396932203,931307512,905076101,2403211011,2748640658
            };
            static quant_filter_t filter0 = {64, 3, weights0, FL2FX(0.2222626507282257)};
            filtros_b[0]=filter0;
              
            static const uint32_t weights1[]={933538693,2879106274,1859796398,2189057694,1883915090,214235741,2470366282,669029525,4269483132,1712137622,3533918406,674180162,85814127,944447816,2235057075,18685122,1981283006,1351661079
            };
            static quant_filter_t filter1 = {64, 3, weights1, FL2FX(0.2462211400270462)};
            filtros_b[1]=filter1;
              
            static const uint32_t weights2[]={4220603212,4061647838,3753382729,662838937,1758160086,2101246684,1401984932,3140625718,1461818617,3835820863,3079349275,2269387696,1006795568,6778344,1064835115,124377572,3059302290,1584694661
            };
            static quant_filter_t filter2 = {64, 3, weights2, FL2FX(0.18215161561965942)};
            filtros_b[2]=filter2;
              
            static const uint32_t weights3[]={3762686404,193063360,4016198280,2406175248,233839497,730226826,1707812370,784372871,1750758630,3106642132,3958034773,1482049174,2549836458,1466476802,444654578,2437975123,219978763,617797311
            };
            static quant_filter_t filter3 = {64, 3, weights3, FL2FX(0.5528016686439514)};
            filtros_b[3]=filter3;
              
            static const uint32_t weights4[]={1614093452,2272150743,3887742963,391008417,1426533386,88239692,4020875246,1434913802,2775692068,2009599109,3152524115,2745333025,2566268871,990082058,798015540,3000068254,1329071495,1711657879
            };
            static quant_filter_t filter4 = {64, 3, weights4, FL2FX(0.35434266924858093)};
            filtros_b[4]=filter4;
              
            static const uint32_t weights5[]={3512739528,228385616,170388160,3302789896,2759891459,282604942,3492614815,1250498626,271281540,565188745,3429922999,4031600781,4172829832,4145602486,162842326,2995850295,83894464,157769336
            };
            static quant_filter_t filter5 = {64, 3, weights5, FL2FX(0.4072438180446625)};
            filtros_b[5]=filter5;
              
            static const uint32_t weights6[]={1635233732,1323401600,3470847876,262927954,3502819718,1292859618,2715888372,1871066917,1088388832,606935556,131297114,2573952406,770340570,1399518732,184629763,3985787148,3738223880,3761562866
            };
            static quant_filter_t filter6 = {64, 3, weights6, FL2FX(0.3050580620765686)};
            filtros_b[6]=filter6;
              
            static const uint32_t weights7[]={70756201,2226333745,3468314177,3851443857,580035842,4065857618,1526563621,3741868207,4231793915,1079883243,68701850,2545390620,2114601257,4093136065,3731365226,1278170057,113841824,3349492066
            };
            static quant_filter_t filter7 = {64, 3, weights7, FL2FX(0.2628178298473358)};
            filtros_b[7]=filter7;
              
            static const uint32_t weights8[]={551415269,2910022839,1055711114,3228417554,141241338,2351389692,293747237,2634035363,628248316,1688829591,327247980,2956436930,385112438,587742017,419434785,84068800,2530925966,3506238527
            };
            static quant_filter_t filter8 = {64, 3, weights8, FL2FX(0.36645638942718506)};
            filtros_b[8]=filter8;
              
            static const uint32_t weights9[]={1283396296,2258019891,2960798490,3691417900,1872255525,906202550,3570959109,3406462893,4006411661,2756497029,4099699320,530210343,847589619,1241269264,3091304602,610210560,2379811937,4101840182
            };
            static quant_filter_t filter9 = {64, 3, weights9, FL2FX(0.427118718624115)};
            filtros_b[9]=filter9;
              
            static const uint32_t weights10[]={39268448,3856992143,20913039,1345285765,1201614646,2192796144,277623069,2639887769,2395330593,797512796,3088780904,76018755,835153924,946671399,3898908963,238666528,583713966,3025713692
            };
            static quant_filter_t filter10 = {64, 3, weights10, FL2FX(0.3838764727115631)};
            filtros_b[10]=filter10;
              
            static const uint32_t weights11[]={674397228,122035325,2174293088,118286688,2053915049,113469737,3480011987,4075212073,4273234555,4216097635,674142066,2786374754,3950397422,3075893259,239296772,3640409755,495979078,115383572
            };
            static quant_filter_t filter11 = {64, 3, weights11, FL2FX(0.3384857177734375)};
            filtros_b[11]=filter11;
              
            static const uint32_t weights12[]={236332922,1163981649,1272448450,2468473989,3552314925,379071948,3036202267,2532870850,2617889042,1565548633,1844499106,4170783309,2326917148,1387630588,843269590,2995856494,1762140485,2170645547
            };
            static quant_filter_t filter12 = {64, 3, weights12, FL2FX(0.6653091311454773)};
            filtros_b[12]=filter12;
              
            static const uint32_t weights13[]={1703026656,575923959,2145883853,1583840256,2346911709,1023403992,154829183,3951332541,1670419990,4291558865,1034141284,3935492720,531897000,1747681752,3220327098,2617029834,3020516338,2631140619
            };
            static quant_filter_t filter13 = {64, 3, weights13, FL2FX(0.35733726620674133)};
            filtros_b[13]=filter13;
              
            static const uint32_t weights14[]={2600923952,214330140,964513494,438226704,2844169140,1620800345,3156148481,2878921877,3930203147,2332661816,1277053995,1946478274,1943624372,3595469301,1342775602,1332593408,1625602570,1216551469
            };
            static quant_filter_t filter14 = {64, 3, weights14, FL2FX(0.414245069026947)};
            filtros_b[14]=filter14;
              
            static const uint32_t weights15[]={2082848047,2788769976,2666538923,1466071276,1517185816,2551284470,3539766628,3003123880,3430169993,1950430413,2978852732,2693107746,199652131,1758752588,2088243496,641825232,1124031542,4094028036
            };
            static quant_filter_t filter15 = {64, 3, weights15, FL2FX(0.5816949605941772)};
            filtros_b[15]=filter15;
              
            static const uint32_t weights16[]={125960993,939769211,1507689305,1051862172,2452161971,1986163265,324147500,740155871,2464947163,4151174450,345646490,3858336477,3893379689,3099525625,2134965562,1911952844,2260955570,65013993
            };
            static quant_filter_t filter16 = {64, 3, weights16, FL2FX(0.15367096662521362)};
            filtros_b[16]=filter16;
              
            static const uint32_t weights17[]={3525607478,1251055918,2205568164,105550666,3032989967,680390877,1526799762,545300436,1390830950,2594584724,1327812980,1900342278,3348909576,2084413518,479414786,2438511775,2369002058,1122167466
            };
            static quant_filter_t filter17 = {64, 3, weights17, FL2FX(0.44560956954956055)};
            filtros_b[17]=filter17;
              
            static const uint32_t weights18[]={3366254228,204295534,3361499657,249367698,1557580944,994961923,2608063120,831000848,3225616484,1530010638,4211450555,1115104600,130604868,4236738623,3443169201,2179555472,4061345822,2112432742
            };
            static quant_filter_t filter18 = {64, 3, weights18, FL2FX(0.1979333907365799)};
            filtros_b[18]=filter18;
              
            static const uint32_t weights19[]={1982175597,1723062481,2301190912,3307694166,1321532165,3122610556,1016636261,3209708792,1221582043,2960505431,48122462,123852680,3300266937,785278469,3393192458,785265224,1620755758,1078566630
            };
            static quant_filter_t filter19 = {64, 3, weights19, FL2FX(0.3402906656265259)};
            filtros_b[19]=filter19;
              
            static const uint32_t weights20[]={591236453,845355425,2321522515,4172449495,3859285776,3996382800,1107053317,3204451503,1819069595,2754426231,2962951738,404617024,1370902961,203173825,2959614218,151683649,344968358,4098135725
            };
            static quant_filter_t filter20 = {64, 3, weights20, FL2FX(0.22537823021411896)};
            filtros_b[20]=filter20;
              
            static const uint32_t weights21[]={2988896165,2997541160,4221161105,2067527702,439624066,4105916362,2642726180,3150972246,824665506,3524375408,271314934,2788472528,2027146109,3080628320,1881151776,21893888,1564891189,3421939940
            };
            static quant_filter_t filter21 = {64, 3, weights21, FL2FX(0.460734486579895)};
            filtros_b[21]=filter21;
              
            static const uint32_t weights22[]={1719731648,213377462,383961996,499358800,513416206,3542125266,1407400748,957297847,9989156,3904499353,1423068825,992698141,698962160,1509278842,3137969826,1686246344,2898844928,4096603240
            };
            static quant_filter_t filter22 = {64, 3, weights22, FL2FX(0.5295552015304565)};
            filtros_b[22]=filter22;
              
            static const uint32_t weights23[]={540300676,735972288,1215976156,417143935,1884836785,692302877,771377166,785827536,1882936908,4177759188,3268100631,676356310,2518031310,389163852,559104018,3544753270,437824582,248174499
            };
            static quant_filter_t filter23 = {64, 3, weights23, FL2FX(0.24541421234607697)};
            filtros_b[23]=filter23;
              
            static const uint32_t weights24[]={572334560,3815706792,4094893867,3494123029,372522769,3307432758,1925720865,3173018031,767343673,781524035,2283765288,926398242,702968243,1063529825,3965792300,738677681,2395653282,1981060118
            };
            static quant_filter_t filter24 = {64, 3, weights24, FL2FX(0.41219499707221985)};
            filtros_b[24]=filter24;
              
            static const uint32_t weights25[]={763388548,1103850560,1692194519,2465326986,828637419,679832037,2374172298,250856465,2493019010,3374189333,983945344,1074398424,1710301260,814824154,873326869,1939146319,1829546245,586349196
            };
            static quant_filter_t filter25 = {64, 3, weights25, FL2FX(0.5158047080039978)};
            filtros_b[25]=filter25;
              
            static const uint32_t weights26[]={1871855982,3510351495,3070069057,4283672140,1335960706,2130852206,3750901398,2462654404,2623017875,4236410399,2938381419,3515618248,2046945444,1782044397,4280323512,2276536500,201867203,3120953549
            };
            static quant_filter_t filter26 = {64, 3, weights26, FL2FX(0.48632562160491943)};
            filtros_b[26]=filter26;
              
            static const uint32_t weights27[]={3657464743,172052222,2331797855,227244642,2631876662,193453020,2911989054,2789348949,1257373898,2141206540,130539590,1044676678,1163552908,1316993679,461906729,153933928,991239050,551210500
            };
            static quant_filter_t filter27 = {64, 3, weights27, FL2FX(0.32913458347320557)};
            filtros_b[27]=filter27;
              
            static const uint32_t weights28[]={195543265,1898081096,1873121219,2056535729,2569480112,3827552761,4064709121,524812431,1615641345,3629855856,282742088,3099244379,601026676,4096794601,644189609,1664160416,2228023731,4292349596
            };
            static quant_filter_t filter28 = {64, 3, weights28, FL2FX(0.29563426971435547)};
            filtros_b[28]=filter28;
              
            static const uint32_t weights29[]={3319987459,2935426375,2402553726,525254724,1832441060,1011879488,1473982180,1649544396,1418445200,4047510275,4182172451,2173113264,446939109,3804173956,1060179120,587243556,1849177498,2869059496
            };
            static quant_filter_t filter29 = {64, 3, weights29, FL2FX(0.4234777092933655)};
            filtros_b[29]=filter29;
              
            static const uint32_t weights30[]={3382667709,2236092150,653054912,756103828,1772824964,1930570040,3601938103,331774594,4096822188,3989051928,757589519,647109188,442759477,4193696,1059261732,2336295128,1312701824,2711570724
            };
            static quant_filter_t filter30 = {64, 3, weights30, FL2FX(0.17939355969429016)};
            filtros_b[30]=filter30;
              
            static const uint32_t weights31[]={1239963656,2626330469,2157765240,790606579,1773194642,1440630345,2204292852,2067841963,1899193316,4176759074,3870766994,1994534520,3694055406,2193836216,226522791,3131932307,722478722,253834704
            };
            static quant_filter_t filter31 = {64, 3, weights31, FL2FX(0.39321380853652954)};
            filtros_b[31]=filter31;
              
            static const uint32_t weights32[]={4099981263,179782064,2808112008,1321867364,3202011413,2837957308,1969045477,764089732,1119497452,3760324036,2546448728,2052541347,105584266,58389510,2585801697,630331680,785423546,1825690423
            };
            static quant_filter_t filter32 = {64, 3, weights32, FL2FX(0.4484628736972809)};
            filtros_b[32]=filter32;
              
            static const uint32_t weights33[]={1683559564,40764720,3836883720,2396788763,1691991649,302233873,3889869521,1327983617,2995927012,1821707654,3574966992,2048003619,2303595618,924347455,496808608,3365081778,3080242891,4107899162
            };
            static quant_filter_t filter33 = {64, 3, weights33, FL2FX(0.26786521077156067)};
            filtros_b[33]=filter33;
              
            static const uint32_t weights34[]={170019204,155042276,3635330649,496709708,1815690290,782462662,3348264614,973595004,3503441853,4237035908,1352886096,862862982,2470699497,2717594732,2064834688,4133911955,534711298,1333890976
            };
            static quant_filter_t filter34 = {64, 3, weights34, FL2FX(0.3944633901119232)};
            filtros_b[34]=filter34;
              
            static const uint32_t weights35[]={2824081908,2194762934,2787857055,3739651309,526184733,2845435708,3699944743,2873134612,3898417167,2919235796,1764481026,2621896326,3545702563,1207674115,4175708426,990582290,463996655,2757751345
            };
            static quant_filter_t filter35 = {64, 3, weights35, FL2FX(0.5904305577278137)};
            filtros_b[35]=filter35;
              
            static const uint32_t weights36[]={1881774861,34575230,2159162808,508221490,836885250,2127970888,4009790176,1839525709,590512103,4182345862,1619207038,397813536,2769700611,927361028,1066557444,3863831194,318542850,3700006801
            };
            static quant_filter_t filter36 = {64, 3, weights36, FL2FX(0.2172268182039261)};
            filtros_b[36]=filter36;
              
            static const uint32_t weights37[]={3716723227,1854222261,843606354,1160342352,2413547940,2585782517,2572396296,4024073797,2700038241,614594589,663931608,3603713864,1567407764,4128228767,470355754,1300939584,1689299209,2562434308
            };
            static quant_filter_t filter37 = {64, 3, weights37, FL2FX(0.1776464879512787)};
            filtros_b[37]=filter37;
              
            static const uint32_t weights38[]={4180951254,1162403273,2661870976,3706793152,963049997,781150604,2914168338,952403844,1357466310,4222887957,2746467190,2691446932,392670013,439792428,693256336,2437425308,1176411283,3230856869
            };
            static quant_filter_t filter38 = {64, 3, weights38, FL2FX(0.5699522495269775)};
            filtros_b[38]=filter38;
              
            static const uint32_t weights39[]={454446852,2183986124,3404518360,948138195,964454870,1871177710,1737673772,959874422,1997163480,3752397586,8602975,393225874,1348974857,925838021,859003267,1600948721,2656581782,1318356133
            };
            static quant_filter_t filter39 = {64, 3, weights39, FL2FX(0.6174395680427551)};
            filtros_b[39]=filter39;
              
            static const uint32_t weights40[]={806964005,53252172,4239923880,948934774,959407096,786526909,2140204114,803505828,3812699732,3004034837,19624751,774154002,7834528,642200395,1729315092,4100796635,504897478,3525096965
            };
            static quant_filter_t filter40 = {64, 3, weights40, FL2FX(0.34128594398498535)};
            filtros_b[40]=filter40;
              
            static const uint32_t weights41[]={1333904896,956058368,1330951634,2016532113,3517064358,4061262043,3018215944,295642003,650674080,3356196920,282652045,688486639,888150105,1941964794,1932376115,3266372068,2554913025,1031361373
            };
            static quant_filter_t filter41 = {64, 3, weights41, FL2FX(0.38788408041000366)};
            filtros_b[41]=filter41;
              
            static const uint32_t weights42[]={2232559376,1147303550,818189665,411066488,1497309514,346291042,664822834,378216838,2709520921,3688378651,2569451073,792213021,199401825,2209285991,1997767972,4152166580,2403867648,4091297346
            };
            static quant_filter_t filter42 = {64, 3, weights42, FL2FX(0.35060226917266846)};
            filtros_b[42]=filter42;
              
            static const uint32_t weights43[]={691756357,444848633,3900817164,3123060787,443836432,3834367806,1696881984,2030314607,2802162138,2586390880,845936602,527448210,1707492315,3128338953,577794560,879363465,2480148529,4011108497
            };
            static quant_filter_t filter43 = {64, 3, weights43, FL2FX(0.32361024618148804)};
            filtros_b[43]=filter43;
              
            static const uint32_t weights44[]={3241193848,2267232795,2608344642,3559869248,1401881445,349720478,2620202255,3470826568,4263889051,936685387,762115187,3431846560,2562688226,101533298,616582732,2434364475,251662700,2355572050
            };
            static quant_filter_t filter44 = {64, 3, weights44, FL2FX(0.6255865097045898)};
            filtros_b[44]=filter44;
              
            static const uint32_t weights45[]={3794899839,1193965660,2707522968,3714194512,1724662017,3994665340,3179355713,1824927644,5000194,3127452245,152520479,633353890,2071442386,1064072005,3999472256,642726800,2347795591,1813879205
            };
            static quant_filter_t filter45 = {64, 3, weights45, FL2FX(0.36750444769859314)};
            filtros_b[45]=filter45;
              
            static const uint32_t weights46[]={10377814,1250043662,3850687441,973496355,2569283062,615704237,2620542480,548472356,130238800,2603164209,2579949051,969408659,123992072,3122013749,809023910,3849425357,4266468123,3949907100
            };
            static quant_filter_t filter46 = {64, 3, weights46, FL2FX(0.4920775294303894)};
            filtros_b[46]=filter46;
              
            static const uint32_t weights47[]={1058326689,2513565945,1451274818,4160040165,455946630,1980322158,4223996966,2479096778,220587795,2052614011,153701534,2729819996,2295524212,2390916687,1863322988,1883517136,101357360,184821957
            };
            static quant_filter_t filter47 = {64, 3, weights47, FL2FX(0.2482643723487854)};
            filtros_b[47]=filter47;
              
            static const uint32_t weights48[]={1817906125,2353522135,3888131020,1850080276,1324054402,534135602,3284737765,1598884107,2717171388,1993415049,2875700058,867311498,443987220,249259272,497190440,2762832051,647500977,2444310118
            };
            static quant_filter_t filter48 = {64, 3, weights48, FL2FX(0.5241041779518127)};
            filtros_b[48]=filter48;
              
            static const uint32_t weights49[]={3830000972,260515729,476731008,373706218,1473885713,975083024,3618188128,1824484541,2487001518,3982856344,1153767989,2985755065,3106948227,298413142,2853019202,655779170,1306004483,1973781490
            };
            static quant_filter_t filter49 = {64, 3, weights49, FL2FX(0.37110888957977295)};
            filtros_b[49]=filter49;
              
            static const uint32_t weights50[]={226524032,299860849,1393217715,956104783,3644367102,124827208,1626262636,1366491432,934537044,3451928960,1893884837,953056285,278055117,2474704918,868079063,3280070774,2046828884,1533199763
            };
            static quant_filter_t filter50 = {64, 3, weights50, FL2FX(0.48096731305122375)};
            filtros_b[50]=filter50;
              
            static const uint32_t weights51[]={577296677,2456884396,2566666586,899085897,1882672443,1805731533,3450884964,2731983940,1516580715,3833636150,1084981075,239975798,4251369889,728575712,727934976,3708378442,396043910,1107355801
            };
            static quant_filter_t filter51 = {64, 3, weights51, FL2FX(0.22812868654727936)};
            filtros_b[51]=filter51;
              
            static const uint32_t weights52[]={808866085,188783981,1356612675,1599187,2827252215,13466181,1864679435,651596935,2003766387,320441911,3297401570,2758731344,3306079012,908209405,37244322,1500354267,1360402958,1088090228
            };
            static quant_filter_t filter52 = {64, 3, weights52, FL2FX(0.1918255090713501)};
            filtros_b[52]=filter52;
              
            static const uint32_t weights53[]={1154913480,203182001,2520169189,85754691,3385372066,2626456641,1196148352,3452500489,3105034853,885214852,1181056720,3550897608,3830259398,982729786,210006627,1821361715,826287758,3300463376
            };
            static quant_filter_t filter53 = {64, 3, weights53, FL2FX(0.44978129863739014)};
            filtros_b[53]=filter53;
              
            static const uint32_t weights54[]={28908226,1433371238,2540614065,957183106,1950305274,1127448131,1083339320,1362049836,3046268804,2135425025,1346545511,2702844322,660795507,4029682488,966234404,1723955484,2589516946,4228324628
            };
            static quant_filter_t filter54 = {64, 3, weights54, FL2FX(0.3402476906776428)};
            filtros_b[54]=filter54;
              
            static const uint32_t weights55[]={689527269,2244226966,3565687673,43266167,771714194,910072696,4022527734,2003378984,3980204872,2497779986,196553290,602801526,828782453,443245065,3980142117,2892456944,1922602116,1928696980
            };
            static quant_filter_t filter55 = {64, 3, weights55, FL2FX(0.3278887867927551)};
            filtros_b[55]=filter55;
              
            static const uint32_t weights56[]={2334273826,1252154665,151610372,3225152274,3669872940,2215570149,2943418641,2799359988,1267096211,1211114838,613738642,2490155093,1842999469,3033797209,1346579330,1227784713,93356328,3305270348
            };
            static quant_filter_t filter56 = {64, 3, weights56, FL2FX(0.4342769682407379)};
            filtros_b[56]=filter56;
              
            static const uint32_t weights57[]={617842580,2728347494,2491341720,2632424666,765193912,3910219276,1442328116,1024861836,651707916,2917419668,3263417125,324431762,3036877651,1664820519,3104247858,626501924,2331561171,3054154658
            };
            static quant_filter_t filter57 = {64, 3, weights57, FL2FX(0.4416690170764923)};
            filtros_b[57]=filter57;
              
            static const uint32_t weights58[]={542497645,2790147008,3484572608,2766477347,1710290576,1780446604,266899557,4292905739,956288630,2219642322,1120030454,952358134,511156063,1065381512,156385493,2421204147,1052542084,82301317
            };
            static quant_filter_t filter58 = {64, 3, weights58, FL2FX(0.3743983507156372)};
            filtros_b[58]=filter58;
              
            static const uint32_t weights59[]={1284667008,2640236517,2989993543,1029489418,2833317029,1422691974,3275478234,1246292353,274338692,738347172,1898443312,2195085358,3934889167,3287499964,2374399574,2461824691,620766936,86069609
            };
            static quant_filter_t filter59 = {64, 3, weights59, FL2FX(0.4047912061214447)};
            filtros_b[59]=filter59;
              
            static const uint32_t weights60[]={3788563749,3117613505,1201262784,1660190259,2566119812,4106096860,1046465765,2445410950,1830258137,280078294,517883823,3368127829,2343779085,114320769,627056430,562403488,3859313193,1540735198
            };
            static quant_filter_t filter60 = {64, 3, weights60, FL2FX(0.47524142265319824)};
            filtros_b[60]=filter60;
              
            static const uint32_t weights61[]={3429404969,226989820,3117120530,1345301014,2612861732,2531835803,1894517028,3641671721,732420399,1536499928,3561597653,1695323273,2207286280,3857515793,546621006,4041094163,746738240,2435266146
            };
            static quant_filter_t filter61 = {64, 3, weights61, FL2FX(0.47586143016815186)};
            filtros_b[61]=filter61;
              
            static const uint32_t weights62[]={3518765591,207813743,3566757912,2660396636,1054956121,1011854310,3204745026,746883028,1125196750,4084497426,2217153366,2272012176,2368652097,3727159396,4231532440,291923138,1430133459,3813144272
            };
            static quant_filter_t filter62 = {64, 3, weights62, FL2FX(0.1815388798713684)};
            filtros_b[62]=filter62;
              
            static const uint32_t weights63[]={3055315863,3280796103,2599208793,764634628,166410256,1762482436,3750770230,290199542,2304805002,4246870121,4087563302,927218834,499376509,1768406756,1059212424,2439178548,90054551,3943160521
            };
            static quant_filter_t filter63 = {64, 3, weights63, FL2FX(0.32161250710487366)};
            filtros_b[63]=filter63;
              
            quantconv2d_layer_t layer = {64,filtros_b};
            return layer;
          }
            
batch_normalization_layer_t init_batch_normalization_14_data(void){

    static const fixed inv_gamma_dev[] ={
    FL2FX(1.033030/sqrt(957.767517+0.001000)), FL2FX(0.947880/sqrt(803.977844+0.001000)), 
    FL2FX(1.038821/sqrt(783.815430+0.001000)), FL2FX(0.868097/sqrt(637.112915+0.001000)), 
    FL2FX(0.872260/sqrt(623.454468+0.001000)), FL2FX(0.900680/sqrt(695.170654+0.001000)), 
    FL2FX(0.824867/sqrt(708.900330+0.001000)), FL2FX(0.964929/sqrt(914.378784+0.001000)), 
    FL2FX(0.992782/sqrt(940.932068+0.001000)), FL2FX(0.885444/sqrt(710.696716+0.001000)), 
    FL2FX(0.847712/sqrt(603.125671+0.001000)), FL2FX(0.829073/sqrt(618.732910+0.001000)), 
    FL2FX(0.930322/sqrt(543.470154+0.001000)), FL2FX(0.863643/sqrt(603.951416+0.001000)), 
    FL2FX(0.846512/sqrt(652.655090+0.001000)), FL2FX(0.852897/sqrt(626.024597+0.001000)), 
    FL2FX(1.126382/sqrt(1121.918457+0.001000)), FL2FX(0.917689/sqrt(754.710815+0.001000)), 
    FL2FX(0.958402/sqrt(848.532776+0.001000)), FL2FX(0.994443/sqrt(728.901855+0.001000)), 
    FL2FX(0.994639/sqrt(1002.790039+0.001000)), FL2FX(0.982703/sqrt(1006.575989+0.001000)), 
    FL2FX(0.999779/sqrt(725.353333+0.001000)), FL2FX(0.846211/sqrt(672.260254+0.001000)), 
    FL2FX(0.888796/sqrt(630.133240+0.001000)), FL2FX(0.787079/sqrt(467.966736+0.001000)), 
    FL2FX(1.032632/sqrt(899.171143+0.001000)), FL2FX(1.004985/sqrt(982.184814+0.001000)), 
    FL2FX(1.064953/sqrt(965.403015+0.001000)), FL2FX(1.058952/sqrt(731.957458+0.001000)), 
    FL2FX(1.009961/sqrt(700.166321+0.001000)), FL2FX(0.813721/sqrt(604.493530+0.001000)), 
    FL2FX(0.945724/sqrt(725.443665+0.001000)), FL2FX(0.914633/sqrt(805.091492+0.001000)), 
    FL2FX(0.948392/sqrt(727.879639+0.001000)), FL2FX(0.854778/sqrt(572.394531+0.001000)), 
    FL2FX(0.897681/sqrt(811.261719+0.001000)), FL2FX(0.927177/sqrt(686.499695+0.001000)), 
    FL2FX(0.927125/sqrt(622.731934+0.001000)), FL2FX(0.968871/sqrt(655.298645+0.001000)), 
    FL2FX(0.918692/sqrt(696.705688+0.001000)), FL2FX(0.924166/sqrt(636.399109+0.001000)), 
    FL2FX(0.988626/sqrt(960.905334+0.001000)), FL2FX(0.942774/sqrt(808.912659+0.001000)), 
    FL2FX(0.977423/sqrt(579.672119+0.001000)), FL2FX(0.861274/sqrt(816.628723+0.001000)), 
    FL2FX(0.856273/sqrt(598.989624+0.001000)), FL2FX(0.938457/sqrt(883.363037+0.001000)), 
    FL2FX(0.889670/sqrt(688.846436+0.001000)), FL2FX(1.039765/sqrt(891.169434+0.001000)), 
    FL2FX(0.967161/sqrt(709.746216+0.001000)), FL2FX(0.947265/sqrt(702.405518+0.001000)), 
    FL2FX(0.821559/sqrt(724.966492+0.001000)), FL2FX(0.821677/sqrt(692.283691+0.001000)), 
    FL2FX(0.934611/sqrt(619.106750+0.001000)), FL2FX(0.941293/sqrt(670.784790+0.001000)), 
    FL2FX(0.897501/sqrt(832.199890+0.001000)), FL2FX(0.991754/sqrt(689.448669+0.001000)), 
    FL2FX(0.870264/sqrt(701.802185+0.001000)), FL2FX(0.987314/sqrt(621.542297+0.001000)), 
    FL2FX(1.004855/sqrt(911.364380+0.001000)), FL2FX(0.906746/sqrt(593.909119+0.001000)), 
    FL2FX(1.018161/sqrt(834.273987+0.001000)), FL2FX(1.002800/sqrt(751.638489+0.001000)), 
  
    };
    static const fixed std_beta[] ={
    FL2FX(0.094591-32.517567*1.033030/sqrt(957.767517+0.001000)), FL2FX(0.196151-32.480602*0.947880/sqrt(803.977844+0.001000)), 
    FL2FX(-0.047353-23.270102*1.038821/sqrt(783.815430+0.001000)), FL2FX(0.205661-33.301003*0.868097/sqrt(637.112915+0.001000)), 
    FL2FX(0.056161-28.662191*0.872260/sqrt(623.454468+0.001000)), FL2FX(0.154002-28.255241*0.900680/sqrt(695.170654+0.001000)), 
    FL2FX(0.005911-26.013172*0.824867/sqrt(708.900330+0.001000)), FL2FX(0.169133-37.491985*0.964929/sqrt(914.378784+0.001000)), 
    FL2FX(0.137819-31.321476*0.992782/sqrt(940.932068+0.001000)), FL2FX(0.171487-29.578537*0.885444/sqrt(710.696716+0.001000)), 
    FL2FX(0.091593-26.581547*0.847712/sqrt(603.125671+0.001000)), FL2FX(0.178270-29.520060*0.829073/sqrt(618.732910+0.001000)), 
    FL2FX(0.225118-27.534529*0.930322/sqrt(543.470154+0.001000)), FL2FX(0.125998-19.667707*0.863643/sqrt(603.951416+0.001000)), 
    FL2FX(0.187558-32.218727*0.846512/sqrt(652.655090+0.001000)), FL2FX(0.038041-28.346365*0.852897/sqrt(626.024597+0.001000)), 
    FL2FX(0.031550-27.937334*1.126382/sqrt(1121.918457+0.001000)), FL2FX(0.226217-32.659557*0.917689/sqrt(754.710815+0.001000)), 
    FL2FX(0.182513-37.133926*0.958402/sqrt(848.532776+0.001000)), FL2FX(0.148776-28.738218*0.994443/sqrt(728.901855+0.001000)), 
    FL2FX(0.101702-37.471874*0.994639/sqrt(1002.790039+0.001000)), FL2FX(0.207880-35.845421*0.982703/sqrt(1006.575989+0.001000)), 
    FL2FX(0.355132-34.313400*0.999779/sqrt(725.353333+0.001000)), FL2FX(0.112866-35.499931*0.846211/sqrt(672.260254+0.001000)), 
    FL2FX(0.070323-27.749334*0.888796/sqrt(630.133240+0.001000)), FL2FX(0.152369-29.000645*0.787079/sqrt(467.966736+0.001000)), 
    FL2FX(0.174876-31.934917*1.032632/sqrt(899.171143+0.001000)), FL2FX(0.123707-31.709917*1.004985/sqrt(982.184814+0.001000)), 
    FL2FX(0.288450-33.192360*1.064953/sqrt(965.403015+0.001000)), FL2FX(0.154620-27.100018*1.058952/sqrt(731.957458+0.001000)), 
    FL2FX(0.038481-24.443768*1.009961/sqrt(700.166321+0.001000)), FL2FX(0.063531-26.330347*0.813721/sqrt(604.493530+0.001000)), 
    FL2FX(0.025588-26.025494*0.945724/sqrt(725.443665+0.001000)), FL2FX(0.169577-33.593914*0.914633/sqrt(805.091492+0.001000)), 
    FL2FX(0.097310-29.697077*0.948392/sqrt(727.879639+0.001000)), FL2FX(0.191817-30.602968*0.854778/sqrt(572.394531+0.001000)), 
    FL2FX(0.155157-36.362553*0.897681/sqrt(811.261719+0.001000)), FL2FX(-0.011594-23.157209*0.927177/sqrt(686.499695+0.001000)), 
    FL2FX(0.230093-26.571438*0.927125/sqrt(622.731934+0.001000)), FL2FX(0.190061-28.336048*0.968871/sqrt(655.298645+0.001000)), 
    FL2FX(0.182127-32.627552*0.918692/sqrt(696.705688+0.001000)), FL2FX(0.135239-27.954987*0.924166/sqrt(636.399109+0.001000)), 
    FL2FX(0.330364-35.881557*0.988626/sqrt(960.905334+0.001000)), FL2FX(0.117537-30.757889*0.942774/sqrt(808.912659+0.001000)), 
    FL2FX(0.201435-24.793514*0.977423/sqrt(579.672119+0.001000)), FL2FX(0.085214-33.596142*0.861274/sqrt(816.628723+0.001000)), 
    FL2FX(0.070259-26.896624*0.856273/sqrt(598.989624+0.001000)), FL2FX(0.134644-35.180706*0.938457/sqrt(883.363037+0.001000)), 
    FL2FX(0.043688-25.503748*0.889670/sqrt(688.846436+0.001000)), FL2FX(0.230358-33.090485*1.039765/sqrt(891.169434+0.001000)), 
    FL2FX(0.175877-27.741859*0.967161/sqrt(709.746216+0.001000)), FL2FX(0.080028-27.401051*0.947265/sqrt(702.405518+0.001000)), 
    FL2FX(0.170991-34.987137*0.821559/sqrt(724.966492+0.001000)), FL2FX(0.196716-37.579712*0.821677/sqrt(692.283691+0.001000)), 
    FL2FX(0.081406-22.586851*0.934611/sqrt(619.106750+0.001000)), FL2FX(0.092307-28.942303*0.941293/sqrt(670.784790+0.001000)), 
    FL2FX(0.145592-34.095261*0.897501/sqrt(832.199890+0.001000)), FL2FX(0.143256-28.367804*0.991754/sqrt(689.448669+0.001000)), 
    FL2FX(0.037257-26.265072*0.870264/sqrt(701.802185+0.001000)), FL2FX(0.182544-26.792944*0.987314/sqrt(621.542297+0.001000)), 
    FL2FX(0.220724-32.341431*1.004855/sqrt(911.364380+0.001000)), FL2FX(0.142595-29.029699*0.906746/sqrt(593.909119+0.001000)), 
    FL2FX(0.123108-28.781618*1.018161/sqrt(834.273987+0.001000)), FL2FX(0.017607-22.413507*1.002800/sqrt(751.638489+0.001000)), 
  
    };

    static const batch_normalization_layer_t norm = { 64, inv_gamma_dev, std_beta };
    return norm;
}

    quantdense_layer_t init_quant_dense_1_data(void){

        static quant_neuron_t neurons[64];
    
        static  uint32_t weights0[] ={
    109590130, 589922627, 931519962, 2682001515, 425587255, 3431847194, 915231031, 4094214896, 
    722391094, 2521137453, 310776374, 2222374923, 2056204630, 1250944620, 684424246, 
    2521465706, 2058166910, 2277696459
    };
        static const quant_neuron_t neuron0 = {weights0, FL2FX(0.09039193391799927)};
        neurons[0]=neuron0;
    
        static  uint32_t weights1[] ={
    3699080008, 3737943143, 3582648438, 3290468248, 589995481, 88288680, 3587828704, 
    3330898105, 1965876074, 382553128, 2005081770, 348886760, 3922640828, 159481195, 
    1774173226, 130385448, 1407688328, 2228322729
    };
        static const quant_neuron_t neuron1 = {weights1, FL2FX(0.011898391880095005)};
        neurons[1]=neuron1;
    
        static  uint32_t weights2[] ={
    1951764741, 1029131260, 234890255, 2862867326, 856755457, 2371809722, 402536845, 
    1875701617, 182084699, 2474516450, 621829133, 98251561, 334642825, 1276170148, 846451856, 
    2579434771, 629456521, 604093758
    };
        static const quant_neuron_t neuron2 = {weights2, FL2FX(0.024028191342949867)};
        neurons[2]=neuron2;
    
        static  uint32_t weights3[] ={
    227029608, 36275255, 4085511536, 4292311151, 257643047, 3692875318, 2289250485, 
    3225375273, 1043593045, 1840402922, 2364560449, 2084453806, 2151382172, 264182150, 
    491517460, 3141470134, 267542164, 1555877766
    };
        static const quant_neuron_t neuron3 = {weights3, FL2FX(0.16251471638679504)};
        neurons[3]=neuron3;
    
        static  uint32_t weights4[] ={
    3353776010, 3663883626, 149097932, 964633409, 4242354548, 4219837940, 2057449371, 
    3453263208, 2993823376, 305398434, 1482659690, 344295571, 67082971, 1431608480, 
    2617262913, 4027098997, 828349606, 815650957
    };
        static const quant_neuron_t neuron4 = {weights4, FL2FX(-0.055098842829465866)};
        neurons[4]=neuron4;
    
        static  uint32_t weights5[] ={
    2504822277, 559571652, 3380380911, 2046505840, 3673552209, 4254350998, 4016681761, 
    3305226263, 3901796599, 4147311955, 135984413, 2906012822, 3471439540, 3997451223, 
    2974868077, 4112776666, 622509437, 290289821
    };
        static const quant_neuron_t neuron5 = {weights5, FL2FX(0.0012251456500962377)};
        neurons[5]=neuron5;
    
        static  uint32_t weights6[] ={
    2892679272, 369177655, 3578733380, 3563613916, 2638219624, 1900667965, 1830856828, 
    3528405501, 3347679645, 1078066892, 3779172744, 3498139980, 2033625338, 495824136, 
    545689129, 350295562, 811710600, 3235087784
    };
        static const quant_neuron_t neuron6 = {weights6, FL2FX(0.12193215638399124)};
        neurons[6]=neuron6;
    
        static  uint32_t weights7[] ={
    3053565495, 4273324299, 3138713446, 3149250923, 676848167, 4236105042, 518403928, 
    3874851292, 862648368, 340713945, 1004075967, 3748058231, 66206940, 100601216, 3341311168, 
    287050980, 333575808, 3558628742
    };
        static const quant_neuron_t neuron7 = {weights7, FL2FX(0.11852763593196869)};
        neurons[7]=neuron7;
    
        static  uint32_t weights8[] ={
    1593727309, 475952748, 2263491202, 3293218709, 1384676728, 594745867, 3739487062, 
    3905486444, 450960775, 2412422077, 379481463, 1665675817, 1608772492, 3359668078, 
    3941555415, 1576182700, 1511940471, 1733204139
    };
        static const quant_neuron_t neuron8 = {weights8, FL2FX(0.2430250197649002)};
        neurons[8]=neuron8;
    
        static  uint32_t weights9[] ={
    4277489108, 3751190017, 2238425663, 1852413590, 3573174222, 398080207, 2790554000, 
    599565218, 2656593047, 3871131683, 224132101, 3158167038, 2690212979, 1002278295, 
    2775732419, 1385971039, 85127040, 409748622
    };
        static const quant_neuron_t neuron9 = {weights9, FL2FX(0.1534280627965927)};
        neurons[9]=neuron9;
    
        static  uint32_t weights10[] ={
    2058314358, 916914387, 45985470, 2544735113, 2221735548, 3614220829, 3515252787, 
    3585584459, 720836246, 1017383782, 284668534, 1147324177, 1498932241, 2621360150, 
    361125765, 4280082733, 3135611030, 3819619155
    };
        static const quant_neuron_t neuron10 = {weights10, FL2FX(0.19545163214206696)};
        neurons[10]=neuron10;
    
        static  uint32_t weights11[] ={
    3788763115, 1895766762, 3455688682, 1778217668, 4193361918, 4223115217, 2713634010, 
    1181814954, 3231507728, 1558194624, 2581109484, 3652358498, 2350662996, 653867187, 
    3649922846, 2998935583, 4282430404, 3933365115
    };
        static const quant_neuron_t neuron11 = {weights11, FL2FX(-0.05059526115655899)};
        neurons[11]=neuron11;
    
        static  uint32_t weights12[] ={
    654879527, 728330398, 4178706252, 1505719401, 2949771785, 1553583548, 2486731647, 
    770183589, 3381653758, 3769727191, 2367478400, 1223812524, 602128042, 2373260744, 
    654743688, 2169851906, 2238995584, 2483575956
    };
        static const quant_neuron_t neuron12 = {weights12, FL2FX(0.01764063723385334)};
        neurons[12]=neuron12;
    
        static  uint32_t weights13[] ={
    866283151, 23305676, 3019692506, 2811738593, 3177036415, 1812842044, 1671675530, 
    267885441, 4056253288, 4116297041, 3011500950, 965333155, 48996044, 802599841, 4261700438, 
    3793856018, 3682424724, 3166435757
    };
        static const quant_neuron_t neuron13 = {weights13, FL2FX(-0.13819189369678497)};
        neurons[13]=neuron13;
    
        static  uint32_t weights14[] ={
    4170488298, 563288171, 692396199, 2063895859, 570522038, 1066799683, 3775210294, 
    2777102915, 4174227759, 2046922867, 4064786471, 2808382019, 1741254711, 2255040379, 
    2038905918, 1315301707, 1789341975, 2406869859
    };
        static const quant_neuron_t neuron14 = {weights14, FL2FX(-0.18090808391571045)};
        neurons[14]=neuron14;
    
        static  uint32_t weights15[] ={
    3376056457, 1236495788, 1801591045, 3915001784, 3987755351, 1136824313, 3308502632, 
    197462447, 4284123932, 4001070812, 4188626414, 3954023675, 1138990056, 2932167400, 
    2538900429, 2086755832, 1505467742, 2610764200
    };
        static const quant_neuron_t neuron15 = {weights15, FL2FX(0.09796720743179321)};
        neurons[15]=neuron15;
    
        static  uint32_t weights16[] ={
    3374225101, 262138798, 167785861, 3898265599, 2061835711, 3206337385, 389863745, 
    1068918375, 519247028, 1052738371, 1807814039, 3613210477, 400432584, 2019331764, 
    1929076616, 2447198446, 1874927277, 2421458744
    };
        static const quant_neuron_t neuron16 = {weights16, FL2FX(0.31948718428611755)};
        neurons[16]=neuron16;
    
        static  uint32_t weights17[] ={
    4279566128, 3466174408, 3876971016, 3223966878, 864718392, 825875855, 4044881464, 
    1236393382, 2743903241, 200514846, 2201458312, 2448686380, 899581715, 988928006, 
    2701838187, 603417399, 2169543881, 2574409868
    };
        static const quant_neuron_t neuron17 = {weights17, FL2FX(-0.20294636487960815)};
        neurons[17]=neuron17;
    
        static  uint32_t weights18[] ={
    1249247680, 1214139808, 1453514768, 3288861844, 397057270, 892012697, 1258250725, 
    3485338542, 682181215, 386770896, 4293500474, 3118193355, 1493061991, 1604873214, 
    514466174, 2678820769, 1051456607, 125772555
    };
        static const quant_neuron_t neuron18 = {weights18, FL2FX(0.16215567290782928)};
        neurons[18]=neuron18;
    
        static  uint32_t weights19[] ={
    3313197167, 2418738194, 3757731800, 2531158726, 1434090440, 4210945686, 2372428917, 
    272306571, 1404207048, 82340563, 3429921732, 4076736198, 2283455606, 935276674, 
    1135857545, 417540252, 3447247564, 3228961222
    };
        static const quant_neuron_t neuron19 = {weights19, FL2FX(0.1771242469549179)};
        neurons[19]=neuron19;
    
        static  uint32_t weights20[] ={
    3247359670, 1613367490, 704483310, 1006096999, 1861745591, 4286898035, 2601100833, 
    1880250474, 961908173, 3788880876, 3148946838, 4288016499, 3305687112, 1369070655, 
    4005601639, 1247868788, 3120812903, 2370203248
    };
        static const quant_neuron_t neuron20 = {weights20, FL2FX(-0.2198513299226761)};
        neurons[20]=neuron20;
    
        static  uint32_t weights21[] ={
    70127872, 55109348, 3892908401, 465452904, 588776528, 223358248, 1325283598, 1833490096, 
    4202788794, 4136076225, 2232936009, 81379209, 1877154447, 1329915884, 2543224402, 
    2315554717, 488082584, 825891885
    };
        static const quant_neuron_t neuron21 = {weights21, FL2FX(0.29604336619377136)};
        neurons[21]=neuron21;
    
        static  uint32_t weights22[] ={
    82468110, 357966836, 95063432, 1416243840, 1353203144, 1104358304, 660831501, 223286960, 
    402071502, 2237407468, 1433067976, 1381928069, 1858706797, 591865403, 1940459421, 
    2294951618, 3461890412, 169708145
    };
        static const quant_neuron_t neuron22 = {weights22, FL2FX(0.009362933225929737)};
        neurons[22]=neuron22;
    
        static  uint32_t weights23[] ={
    504390481, 4135899253, 164397806, 1095296574, 4060374856, 2716213762, 4246300469, 
    3489920349, 2042895205, 3922165369, 3559137797, 4016448087, 3885232111, 1279524185, 
    3987962633, 2370596093, 1421618031, 1736765297
    };
        static const quant_neuron_t neuron23 = {weights23, FL2FX(0.04867268726229668)};
        neurons[23]=neuron23;
    
        static  uint32_t weights24[] ={
    2599362142, 3212768195, 742208023, 1857437444, 1611818065, 736246147, 325915384, 
    4101302208, 2949454774, 4223638007, 2248806737, 4180507991, 48447166, 3564553160, 
    611391714, 2567178144, 504912213, 3725335736
    };
        static const quant_neuron_t neuron24 = {weights24, FL2FX(0.22285014390945435)};
        neurons[24]=neuron24;
    
        static  uint32_t weights25[] ={
    3058497393, 3064055910, 570229551, 1807188852, 1655909976, 1003895619, 3716310063, 
    3427028905, 350294390, 528598496, 4127570743, 4286961491, 4250722631, 3071867930, 
    1482318199, 1877043175, 994440823, 4017266267
    };
        static const quant_neuron_t neuron25 = {weights25, FL2FX(0.2148810774087906)};
        neurons[25]=neuron25;
    
        static  uint32_t weights26[] ={
    3372262925, 140765630, 2058509768, 467040746, 804327591, 3432810361, 2171575437, 
    206207471, 1351044875, 1138255468, 1056268286, 4204902697, 599432622, 525801903, 
    1771175276, 3800130155, 1071996580, 2092217071
    };
        static const quant_neuron_t neuron26 = {weights26, FL2FX(-0.01313908863812685)};
        neurons[26]=neuron26;
    
        static  uint32_t weights27[] ={
    3239500660, 61749899, 3914719911, 1630377488, 4089638550, 2733988039, 2512966196, 
    2762684951, 1772985770, 3153582458, 3872582297, 3478450054, 3339363900, 3835898075, 
    1911267244, 331835500, 4276714444, 3402424998
    };
        static const quant_neuron_t neuron27 = {weights27, FL2FX(0.022487996146082878)};
        neurons[27]=neuron27;
    
        static  uint32_t weights28[] ={
    826288751, 4251313399, 2566489534, 1006070635, 1914007953, 4021325682, 3948503684, 
    3433781071, 2255675477, 3844514720, 3648664703, 535502271, 2212707982, 3430303560, 
    2205389503, 3182806419, 354352499, 2141796787
    };
        static const quant_neuron_t neuron28 = {weights28, FL2FX(0.003889655927196145)};
        neurons[28]=neuron28;
    
        static  uint32_t weights29[] ={
    2356889972, 3066953891, 83837686, 3496478293, 342520536, 4268050565, 339505254, 
    42868875, 1678794947, 3628721684, 3564305496, 2652390534, 271588587, 119346320, 
    68472201, 1615935000, 291969152, 3491512492
    };
        static const quant_neuron_t neuron29 = {weights29, FL2FX(0.24189653992652893)};
        neurons[29]=neuron29;
    
        static  uint32_t weights30[] ={
    1011930481, 4170888319, 2412754571, 3864641692, 344376525, 1096000794, 1061446437, 
    1342760564, 408666063, 99151084, 671393069, 1767024152, 3925321643, 3632377176, 
    593968824, 165611530, 1088764396, 1375781720
    };
        static const quant_neuron_t neuron30 = {weights30, FL2FX(0.14262692630290985)};
        neurons[30]=neuron30;
    
        static  uint32_t weights31[] ={
    763932219, 41531297, 4077181848, 2543196267, 2371727495, 1278068500, 713657054, 
    3303374977, 4270113776, 4018237555, 3579663018, 2767753235, 2089615719, 787277868, 
    3934773628, 3389183009, 378078063, 3821773356
    };
        static const quant_neuron_t neuron31 = {weights31, FL2FX(-0.012024731375277042)};
        neurons[31]=neuron31;
    
        static  uint32_t weights32[] ={
    475545955, 4062986961, 1416331258, 4268541143, 2262984397, 3529926359, 1427923028, 
    1879630928, 517372201, 2424202181, 75506253, 3814786579, 3918848004, 3492359453, 
    1140013791, 1570696320, 1287004543, 2007150423
    };
        static const quant_neuron_t neuron32 = {weights32, FL2FX(0.35102546215057373)};
        neurons[32]=neuron32;
    
        static  uint32_t weights33[] ={
    3775761008, 86595740, 1762437713, 1080276280, 1671731976, 23285760, 4018464594, 
    1094580318, 3817320330, 870420844, 2816789265, 2839594061, 4144424920, 3858431324, 
    3775237003, 2840785208, 2251670353, 3899797269
    };
        static const quant_neuron_t neuron33 = {weights33, FL2FX(0.07849222421646118)};
        neurons[33]=neuron33;
    
        static  uint32_t weights34[] ={
    1888861050, 2746325843, 4177903275, 1810209131, 3255230509, 2396342339, 4157914930, 
    4158750785, 3771588539, 4024817931, 1611056436, 233116739, 1938144944, 77693955, 
    3773212844, 2242369935, 539463696, 3165417735
    };
        static const quant_neuron_t neuron34 = {weights34, FL2FX(0.023466667160391808)};
        neurons[34]=neuron34;
    
        static  uint32_t weights35[] ={
    2541374832, 952710772, 1576680218, 14756589, 3555929552, 3921402627, 2350440741, 
    1763515026, 3554322312, 3437353614, 3746675085, 3686779050, 2919821693, 1697067195, 
    3474049113, 1291418012, 1328734689, 1517758198
    };
        static const quant_neuron_t neuron35 = {weights35, FL2FX(-0.2081151157617569)};
        neurons[35]=neuron35;
    
        static  uint32_t weights36[] ={
    109648257, 2986422397, 4091139008, 2292316591, 3440928073, 3436799158, 202920820, 
    3482071669, 3516129616, 4263693754, 3440551019, 1211575486, 3692618549, 1787390591, 
    3548112013, 255581070, 3976241279, 602586937
    };
        static const quant_neuron_t neuron36 = {weights36, FL2FX(0.05101194605231285)};
        neurons[36]=neuron36;
    
        static  uint32_t weights37[] ={
    883854134, 21656420, 1711247256, 971749924, 1621944024, 3824302081, 262636061, 1685643777, 
    4001729173, 3789087303, 2557755605, 3978691094, 3672191700, 3967211988, 2386600533, 
    3978491862, 2618967361, 2779936979
    };
        static const quant_neuron_t neuron37 = {weights37, FL2FX(0.18476590514183044)};
        neurons[37]=neuron37;
    
        static  uint32_t weights38[] ={
    384840344, 3573581684, 3548963707, 3467081930, 2769846858, 6922664, 1106452145, 
    733318789, 2451237744, 3873968274, 757631689, 544747525, 1549627661, 1806953463, 
    3091817279, 3869978501, 4289291039, 3825199723
    };
        static const quant_neuron_t neuron38 = {weights38, FL2FX(0.06672549992799759)};
        neurons[38]=neuron38;
    
        static  uint32_t weights39[] ={
    3912342605, 228327278, 4212160529, 2249854882, 1005149606, 856537849, 3949927553, 
    417440590, 799235523, 1993325028, 2063423463, 730178481, 2084396357, 197901868, 
    1588165950, 3595281071, 1993281454, 2211947369
    };
        static const quant_neuron_t neuron39 = {weights39, FL2FX(0.2470587193965912)};
        neurons[39]=neuron39;
    
        static  uint32_t weights40[] ={
    3988384699, 1351424107, 1440196677, 1872858146, 4189540598, 3198508511, 3348744880, 
    4027062483, 2501179171, 4218805363, 3637739413, 3153480390, 3558392178, 269047991, 
    3164382055, 2017588602, 3009666735, 2843099721
    };
        static const quant_neuron_t neuron40 = {weights40, FL2FX(-0.12423312664031982)};
        neurons[40]=neuron40;
    
        static  uint32_t weights41[] ={
    3820320934, 2820739137, 3761695268, 10865666, 2165106280, 75187214, 2183115311, 
    2812385695, 2977424044, 2879606807, 5751392, 1948786838, 2919883637, 598934487, 
    2606032153, 4180308063, 2306831008, 2621261966
    };
        static const quant_neuron_t neuron41 = {weights41, FL2FX(-0.2282693088054657)};
        neurons[41]=neuron41;
    
        static  uint32_t weights42[] ={
    1235225472, 1213099644, 1479991365, 1104320416, 773587011, 21014360, 3790987911, 
    1235933452, 2284122090, 569869193, 2955718782, 560075857, 4229816659, 2050550838, 
    3939312487, 1796403043, 1657218871, 2713105241
    };
        static const quant_neuron_t neuron42 = {weights42, FL2FX(0.3316037058830261)};
        neurons[42]=neuron42;
    
        static  uint32_t weights43[] ={
    3797246806, 1286085136, 3988449963, 1118667788, 3339247721, 6452482, 2970871485, 
    1105434887, 2825904074, 469703654, 3341312889, 3699959191, 850917051, 737657231, 
    640670600, 167729796, 3736420496, 2091701670
    };
        static const quant_neuron_t neuron43 = {weights43, FL2FX(0.008772794157266617)};
        neurons[43]=neuron43;
    
        static  uint32_t weights44[] ={
    602050221, 634840004, 3363374294, 415216228, 2062646749, 4204148642, 182656253, 
    3872861808, 53294850, 2617337998, 475547980, 1783603628, 602095260, 3344401385, 
    1399158960, 1580020119, 1562106196, 436821166
    };
        static const quant_neuron_t neuron44 = {weights44, FL2FX(0.04873025417327881)};
        neurons[44]=neuron44;
    
        static  uint32_t weights45[] ={
    2285519297, 250371443, 992899154, 348165520, 1873001655, 866777419, 2222674372, 
    1079925228, 508596678, 1352386406, 4224802706, 4223523179, 1332332996, 743209548, 
    503849424, 1650115892, 3733997395, 1773403826
    };
        static const quant_neuron_t neuron45 = {weights45, FL2FX(0.013996907509863377)};
        neurons[45]=neuron45;
    
        static  uint32_t weights46[] ={
    3477573323, 137994653, 512679580, 2519535751, 2632767468, 272532132, 1608813707, 
    4057564956, 2191875468, 2501816812, 3627864316, 1449025854, 1571263497, 3495503004, 
    1185226380, 2219417900, 3691790568, 2186683076
    };
        static const quant_neuron_t neuron46 = {weights46, FL2FX(0.20642779767513275)};
        neurons[46]=neuron46;
    
        static  uint32_t weights47[] ={
    221799233, 2053717813, 4087054100, 2413197547, 693739591, 1751183122, 4015908657, 
    3897095188, 1360191454, 2950492910, 2493637615, 3953581139, 3575311257, 3907091672, 
    1175984977, 1791261876, 4293944043, 1248304986
    };
        static const quant_neuron_t neuron47 = {weights47, FL2FX(0.04451838135719299)};
        neurons[47]=neuron47;
    
        static  uint32_t weights48[] ={
    1056365978, 4160587686, 1447326516, 3733864499, 368989714, 2633878972, 478057765, 
    1067246244, 3053548196, 3592307963, 134166186, 337621132, 48315028, 2415174636, 
    226549876, 3271964787, 396320026, 272591022
    };
        static const quant_neuron_t neuron48 = {weights48, FL2FX(0.14390647411346436)};
        neurons[48]=neuron48;
    
        static  uint32_t weights49[] ={
    4078225023, 3890447051, 2840431342, 695071311, 4165325716, 3903123654, 954607998, 
    2540823913, 1889895218, 2613195074, 1649275217, 209838374, 2079538815, 3338063329, 
    106167860, 2545876623, 3394794101, 3735677391
    };
        static const quant_neuron_t neuron49 = {weights49, FL2FX(-0.04765201732516289)};
        neurons[49]=neuron49;
    
        static  uint32_t weights50[] ={
    4254706682, 1810209907, 4153782202, 3954548003, 3238449195, 485993967, 4152752930, 
    3903151127, 4284389230, 2017301807, 809947696, 1034736090, 2853040658, 602134470, 
    2182722099, 2410898760, 2572063232, 2363123102
    };
        static const quant_neuron_t neuron50 = {weights50, FL2FX(-0.033822521567344666)};
        neurons[50]=neuron50;
    
        static  uint32_t weights51[] ={
    4068805626, 1944388163, 1623579579, 736467299, 2166815656, 2053513411, 847652470, 
    2535719443, 881295918, 510496781, 951050878, 2523519171, 814740726, 2769946643, 
    878470992, 4169679943, 4200522693, 4004141526
    };
        static const quant_neuron_t neuron51 = {weights51, FL2FX(-0.14888721704483032)};
        neurons[51]=neuron51;
    
        static  uint32_t weights52[] ={
    3923171719, 1000299478, 1152363205, 1680385813, 4205092479, 2981329099, 404540226, 
    1072258358, 2340205235, 1236459401, 2718221403, 2706539897, 3692356097, 2015915038, 
    4285694145, 431669048, 537166923, 2839688793
    };
        static const quant_neuron_t neuron52 = {weights52, FL2FX(0.2971479296684265)};
        neurons[52]=neuron52;
    
        static  uint32_t weights53[] ={
    533817751, 4204475251, 1306505775, 2686571530, 2760043951, 1642281043, 2547049268, 
    4238403197, 1906556925, 2354159322, 2395235791, 3814127226, 85189177, 2023962460, 
    359341741, 2505885892, 3701793137, 1805954674
    };
        static const quant_neuron_t neuron53 = {weights53, FL2FX(0.055177610367536545)};
        neurons[53]=neuron53;
    
        static  uint32_t weights54[] ={
    388675085, 2809394149, 3632864580, 4225131368, 727254273, 3915276984, 1727667465, 
    265777140, 1655385356, 273821080, 2139826317, 18037285, 3557968195, 790249015, 3087502, 
    1930989192, 2814488193, 1662006864
    };
        static const quant_neuron_t neuron54 = {weights54, FL2FX(0.18784397840499878)};
        neurons[54]=neuron54;
    
        static  uint32_t weights55[] ={
    3851132859, 31932313, 3283536712, 275725013, 2557248232, 1081860636, 3951427372, 
    972665371, 3256833209, 3626827919, 2020266728, 2263424, 1779360373, 3885181467, 
    3246492809, 2285744332, 1432984456, 3491512492
    };
        static const quant_neuron_t neuron55 = {weights55, FL2FX(0.17118363082408905)};
        neurons[55]=neuron55;
    
        static  uint32_t weights56[] ={
    3726071185, 4271144566, 231875725, 75016084, 1856016004, 603077122, 3271991849, 
    3514002890, 159259279, 157148472, 2058432630, 2802206065, 3709652295, 1503362110, 
    419236975, 264943484, 1671522982, 3880129105
    };
        static const quant_neuron_t neuron56 = {weights56, FL2FX(0.03816239908337593)};
        neurons[56]=neuron56;
    
        static  uint32_t weights57[] ={
    3374102017, 1077542190, 1527903573, 2677205370, 2876315815, 3424503612, 3919363424, 
    166460302, 4216732048, 2576797479, 1236104338, 2669644764, 3894301809, 263928231, 
    2333271095, 2952777610, 738041535, 488358181
    };
        static const quant_neuron_t neuron57 = {weights57, FL2FX(0.24388930201530457)};
        neurons[57]=neuron57;
    
        static  uint32_t weights58[] ={
    3538348936, 4173847247, 217943018, 1659319156, 4072067065, 3824557777, 3766853404, 
    1405215050, 4174696789, 100898672, 1695521646, 591566129, 3701295433, 3659036722, 
    1492192710, 4068538064, 4225965895, 3821032315
    };
        static const quant_neuron_t neuron58 = {weights58, FL2FX(-0.031518131494522095)};
        neurons[58]=neuron58;
    
        static  uint32_t weights59[] ={
    1861515755, 4273913696, 3879250489, 3324614293, 2777583598, 568572141, 368939915, 
    1340118804, 2992613426, 2542280069, 3987508770, 2045498681, 995394188, 3916748645, 
    1450725010, 370937889, 2669011785, 3894654740
    };
        static const quant_neuron_t neuron59 = {weights59, FL2FX(0.06826182454824448)};
        neurons[59]=neuron59;
    
        static  uint32_t weights60[] ={
    3881405296, 715675871, 207545451, 1751901494, 1376928329, 22209612, 3830239792, 
    2163175523, 3791366404, 1789525294, 604415257, 3309450752, 3834176306, 2275614987, 
    3922968239, 167222058, 1631271966, 3632897197
    };
        static const quant_neuron_t neuron60 = {weights60, FL2FX(0.0861067920923233)};
        neurons[60]=neuron60;
    
        static  uint32_t weights61[] ={
    3611925224, 3762297511, 2303036143, 3762607738, 106943748, 2803692166, 1864977184, 
    1112044671, 2298553191, 4060145144, 3993099029, 2883416675, 3789916718, 3569479294, 
    2056882414, 3609309624, 1997859086, 4017171691
    };
        static const quant_neuron_t neuron61 = {weights61, FL2FX(0.16091834008693695)};
        neurons[61]=neuron61;
    
        static  uint32_t weights62[] ={
    3782386874, 1572679748, 1697565308, 633935001, 1703775896, 1625927822, 3791097806, 
    3891772687, 2158812122, 13176921, 380117888, 3379609922, 248941915, 600945181, 1319439689, 
    1481143921, 2651192136, 3997601282
    };
        static const quant_neuron_t neuron62 = {weights62, FL2FX(0.05581909790635109)};
        neurons[62]=neuron62;
    
        static  uint32_t weights63[] ={
    3986394845, 677752863, 3385696207, 1006105195, 2832507685, 3991262804, 1370927084, 
    2851859278, 3949201229, 1241146681, 1760270646, 336162073, 831819689, 4178955863, 
    2773893377, 2014388240, 1374231968, 2953072728
    };
        static const quant_neuron_t neuron63 = {weights63, FL2FX(0.05954379588365555)};
        neurons[63]=neuron63;
    
        quantdense_layer_t layer= {64, neurons};
        return layer;
    }
    
batch_normalization_layer_t init_batch_normalization_15_data(void){

    static const fixed inv_gamma_dev[] ={
    FL2FX(1.156417/sqrt(294.491364+0.001000)), FL2FX(1.193787/sqrt(711.629150+0.001000)), 
    FL2FX(1.167682/sqrt(964.642395+0.001000)), FL2FX(1.002378/sqrt(880.095398+0.001000)), 
    FL2FX(1.158031/sqrt(1003.316833+0.001000)), FL2FX(1.063400/sqrt(810.583069+0.001000)), 
    FL2FX(1.119626/sqrt(600.621948+0.001000)), FL2FX(1.059211/sqrt(1004.824646+0.001000)), 
    FL2FX(1.111642/sqrt(1148.219482+0.001000)), FL2FX(1.071459/sqrt(698.850281+0.001000)), 
    FL2FX(1.072767/sqrt(524.719543+0.001000)), FL2FX(1.419515/sqrt(1019.589172+0.001000)), 
    FL2FX(1.300755/sqrt(714.925659+0.001000)), FL2FX(1.068900/sqrt(886.951294+0.001000)), 
    FL2FX(1.266076/sqrt(759.418457+0.001000)), FL2FX(1.158644/sqrt(965.064514+0.001000)), 
    FL2FX(1.187059/sqrt(1269.342896+0.001000)), FL2FX(1.566671/sqrt(1611.052368+0.001000)), 
    FL2FX(1.102931/sqrt(1039.726440+0.001000)), FL2FX(1.330246/sqrt(1031.183838+0.001000)), 
    FL2FX(1.103950/sqrt(1201.183105+0.001000)), FL2FX(1.195854/sqrt(809.275757+0.001000)), 
    FL2FX(1.167711/sqrt(750.830444+0.001000)), FL2FX(1.180456/sqrt(838.413757+0.001000)), 
    FL2FX(1.051793/sqrt(663.508972+0.001000)), FL2FX(1.067291/sqrt(961.946716+0.001000)), 
    FL2FX(1.092200/sqrt(1286.105103+0.001000)), FL2FX(1.315862/sqrt(1444.353638+0.001000)), 
    FL2FX(1.123837/sqrt(1418.983398+0.001000)), FL2FX(1.166286/sqrt(415.613556+0.001000)), 
    FL2FX(1.041121/sqrt(638.833618+0.001000)), FL2FX(0.995931/sqrt(339.860657+0.001000)), 
    FL2FX(1.293468/sqrt(992.630371+0.001000)), FL2FX(1.177297/sqrt(990.407227+0.001000)), 
    FL2FX(1.380009/sqrt(653.584167+0.001000)), FL2FX(1.194985/sqrt(1211.004272+0.001000)), 
    FL2FX(1.141933/sqrt(936.900574+0.001000)), FL2FX(1.143139/sqrt(657.580139+0.001000)), 
    FL2FX(0.954105/sqrt(586.771729+0.001000)), FL2FX(1.106526/sqrt(985.979553+0.001000)), 
    FL2FX(1.293035/sqrt(1488.807983+0.001000)), FL2FX(1.358942/sqrt(792.734619+0.001000)), 
    FL2FX(1.262569/sqrt(794.301025+0.001000)), FL2FX(1.511795/sqrt(581.532776+0.001000)), 
    FL2FX(1.222648/sqrt(652.997437+0.001000)), FL2FX(1.155919/sqrt(774.897583+0.001000)), 
    FL2FX(1.082585/sqrt(546.396851+0.001000)), FL2FX(0.941494/sqrt(869.818542+0.001000)), 
    FL2FX(1.024119/sqrt(894.422485+0.001000)), FL2FX(1.085952/sqrt(526.124146+0.001000)), 
    FL2FX(1.869671/sqrt(1191.380127+0.001000)), FL2FX(1.273238/sqrt(671.564758+0.001000)), 
    FL2FX(1.211476/sqrt(910.612915+0.001000)), FL2FX(0.982816/sqrt(791.622070+0.001000)), 
    FL2FX(1.062108/sqrt(673.746521+0.001000)), FL2FX(1.152809/sqrt(768.257019+0.001000)), 
    FL2FX(1.300704/sqrt(1409.031494+0.001000)), FL2FX(0.964297/sqrt(1403.769653+0.001000)), 
    FL2FX(1.013923/sqrt(874.841003+0.001000)), FL2FX(1.130026/sqrt(739.460754+0.001000)), 
    FL2FX(1.119383/sqrt(803.734070+0.001000)), FL2FX(1.197154/sqrt(545.765686+0.001000)), 
    FL2FX(1.362565/sqrt(917.597595+0.001000)), FL2FX(0.938674/sqrt(683.524780+0.001000)), 
  
    };
    static const fixed std_beta[] ={
    FL2FX(0.430080-7.750697*1.156417/sqrt(294.491364+0.001000)), FL2FX(-0.046054-20.912722*1.193787/sqrt(711.629150+0.001000)), 
    FL2FX(0.087209-22.180738*1.167682/sqrt(964.642395+0.001000)), FL2FX(-0.049029-22.096081*1.002378/sqrt(880.095398+0.001000)), 
    FL2FX(0.021265-23.083979*1.158031/sqrt(1003.316833+0.001000)), FL2FX(-0.000442-24.589859*1.063400/sqrt(810.583069+0.001000)), 
    FL2FX(-0.063041-12.212884*1.119626/sqrt(600.621948+0.001000)), FL2FX(0.037962-23.852728*1.059211/sqrt(1004.824646+0.001000)), 
    FL2FX(0.165335-25.316757*1.111642/sqrt(1148.219482+0.001000)), FL2FX(0.041987-18.440145*1.071459/sqrt(698.850281+0.001000)), 
    FL2FX(0.069454-15.712477*1.072767/sqrt(524.719543+0.001000)), FL2FX(-0.346328-28.969461*1.419515/sqrt(1019.589172+0.001000)), 
    FL2FX(-0.220750-17.938375*1.300755/sqrt(714.925659+0.001000)), FL2FX(0.149646-23.485332*1.068900/sqrt(886.951294+0.001000)), 
    FL2FX(-0.039192-14.709850*1.266076/sqrt(759.418457+0.001000)), FL2FX(0.193632-31.160183*1.158644/sqrt(965.064514+0.001000)), 
    FL2FX(0.088944-36.289413*1.187059/sqrt(1269.342896+0.001000)), FL2FX(-0.125762-26.100643*1.566671/sqrt(1611.052368+0.001000)), 
    FL2FX(0.188549-22.969868*1.102931/sqrt(1039.726440+0.001000)), FL2FX(-0.335425-22.454494*1.330246/sqrt(1031.183838+0.001000)), 
    FL2FX(-0.126420-31.581415*1.103950/sqrt(1201.183105+0.001000)), FL2FX(0.216799-17.741543*1.195854/sqrt(809.275757+0.001000)), 
    FL2FX(0.055575-12.728655*1.167711/sqrt(750.830444+0.001000)), FL2FX(0.147974-19.896114*1.180456/sqrt(838.413757+0.001000)), 
    FL2FX(-0.019724-17.206940*1.051793/sqrt(663.508972+0.001000)), FL2FX(0.101093-29.925812*1.067291/sqrt(961.946716+0.001000)), 
    FL2FX(0.089725-27.183617*1.092200/sqrt(1286.105103+0.001000)), FL2FX(-0.172209-30.484953*1.315862/sqrt(1444.353638+0.001000)), 
    FL2FX(0.113009-34.681988*1.123837/sqrt(1418.983398+0.001000)), FL2FX(0.039017-9.946341*1.166286/sqrt(415.613556+0.001000)), 
    FL2FX(0.084468-15.543773*1.041121/sqrt(638.833618+0.001000)), FL2FX(0.021984-10.728850*0.995931/sqrt(339.860657+0.001000)), 
    FL2FX(0.236988-18.689186*1.293468/sqrt(992.630371+0.001000)), FL2FX(0.038918-24.288904*1.177297/sqrt(990.407227+0.001000)), 
    FL2FX(-0.181156-15.826189*1.380009/sqrt(653.584167+0.001000)), FL2FX(-0.192769-29.631323*1.194985/sqrt(1211.004272+0.001000)), 
    FL2FX(0.051312-20.661785*1.141933/sqrt(936.900574+0.001000)), FL2FX(-0.022651-15.533853*1.143139/sqrt(657.580139+0.001000)), 
    FL2FX(0.258828-15.443150*0.954105/sqrt(586.771729+0.001000)), FL2FX(0.044285-29.230892*1.106526/sqrt(985.979553+0.001000)), 
    FL2FX(-0.214414-32.910805*1.293035/sqrt(1488.807983+0.001000)), FL2FX(-0.236871-17.129440*1.358942/sqrt(792.734619+0.001000)), 
    FL2FX(0.290413-18.452730*1.262569/sqrt(794.301025+0.001000)), FL2FX(-0.127464-16.397570*1.511795/sqrt(581.532776+0.001000)), 
    FL2FX(0.010111-15.704630*1.222648/sqrt(652.997437+0.001000)), FL2FX(0.051114-23.903181*1.155919/sqrt(774.897583+0.001000)), 
    FL2FX(-0.046438-12.421302*1.082585/sqrt(546.396851+0.001000)), FL2FX(0.017064-30.449419*0.941494/sqrt(869.818542+0.001000)), 
    FL2FX(0.113624-20.498892*1.024119/sqrt(894.422485+0.001000)), FL2FX(-0.000671-13.554240*1.085952/sqrt(526.124146+0.001000)), 
    FL2FX(-0.208516-25.269264*1.869671/sqrt(1191.380127+0.001000)), FL2FX(0.020555-13.950003*1.273238/sqrt(671.564758+0.001000)), 
    FL2FX(0.109344-26.188326*1.211476/sqrt(910.612915+0.001000)), FL2FX(0.081801-23.993250*0.982816/sqrt(791.622070+0.001000)), 
    FL2FX(0.079202-20.397854*1.062108/sqrt(673.746521+0.001000)), FL2FX(-0.226092-17.282356*1.152809/sqrt(768.257019+0.001000)), 
    FL2FX(0.110540-26.206495*1.300704/sqrt(1409.031494+0.001000)), FL2FX(0.016151-33.125019*0.964297/sqrt(1403.769653+0.001000)), 
    FL2FX(-0.064496-27.580004*1.013923/sqrt(874.841003+0.001000)), FL2FX(-0.019941-18.702282*1.130026/sqrt(739.460754+0.001000)), 
    FL2FX(-0.025797-17.651493*1.119383/sqrt(803.734070+0.001000)), FL2FX(-0.002558-14.601755*1.197154/sqrt(545.765686+0.001000)), 
    FL2FX(-0.041097-17.905884*1.362565/sqrt(917.597595+0.001000)), FL2FX(-0.089684-24.910742*0.938674/sqrt(683.524780+0.001000)), 
  
    };

    static const batch_normalization_layer_t norm = { 64, inv_gamma_dev, std_beta };
    return norm;
}

dense_layer_t init_dense_5_data(void){

    static neuron_t neurons[10];

    static const fixed weights0[] ={
    FL2FX(0.2142784297466278), FL2FX(-0.045506302267313004), FL2FX(-0.012926764786243439), 
    FL2FX(-0.15842320024967194), FL2FX(-0.31070101261138916), FL2FX(0.2899979054927826), 
    FL2FX(-0.14230972528457642), FL2FX(0.15944434702396393), FL2FX(0.09549380838871002), 
    FL2FX(-0.008602380752563477), FL2FX(0.034200992435216904), FL2FX(-0.224161759018898), 
    FL2FX(-0.13932131230831146), FL2FX(-0.28419601917266846), FL2FX(-0.16192267835140228), 
    FL2FX(-0.12328211218118668), FL2FX(-0.14789362251758575), FL2FX(-0.10185825824737549), 
    FL2FX(0.15325000882148743), FL2FX(-0.3349604308605194), FL2FX(-0.30897101759910583), 
    FL2FX(0.2897741198539734), FL2FX(0.2251095473766327), FL2FX(0.3236715495586395), 
    FL2FX(-0.20989198982715607), FL2FX(0.1060415580868721), FL2FX(-0.21902203559875488), 
    FL2FX(-0.28500714898109436), FL2FX(-0.04780374839901924), FL2FX(0.10886851698160172), 
    FL2FX(0.1600479781627655), FL2FX(-0.012475933879613876), FL2FX(0.14615163207054138), 
    FL2FX(0.02622060477733612), FL2FX(-0.3583351969718933), FL2FX(-0.009227038361132145), 
    FL2FX(0.20996835827827454), FL2FX(0.1807277351617813), FL2FX(0.24879547953605652), 
    FL2FX(-0.1275564283132553), FL2FX(-0.21138352155685425), FL2FX(0.06832956522703171), 
    FL2FX(0.2002876102924347), FL2FX(-0.06838489323854446), FL2FX(-0.2875097095966339), 
    FL2FX(-0.03536615148186684), FL2FX(-0.19069242477416992), FL2FX(0.15629497170448303), 
    FL2FX(-0.22510699927806854), FL2FX(0.1428697407245636), FL2FX(-0.10629770904779434), 
    FL2FX(-0.0874609649181366), FL2FX(0.13445225358009338), FL2FX(0.10287131369113922), 
    FL2FX(0.2602454721927643), FL2FX(0.01663563959300518), FL2FX(0.2622160315513611), 
    FL2FX(-0.11150343716144562), FL2FX(-0.0369160957634449), FL2FX(-0.21236220002174377), 
    FL2FX(0.010626967065036297), FL2FX(0.16382712125778198), FL2FX(-0.0004982355167157948), 
    FL2FX(-0.15654593706130981)
    };
    static const neuron_t neuron0 = {weights0, FL2FX(0.08552496135234833)};
    neurons[0]=neuron0;

    static const fixed weights1[] ={
    FL2FX(-0.01233584526926279), FL2FX(-0.028456328436732292), FL2FX(-0.27705493569374084), 
    FL2FX(-0.13050824403762817), FL2FX(-0.34918615221977234), FL2FX(-0.29410862922668457), 
    FL2FX(0.12539339065551758), FL2FX(0.01087993010878563), FL2FX(-0.03978334739804268), 
    FL2FX(-0.14556491374969482), FL2FX(0.012750263325870037), FL2FX(0.3425998091697693), 
    FL2FX(-0.09295042604207993), FL2FX(-0.23570497334003448), FL2FX(0.35364532470703125), 
    FL2FX(0.10012461990118027), FL2FX(0.21483761072158813), FL2FX(-0.04487287998199463), 
    FL2FX(-0.4695234000682831), FL2FX(0.3566790521144867), FL2FX(-0.12210305035114288), 
    FL2FX(-0.12229176610708237), FL2FX(0.558925449848175), FL2FX(0.15174545347690582), 
    FL2FX(0.4805959165096283), FL2FX(-0.021032119169831276), FL2FX(0.16386952996253967), 
    FL2FX(0.4977436065673828), FL2FX(-0.06478646397590637), FL2FX(-0.023510754108428955), 
    FL2FX(0.3048786520957947), FL2FX(-0.036846455186605453), FL2FX(0.05914461240172386), 
    FL2FX(-0.24488241970539093), FL2FX(0.06929777562618256), FL2FX(0.16510562598705292), 
    FL2FX(-0.06734049320220947), FL2FX(-0.12653271853923798), FL2FX(-0.2850053608417511), 
    FL2FX(0.2928813397884369), FL2FX(0.328684002161026), FL2FX(0.04864059016108513), 
    FL2FX(-0.21813665330410004), FL2FX(-0.08106515556573868), FL2FX(0.3498973846435547), 
    FL2FX(-0.375560462474823), FL2FX(0.3557276427745819), FL2FX(0.07929439842700958), 
    FL2FX(-0.03543277084827423), FL2FX(-0.2802942991256714), FL2FX(-0.16544035077095032), 
    FL2FX(-0.09837750345468521), FL2FX(-0.41318199038505554), FL2FX(0.2723836302757263), 
    FL2FX(0.23829913139343262), FL2FX(0.4480942487716675), FL2FX(0.3357555568218231), 
    FL2FX(-0.1843726634979248), FL2FX(0.18425443768501282), FL2FX(0.051292963325977325), 
    FL2FX(-0.13189329206943512), FL2FX(0.31321030855178833), FL2FX(0.21308869123458862), 
    FL2FX(-0.01955213025212288)
    };
    static const neuron_t neuron1 = {weights1, FL2FX(-0.17740873992443085)};
    neurons[1]=neuron1;

    static const fixed weights2[] ={
    FL2FX(0.13133752346038818), FL2FX(0.18500472605228424), FL2FX(-0.03639983758330345), 
    FL2FX(-0.17127402126789093), FL2FX(0.18551267683506012), FL2FX(0.3421644866466522), 
    FL2FX(0.03175598755478859), FL2FX(-0.21718789637088776), FL2FX(0.3062807619571686), 
    FL2FX(-0.08041384071111679), FL2FX(-0.14636485278606415), FL2FX(-0.15989167988300323), 
    FL2FX(-0.17275302112102509), FL2FX(0.04622849076986313), FL2FX(-0.20622912049293518), 
    FL2FX(0.2236773520708084), FL2FX(-0.038654524832963943), FL2FX(-0.08098374307155609), 
    FL2FX(0.11766652762889862), FL2FX(-0.17665281891822815), FL2FX(0.15158113837242126), 
    FL2FX(0.2161601036787033), FL2FX(0.1477949470281601), FL2FX(-0.22094658017158508), 
    FL2FX(0.18661150336265564), FL2FX(-0.1778121143579483), FL2FX(-0.02272755280137062), 
    FL2FX(-0.01635328307747841), FL2FX(0.2628481984138489), FL2FX(-0.06138395518064499), 
    FL2FX(-0.23122204840183258), FL2FX(-0.19909676909446716), FL2FX(-0.1233188658952713), 
    FL2FX(0.008164672181010246), FL2FX(-0.22924382984638214), FL2FX(-0.19617633521556854), 
    FL2FX(-0.08471197634935379), FL2FX(0.20873582363128662), FL2FX(0.1384515017271042), 
    FL2FX(-0.10214541107416153), FL2FX(-0.30851635336875916), FL2FX(-0.10737954080104828), 
    FL2FX(0.008845704607665539), FL2FX(-0.25977087020874023), FL2FX(0.18496085703372955), 
    FL2FX(0.014723642729222775), FL2FX(-0.25775599479675293), FL2FX(-0.18709751963615417), 
    FL2FX(0.048472125083208084), FL2FX(0.07366833835840225), FL2FX(-0.09532473981380463), 
    FL2FX(-0.2515221834182739), FL2FX(-0.088630810379982), FL2FX(-0.04917316511273384), 
    FL2FX(-0.062458183616399765), FL2FX(-0.27903687953948975), FL2FX(-0.18038547039031982), 
    FL2FX(-0.21449550986289978), FL2FX(0.10300946980714798), FL2FX(0.04242879897356033), 
    FL2FX(-0.00046424363972619176), FL2FX(0.1438218653202057), FL2FX(-0.024309229105710983), 
    FL2FX(0.05118842050433159)
    };
    static const neuron_t neuron2 = {weights2, FL2FX(0.053153473883867264)};
    neurons[2]=neuron2;

    static const fixed weights3[] ={
    FL2FX(0.11901230365037918), FL2FX(-0.11657514423131943), FL2FX(-0.16630305349826813), 
    FL2FX(-0.06334520131349564), FL2FX(0.3115867078304291), FL2FX(0.22983881831169128), 
    FL2FX(-0.2799297273159027), FL2FX(-0.15158744156360626), FL2FX(0.21897470951080322), 
    FL2FX(0.005848769098520279), FL2FX(0.07269061356782913), FL2FX(0.05317259952425957), 
    FL2FX(-0.24180294573307037), FL2FX(-0.06686399132013321), FL2FX(0.23343081772327423), 
    FL2FX(0.02977103926241398), FL2FX(0.17944884300231934), FL2FX(-0.3699615001678467), 
    FL2FX(0.33189553022384644), FL2FX(-0.15336716175079346), FL2FX(0.04493102803826332), 
    FL2FX(-0.3063546419143677), FL2FX(-0.020522937178611755), FL2FX(0.03985854238271713), 
    FL2FX(-0.281328946352005), FL2FX(0.06229708343744278), FL2FX(0.0606871135532856), 
    FL2FX(0.3051358163356781), FL2FX(-0.39175382256507874), FL2FX(-0.31492504477500916), 
    FL2FX(-0.11596698313951492), FL2FX(-0.0310527253895998), FL2FX(0.12968692183494568), 
    FL2FX(0.14861994981765747), FL2FX(0.09665821492671967), FL2FX(-0.18533636629581451), 
    FL2FX(-0.1331232488155365), FL2FX(-0.11498747766017914), FL2FX(-0.11437173932790756), 
    FL2FX(0.18825992941856384), FL2FX(0.3220142126083374), FL2FX(-0.07653196901082993), 
    FL2FX(0.2903030514717102), FL2FX(0.3850746750831604), FL2FX(-0.15822964906692505), 
    FL2FX(0.05885133519768715), FL2FX(-0.0752851739525795), FL2FX(-0.12122377753257751), 
    FL2FX(-0.28169122338294983), FL2FX(-0.2802356481552124), FL2FX(-0.2714262306690216), 
    FL2FX(-0.15970782935619354), FL2FX(0.19425362348556519), FL2FX(-0.138706237077713), 
    FL2FX(0.03247697651386261), FL2FX(-0.15256547927856445), FL2FX(0.314482182264328), 
    FL2FX(0.10370907187461853), FL2FX(0.09380335360765457), FL2FX(-0.26465559005737305), 
    FL2FX(0.24769608676433563), FL2FX(0.2624608278274536), FL2FX(-0.24788397550582886), 
    FL2FX(-0.1804606169462204)
    };
    static const neuron_t neuron3 = {weights3, FL2FX(-0.0032065645791590214)};
    neurons[3]=neuron3;

    static const fixed weights4[] ={
    FL2FX(0.1607324481010437), FL2FX(-0.23289059102535248), FL2FX(0.19986212253570557), 
    FL2FX(0.054332323372364044), FL2FX(0.18645918369293213), FL2FX(0.08037342876195908), 
    FL2FX(-0.20688430964946747), FL2FX(0.13710294663906097), FL2FX(-0.13641636073589325), 
    FL2FX(-0.33008378744125366), FL2FX(0.001499695936217904), FL2FX(0.10778427124023438), 
    FL2FX(0.19232067465782166), FL2FX(0.18168455362319946), FL2FX(-0.006559767760336399), 
    FL2FX(0.08966703712940216), FL2FX(0.2541996240615845), FL2FX(-0.2770363390445709), 
    FL2FX(-0.10268009454011917), FL2FX(-0.2110135704278946), FL2FX(0.061983395367860794), 
    FL2FX(0.21579088270664215), FL2FX(-0.07496042549610138), FL2FX(-0.11518257856369019), 
    FL2FX(0.07402156293392181), FL2FX(0.14135012030601501), FL2FX(0.25367024540901184), 
    FL2FX(-0.20495180785655975), FL2FX(0.1078808382153511), FL2FX(-0.10664530843496323), 
    FL2FX(-0.030807897448539734), FL2FX(0.09331583231687546), FL2FX(-0.1354781538248062), 
    FL2FX(-0.2048586755990982), FL2FX(-0.23976275324821472), FL2FX(-0.18950992822647095), 
    FL2FX(-0.29221996665000916), FL2FX(-0.22341658174991608), FL2FX(-0.1285259872674942), 
    FL2FX(-0.14420239627361298), FL2FX(0.2621932923793793), FL2FX(-0.08289322257041931), 
    FL2FX(0.2831972539424896), FL2FX(-0.2688813805580139), FL2FX(0.18653282523155212), 
    FL2FX(-0.07087347656488419), FL2FX(-0.22217470407485962), FL2FX(0.19500070810317993), 
    FL2FX(0.07340846210718155), FL2FX(0.3101535141468048), FL2FX(0.05446586757898331), 
    FL2FX(-0.344401091337204), FL2FX(-0.027303505688905716), FL2FX(-0.21008940041065216), 
    FL2FX(0.19352103769779205), FL2FX(-0.024760344997048378), FL2FX(0.24337133765220642), 
    FL2FX(0.15086013078689575), FL2FX(0.13970758020877838), FL2FX(-0.16998609900474548), 
    FL2FX(-0.20296871662139893), FL2FX(-0.019863903522491455), FL2FX(-0.12592196464538574), 
    FL2FX(0.0997367799282074)
    };
    static const neuron_t neuron4 = {weights4, FL2FX(-0.021726425737142563)};
    neurons[4]=neuron4;

    static const fixed weights5[] ={
    FL2FX(-0.17661623656749725), FL2FX(0.2495340257883072), FL2FX(-0.19546914100646973), 
    FL2FX(-0.14928525686264038), FL2FX(0.15060126781463623), FL2FX(-0.03120332397520542), 
    FL2FX(0.3883804380893707), FL2FX(-0.04406479001045227), FL2FX(-0.22097407281398773), 
    FL2FX(-0.15824514627456665), FL2FX(0.28003832697868347), FL2FX(0.18372631072998047), 
    FL2FX(0.2167777717113495), FL2FX(0.06988893449306488), FL2FX(-0.4392224848270416), 
    FL2FX(-0.2368713915348053), FL2FX(-0.331654816865921), FL2FX(0.41473811864852905), 
    FL2FX(-0.03799348697066307), FL2FX(-0.08835221081972122), FL2FX(-0.22117190062999725), 
    FL2FX(0.05852099508047104), FL2FX(-0.3351142108440399), FL2FX(-0.2520204186439514), 
    FL2FX(-0.10270914435386658), FL2FX(-0.3214898407459259), FL2FX(-0.27099791169166565), 
    FL2FX(-0.009822363033890724), FL2FX(-0.4936971664428711), FL2FX(0.026254232972860336), 
    FL2FX(0.31564027070999146), FL2FX(0.23229192197322845), FL2FX(-0.24811123311519623), 
    FL2FX(0.2979777455329895), FL2FX(0.19036537408828735), FL2FX(-0.3713149130344391), 
    FL2FX(-0.14157244563102722), FL2FX(0.32525837421417236), FL2FX(0.2792966961860657), 
    FL2FX(-0.3732375502586365), FL2FX(-0.025161774829030037), FL2FX(0.2572100758552551), 
    FL2FX(0.21518711745738983), FL2FX(0.29514241218566895), FL2FX(-0.3187691569328308), 
    FL2FX(-0.3768608272075653), FL2FX(0.05230439081788063), FL2FX(-0.22167636454105377), 
    FL2FX(-0.16670994460582733), FL2FX(0.15293751657009125), FL2FX(0.4196985960006714), 
    FL2FX(0.19019891321659088), FL2FX(0.2135888785123825), FL2FX(-0.006955066230148077), 
    FL2FX(-0.36306077241897583), FL2FX(0.41126859188079834), FL2FX(-0.29760345816612244), 
    FL2FX(-0.12964800000190735), FL2FX(0.09745414555072784), FL2FX(0.2322370707988739), 
    FL2FX(0.2737322747707367), FL2FX(0.2423277050256729), FL2FX(0.28961682319641113), 
    FL2FX(0.26674559712409973)
    };
    static const neuron_t neuron5 = {weights5, FL2FX(-0.07308455556631088)};
    neurons[5]=neuron5;

    static const fixed weights6[] ={
    FL2FX(0.38420066237449646), FL2FX(-0.16141489148139954), FL2FX(-0.016752801835536957), 
    FL2FX(0.08698864281177521), FL2FX(0.014483424834907055), FL2FX(-0.09425986558198929), 
    FL2FX(-0.16427606344223022), FL2FX(-0.07641095668077469), FL2FX(0.05767343193292618), 
    FL2FX(0.24394093453884125), FL2FX(0.04702195152640343), FL2FX(-0.14830033481121063), 
    FL2FX(-0.2812623679637909), FL2FX(0.34283819794654846), FL2FX(-0.13284295797348022), 
    FL2FX(0.2118939757347107), FL2FX(0.0974796712398529), FL2FX(-0.0471034049987793), 
    FL2FX(0.15954391658306122), FL2FX(-0.30165866017341614), FL2FX(-0.26068127155303955), 
    FL2FX(0.135650172829628), FL2FX(-0.05618606135249138), FL2FX(0.22702772915363312), 
    FL2FX(-0.07187574356794357), FL2FX(-0.13416580855846405), FL2FX(0.2527848780155182), 
    FL2FX(-0.11628799885511398), FL2FX(0.0038998760282993317), FL2FX(0.1434144675731659), 
    FL2FX(0.33405977487564087), FL2FX(0.2654869258403778), FL2FX(0.23069177567958832), 
    FL2FX(0.004026437643915415), FL2FX(-0.22702622413635254), FL2FX(-0.24709363281726837), 
    FL2FX(0.18821777403354645), FL2FX(-0.1873897761106491), FL2FX(0.2548965513706207), 
    FL2FX(0.04545373097062111), FL2FX(-0.07497507333755493), FL2FX(-0.28318122029304504), 
    FL2FX(0.17111892998218536), FL2FX(-0.10337480157613754), FL2FX(0.06305188685655594), 
    FL2FX(0.008627342991530895), FL2FX(-0.010558041743934155), FL2FX(0.1347043663263321), 
    FL2FX(0.2252051681280136), FL2FX(-0.14941078424453735), FL2FX(-0.02578485570847988), 
    FL2FX(-0.16133449971675873), FL2FX(0.19225451350212097), FL2FX(0.21140287816524506), 
    FL2FX(0.08963456004858017), FL2FX(-0.27889326214790344), FL2FX(0.16914059221744537), 
    FL2FX(0.20439262688159943), FL2FX(-0.21686255931854248), FL2FX(0.000698607531376183), 
    FL2FX(-0.03415695205330849), FL2FX(0.0450718030333519), FL2FX(-0.06223335117101669), 
    FL2FX(-0.22860053181648254)
    };
    static const neuron_t neuron6 = {weights6, FL2FX(0.18685857951641083)};
    neurons[6]=neuron6;

    static const fixed weights7[] ={
    FL2FX(-0.15536609292030334), FL2FX(-0.2590911388397217), FL2FX(-0.21348945796489716), 
    FL2FX(0.1608840525150299), FL2FX(-0.3181664049625397), FL2FX(0.18154451251029968), 
    FL2FX(-0.2089768946170807), FL2FX(-0.1484844982624054), FL2FX(-0.2792835533618927), 
    FL2FX(0.046436015516519547), FL2FX(-0.13617266714572906), FL2FX(0.34517380595207214), 
    FL2FX(0.2183564454317093), FL2FX(-0.004394595045596361), FL2FX(-0.3215593695640564), 
    FL2FX(-0.26106467843055725), FL2FX(-0.3043805956840515), FL2FX(0.4636325538158417), 
    FL2FX(-0.03077433630824089), FL2FX(0.11373691260814667), FL2FX(0.26664426922798157), 
    FL2FX(-0.2184770703315735), FL2FX(-0.029450148344039917), FL2FX(-0.11600485444068909), 
    FL2FX(0.21503354609012604), FL2FX(-0.10534878075122833), FL2FX(-0.32586032152175903), 
    FL2FX(0.22645771503448486), FL2FX(-0.07801075279712677), FL2FX(-0.11228718608617783), 
    FL2FX(-0.2670682370662689), FL2FX(0.13079003989696503), FL2FX(-0.2005361169576645), 
    FL2FX(0.26943257451057434), FL2FX(0.047901567071676254), FL2FX(0.30310896039009094), 
    FL2FX(0.1244460940361023), FL2FX(0.2544086277484894), FL2FX(-0.3912825882434845), 
    FL2FX(-0.34424757957458496), FL2FX(0.4273395240306854), FL2FX(0.10073230415582657), 
    FL2FX(-0.37808483839035034), FL2FX(0.3383045494556427), FL2FX(-0.19452573359012604), 
    FL2FX(0.2694661319255829), FL2FX(-0.11944736540317535), FL2FX(0.23000077903270721), 
    FL2FX(-0.2889947295188904), FL2FX(-0.10100339353084564), FL2FX(0.43218809366226196), 
    FL2FX(-0.008798900991678238), FL2FX(0.22831125557422638), FL2FX(0.1979914754629135), 
    FL2FX(-0.07815156877040863), FL2FX(-0.012383525259792805), FL2FX(-0.31067851185798645), 
    FL2FX(0.01342637836933136), FL2FX(-0.07886024564504623), FL2FX(0.24058327078819275), 
    FL2FX(-0.11631281673908234), FL2FX(-0.31604212522506714), FL2FX(0.3764156699180603), 
    FL2FX(-0.09882396459579468)
    };
    static const neuron_t neuron7 = {weights7, FL2FX(-0.11386038362979889)};
    neurons[7]=neuron7;

    static const fixed weights8[] ={
    FL2FX(0.1509362757205963), FL2FX(-0.4024926424026489), FL2FX(-0.5414544939994812), 
    FL2FX(0.2916300594806671), FL2FX(-0.24657265841960907), FL2FX(-0.0771251767873764), 
    FL2FX(0.19652704894542694), FL2FX(0.328203022480011), FL2FX(-0.3229736089706421), 
    FL2FX(0.09088172763586044), FL2FX(0.16415292024612427), FL2FX(-0.13234104216098785), 
    FL2FX(-0.3010413646697998), FL2FX(-0.2705618739128113), FL2FX(0.38961488008499146), 
    FL2FX(-0.4881046414375305), FL2FX(-0.3926890194416046), FL2FX(-0.5005074143409729), 
    FL2FX(0.07061467319726944), FL2FX(0.4647577106952667), FL2FX(-0.056319549679756165), 
    FL2FX(-0.44087329506874084), FL2FX(-0.04732540622353554), FL2FX(-0.22912926971912384), 
    FL2FX(0.19938939809799194), FL2FX(0.09091423451900482), FL2FX(-0.2520804703235626), 
    FL2FX(-0.32636070251464844), FL2FX(0.01854228414595127), FL2FX(0.44256240129470825), 
    FL2FX(-0.11920550465583801), FL2FX(-0.012146785855293274), FL2FX(0.36331865191459656), 
    FL2FX(-0.3665219843387604), FL2FX(0.4500754475593567), FL2FX(-0.22157400846481323), 
    FL2FX(-0.34968000650405884), FL2FX(-0.007006368599832058), FL2FX(-0.1390821635723114), 
    FL2FX(-0.28054311871528625), FL2FX(0.10692033171653748), FL2FX(-0.3919071853160858), 
    FL2FX(0.25252068042755127), FL2FX(-0.1760236769914627), FL2FX(-0.10693255066871643), 
    FL2FX(-0.11585178226232529), FL2FX(0.03351728245615959), FL2FX(-0.03458881378173828), 
    FL2FX(0.04531211033463478), FL2FX(0.15107089281082153), FL2FX(0.37592557072639465), 
    FL2FX(0.37599262595176697), FL2FX(-0.251200407743454), FL2FX(-0.09108835458755493), 
    FL2FX(-0.18700385093688965), FL2FX(-0.26704657077789307), FL2FX(-0.24851028621196747), 
    FL2FX(-0.13298559188842773), FL2FX(-0.194774329662323), FL2FX(-0.3426775634288788), 
    FL2FX(-0.09634944051504135), FL2FX(0.079753078520298), FL2FX(-0.1505660116672516), 
    FL2FX(-0.24710123240947723)
    };
    static const neuron_t neuron8 = {weights8, FL2FX(-0.010979749262332916)};
    neurons[8]=neuron8;

    static const fixed weights9[] ={
    FL2FX(-0.2581338584423065), FL2FX(0.19995860755443573), FL2FX(-0.16916370391845703), 
    FL2FX(0.3223784863948822), FL2FX(-0.24205580353736877), FL2FX(0.372188001871109), 
    FL2FX(0.0037300819531083107), FL2FX(-0.19173108041286469), FL2FX(-0.23403812944889069), 
    FL2FX(0.29856744408607483), FL2FX(-0.14550146460533142), FL2FX(0.28975334763526917), 
    FL2FX(0.24807944893836975), FL2FX(-0.09618125110864639), FL2FX(-0.3411104679107666), 
    FL2FX(-0.15376953780651093), FL2FX(-0.2960701584815979), FL2FX(0.2442365288734436), 
    FL2FX(-0.2305898368358612), FL2FX(0.3631364703178406), FL2FX(-0.058687999844551086), 
    FL2FX(-0.29963263869285583), FL2FX(-0.09703287482261658), FL2FX(-0.2963058054447174), 
    FL2FX(-0.3600359261035919), FL2FX(-0.3819690942764282), FL2FX(-0.05444089323282242), 
    FL2FX(0.18404753506183624), FL2FX(-0.19518102705478668), FL2FX(0.13897886872291565), 
    FL2FX(-0.09480994939804077), FL2FX(0.1213163360953331), FL2FX(-0.18117468059062958), 
    FL2FX(-0.05182817205786705), FL2FX(0.1632097214460373), FL2FX(0.29773131012916565), 
    FL2FX(0.25860047340393066), FL2FX(0.003966816700994968), FL2FX(-0.20441332459449768), 
    FL2FX(-0.1620216965675354), FL2FX(0.4706183075904846), FL2FX(0.48359793424606323), 
    FL2FX(-0.25364920496940613), FL2FX(0.3754686713218689), FL2FX(-0.15485279262065887), 
    FL2FX(-0.11659248173236847), FL2FX(-0.29163745045661926), FL2FX(0.014904141426086426), 
    FL2FX(-0.0038242607843130827), FL2FX(-0.04369167611002922), FL2FX(0.43645215034484863), 
    FL2FX(-0.18806518614292145), FL2FX(0.17742681503295898), FL2FX(-0.06181076914072037), 
    FL2FX(-0.1808740794658661), FL2FX(0.31987056136131287), FL2FX(-0.07780216634273529), 
    FL2FX(0.36759838461875916), FL2FX(-0.2728458046913147), FL2FX(-0.15151666104793549), 
    FL2FX(0.47300562262535095), FL2FX(-0.1874697059392929), FL2FX(-0.25344544649124146), 
    FL2FX(-0.15278790891170502)
    };
    static const neuron_t neuron9 = {weights9, FL2FX(-0.18889763951301575)};
    neurons[9]=neuron9;

    dense_layer_t layer= {10, neurons};
    return layer;
}

