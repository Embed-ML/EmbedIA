#include "separable_model.h"

// Initialization function prototypes
conv2d_layer_t init_conv2d_4_data(void);
batch_normalization_layer_t init_batch_normalization_394_data(void);
quant_separable_conv2d_layer_t init_quant_separable_conv2d_300_data(void);
batch_normalization_layer_t init_batch_normalization_395_data(void);
quant_separable_conv2d_layer_t init_quant_separable_conv2d_301_data(void);
batch_normalization_layer_t init_batch_normalization_396_data(void);
quant_separable_conv2d_layer_t init_quant_separable_conv2d_302_data(void);
batch_normalization_layer_t init_batch_normalization_397_data(void);
dense_layer_t init_dense_80_data(void);


// Global Variables
conv2d_layer_t conv2d_4_data;
batch_normalization_layer_t batch_normalization_394_data;
quant_separable_conv2d_layer_t quant_separable_conv2d_300_data;
batch_normalization_layer_t batch_normalization_395_data;
quant_separable_conv2d_layer_t quant_separable_conv2d_301_data;
batch_normalization_layer_t batch_normalization_396_data;
quant_separable_conv2d_layer_t quant_separable_conv2d_302_data;
batch_normalization_layer_t batch_normalization_397_data;
dense_layer_t dense_80_data;


void model_init(){
    conv2d_4_data = init_conv2d_4_data();
    batch_normalization_394_data = init_batch_normalization_394_data();
    quant_separable_conv2d_300_data = init_quant_separable_conv2d_300_data();
    batch_normalization_395_data = init_batch_normalization_395_data();
    quant_separable_conv2d_301_data = init_quant_separable_conv2d_301_data();
    batch_normalization_396_data = init_batch_normalization_396_data();
    quant_separable_conv2d_302_data = init_quant_separable_conv2d_302_data();
    batch_normalization_397_data = init_batch_normalization_397_data();
    dense_80_data = init_dense_80_data();

}

void model_predict(data3d_t input, data1d_t * output){
  
    prepare_buffers();
    
    //*************** LAYER 0 **************//
    // Layer name: conv2d_4
    data3d_t output0;
    // convert image for first EmbedIA Conv2d layer
    channel_adapt_layer(input, &output0);
    input = output0;
    
     conv2d_layer(conv2d_4_data, input, &output0);
    
    //*************** LAYER 1 **************//
    // Layer name: max_pooling2d_130
    input = output0;
    static const pooling2d_layer_t max_pooling2d_130_data = { 2, 2 };
    max_pooling2d_layer(max_pooling2d_130_data, input, &output0);
    
    //*************** LAYER 2 **************//
    // Layer name: batch_normalization_394
    batch_normalization3d_layer(batch_normalization_394_data, &output0);
    
    //*************** LAYER 3 **************//
    // Layer name: activation_401
    tanh_activation(output0.data, 3600);
    //tanh_activation(output0.data, 3600);
    
    //*************** LAYER 4 **************//
    // Layer name: quant_separable_conv2d_300
    input = output0;
        quantSeparableConv2D_layer(quant_separable_conv2d_300_data, input, &output0);
    
    
    //*************** LAYER 5 **************//
    // Layer name: batch_normalization_395
    batch_normalization3d_layer(batch_normalization_395_data, &output0);
    
    //*************** LAYER 6 **************//
    // Layer name: activation_402
    tanh_activation(output0.data, 10816);
    //tanh_activation(output0.data, 10816);
    
    //*************** LAYER 7 **************//
    // Layer name: quant_separable_conv2d_301
    input = output0;
        quantSeparableConv2D_layer(quant_separable_conv2d_301_data, input, &output0);
    
    
    //*************** LAYER 8 **************//
    // Layer name: max_pooling2d_131
    input = output0;
    static const pooling2d_layer_t max_pooling2d_131_data = { 2, 2 };
    max_pooling2d_layer(max_pooling2d_131_data, input, &output0);
    
    //*************** LAYER 9 **************//
    // Layer name: batch_normalization_396
    batch_normalization3d_layer(batch_normalization_396_data, &output0);
    
    //*************** LAYER 10 **************//
    // Layer name: activation_403
    tanh_activation(output0.data, 2400);
    //tanh_activation(output0.data, 2400);
    
    //*************** LAYER 11 **************//
    // Layer name: quant_separable_conv2d_302
    input = output0;
        quantSeparableConv2D_layer(quant_separable_conv2d_302_data, input, &output0);
    
    
    //*************** LAYER 12 **************//
    // Layer name: average_pooling2d_19
    input = output0;
    static const pooling2d_layer_t average_pooling2d_19_data = { 3, 3 };
    avg_pooling2d_layer(average_pooling2d_19_data, input, &output0);
    
    //*************** LAYER 13 **************//
    // Layer name: batch_normalization_397
    batch_normalization3d_layer(batch_normalization_397_data, &output0);
    
    //*************** LAYER 14 **************//
    // Layer name: activation_404
    tanh_activation(output0.data, 512);
    //tanh_activation(output0.data, 512);
    
    //*************** LAYER 15 **************//
    // Layer name: flatten_81
    input = output0;
    data1d_t output1;
    flatten3d_layer(input, &output1);
    
    //*************** LAYER 16 **************//
    // Layer name: dropout_81
    data1d_t input1;
    input1 = output1;
    
    
    //*************** LAYER 17 **************//
    // Layer name: dense_80
    input1 = output1;
    dense_layer(dense_80_data, input1, &output1);
    
    
    //*************** LAYER 18 **************//
    // Layer name: activation_405
    softmax_activation(output1.data, 10);
    //softmax_activation(output1.data, 10);
    

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
        
        static const float weights0[]={ 
    0.08228804171085358, -0.0963955670595169, -0.04620002955198288, 
    0.020626971498131752, -0.12481505423784256, 0.09590932726860046, 
    0.06493749469518661, -0.001929971855133772, 0.008858618326485157, 

    0.08645451068878174, 0.027937674894928932, 0.04755430296063423, 
    0.17451880872249603, 0.02846948802471161, 0.0801004096865654, 
    0.3331131339073181, 0.09913935512304306, 0.07491442561149597, 

    -0.1735093891620636, 0.05173778533935547, -0.03341234102845192, 
    -0.013243959285318851, -0.22073906660079956, 0.033832352608442307, 
    -0.3851702809333801, -0.06594853103160858, -0.1657637059688568, 

        };
        static filter_t filter0 = {3, 3, weights0, 0.004690014757215977};
        filters[0]=filter0;
            
        static const float weights1[]={ 
    0.10211329162120819, -0.12470992654561996, -0.004398512654006481, 
    -0.060998376458883286, 0.11848388612270355, 0.20827628672122955, 
    0.03240258991718292, -0.015354417264461517, -0.19730065762996674, 

    -0.00013462806236930192, 0.16189810633659363, -0.14052148163318634, 
    -0.19981656968593597, -0.11584360897541046, 0.18885715305805206, 
    0.06403829157352448, -0.06730985641479492, -0.2074720710515976, 

    0.09367457032203674, -0.01941933110356331, -0.1700679510831833, 
    -0.13615423440933228, -0.03275391086935997, 0.13413803279399872, 
    0.06785587221384048, 0.0750606507062912, 0.0810794085264206, 

        };
        static filter_t filter1 = {3, 3, weights1, -0.02598356269299984};
        filters[1]=filter1;
            
        static const float weights2[]={ 
    -0.16793937981128693, 0.03168069198727608, -0.07671014964580536, 
    -0.2583506107330322, -0.04262114688754082, -0.09083450585603714, 
    0.007640570867806673, 0.3129176199436188, 0.21314498782157898, 

    0.23578645288944244, 0.17702199518680573, 0.01782919466495514, 
    -0.035915810614824295, -0.10543452203273773, -0.14821095764636993, 
    -0.020010381937026978, 0.03198579326272011, -0.11853165924549103, 

    0.0408768430352211, 0.11746351420879364, 0.17579206824302673, 
    -0.05947623401880264, -0.022399870678782463, -0.19497744739055634, 
    0.04562315717339516, -0.07723047584295273, -0.01732741668820381, 

        };
        static filter_t filter2 = {3, 3, weights2, -0.02325921133160591};
        filters[2]=filter2;
            
        static const float weights3[]={ 
    0.03019675984978676, -0.024400364607572556, 0.3653671443462372, 
    -0.0887872725725174, 0.011335883289575577, 0.10548269748687744, 
    -0.0811997503042221, 0.03668796643614769, -0.022846952080726624, 

    0.06337292492389679, -0.15330354869365692, 0.007012385409325361, 
    0.07847096771001816, 0.07570832222700119, -0.09771154075860977, 
    -0.10460770130157471, 0.09629963338375092, 0.0237538181245327, 

    -0.07615582644939423, 0.006383798085153103, -0.2103724330663681, 
    0.15973132848739624, -0.0309780053794384, -0.2173844426870346, 
    0.021435752511024475, 0.022669998928904533, -0.0019919825717806816, 

        };
        static filter_t filter3 = {3, 3, weights3, -0.015665389597415924};
        filters[3]=filter3;
            
        static const float weights4[]={ 
    0.07808686047792435, -0.33512386679649353, 0.054373227059841156, 
    0.029142286628484726, -0.12116800993680954, 0.057221293449401855, 
    -0.11581441015005112, -0.18652218580245972, -0.030103400349617004, 

    -0.13667161762714386, 0.17093360424041748, -0.034758202731609344, 
    0.22893604636192322, 0.11706513166427612, 0.11306778341531754, 
    0.024875205010175705, 0.18937942385673523, 0.16250693798065186, 

    0.044474903494119644, 0.03193109482526779, -0.09683395177125931, 
    -0.12411446869373322, 0.01803017221391201, 0.043256282806396484, 
    0.10247809439897537, -0.1523720622062683, -0.14259080588817596, 

        };
        static filter_t filter4 = {3, 3, weights4, 0.015336208045482635};
        filters[4]=filter4;
            
        static const float weights5[]={ 
    0.041611794382333755, -0.14550085365772247, 0.0824994370341301, 
    0.060813985764980316, -0.04505341872572899, 0.061086881905794144, 
    -0.03123951330780983, 0.012263060547411442, -0.05649837478995323, 

    0.09787383675575256, -0.09191825985908508, -0.06886423379182816, 
    0.09862851351499557, -0.2423498034477234, 0.0951862633228302, 
    0.11287878453731537, -0.2925305664539337, 0.16010333597660065, 

    0.026365913450717926, -0.18231825530529022, 0.15991166234016418, 
    0.09625604003667831, -0.0853051170706749, 0.07144800573587418, 
    0.10942021757364273, -0.2517184019088745, 0.10721128433942795, 

        };
        static filter_t filter5 = {3, 3, weights5, 0.002228367142379284};
        filters[5]=filter5;
            
        static const float weights6[]={ 
    -0.0068972003646194935, 0.1612379252910614, -0.09840334206819534, 
    -0.14844438433647156, 0.1764647513628006, -0.11370702087879181, 
    -0.07005379348993301, 0.14897067844867706, 0.14976483583450317, 

    -0.006153265945613384, -0.011188006028532982, -0.16950605809688568, 
    0.002476590219885111, 0.21135476231575012, -0.22480961680412292, 
    -0.06731913238763809, 0.23173686861991882, -0.13037440180778503, 

    -0.046213164925575256, 0.14375752210617065, -0.10184235870838165, 
    -0.04187711328268051, 0.1574469804763794, 0.02481950633227825, 
    -0.16522599756717682, 0.046289216727018356, -0.09444297850131989, 

        };
        static filter_t filter6 = {3, 3, weights6, 0.0165290255099535};
        filters[6]=filter6;
            
        static const float weights7[]={ 
    0.07592779397964478, -0.025073103606700897, 0.08566466718912125, 
    -0.021667849272489548, 0.11302012205123901, 0.011758713982999325, 
    0.0017444310942664742, -0.23097741603851318, 0.005019108299165964, 

    -0.053204894065856934, 0.1505800485610962, -0.2037469893693924, 
    0.032077863812446594, -0.07335863262414932, -0.03297124803066254, 
    -0.11705446988344193, -0.07307436317205429, -0.09666412323713303, 

    -0.24477128684520721, -0.3508206009864807, 0.1390736997127533, 
    -0.016883838921785355, -0.09275685250759125, 0.07208053767681122, 
    0.03795615956187248, 0.18672917783260345, 0.22478164732456207, 

        };
        static filter_t filter7 = {3, 3, weights7, 0.006412619259208441};
        filters[7]=filter7;
            
        static const float weights8[]={ 
    0.2237386256456375, -0.27354100346565247, 0.14161697030067444, 
    -0.2074686586856842, 0.1433587223291397, 0.08769561350345612, 
    0.045516252517700195, 0.04012812301516533, -0.02723914012312889, 

    0.14079135656356812, -0.277194619178772, 0.03424738347530365, 
    -0.05668613314628601, 0.13622254133224487, -0.111358143389225, 
    -0.06880483031272888, 0.024710342288017273, -0.08435513824224472, 

    0.11433204263448715, -0.22399812936782837, -0.052539292722940445, 
    0.09427870810031891, 0.06758082658052444, 0.06094881519675255, 
    0.011214219033718109, 0.039482004940509796, -0.037661049515008926, 

        };
        static filter_t filter8 = {3, 3, weights8, -0.0079775620251894};
        filters[8]=filter8;
            
        static const float weights9[]={ 
    -0.06553871929645538, -0.02251601405441761, 0.08531228452920914, 
    -0.10202302783727646, 0.12802037596702576, -0.048695676028728485, 
    -0.18298351764678955, -0.06307411938905716, 0.04938069358468056, 

    -0.11595537513494492, -0.06946826726198196, 0.25039178133010864, 
    -0.015176243148744106, 0.1300937533378601, 0.06928146630525589, 
    -0.05367019400000572, -0.04569004848599434, 0.08240823447704315, 

    -0.21627555787563324, -0.09622134268283844, 0.12823781371116638, 
    -0.2980712354183197, -0.009763708338141441, 0.1974317878484726, 
    -0.14766144752502441, 0.1081399917602539, 0.17256878316402435, 

        };
        static filter_t filter9 = {3, 3, weights9, 0.004438153468072414};
        filters[9]=filter9;
            
        static const float weights10[]={ 
    -0.08356974273920059, -0.06455706059932709, -0.029639089480042458, 
    0.21119338274002075, 0.2452361285686493, 0.21330678462982178, 
    -0.07610131055116653, -0.21483662724494934, -0.1391201913356781, 

    0.17866399884223938, -0.14951634407043457, 0.10979052633047104, 
    0.07220170646905899, -0.05615140497684479, 0.05725390091538429, 
    -0.2963865101337433, 0.004342771600931883, -0.09383980929851532, 

    -0.09464126825332642, 0.10989702492952347, -0.11082949489355087, 
    -0.09830819070339203, 0.030290979892015457, -0.05313422530889511, 
    0.13682375848293304, 0.13586360216140747, 0.011422093026340008, 

        };
        static filter_t filter10 = {3, 3, weights10, 0.009756894782185555};
        filters[10]=filter10;
            
        static const float weights11[]={ 
    -0.11338528990745544, 0.034128379076719284, -0.06327086687088013, 
    -0.014907756820321083, 0.29984062910079956, -0.05161198973655701, 
    0.13995864987373352, 0.12286589294672012, -0.17607247829437256, 

    -0.2739790976047516, 0.10617371648550034, -0.12551109492778778, 
    0.13414855301380157, 0.1816655546426773, -0.13429710268974304, 
    -0.13576050102710724, 0.07014161348342896, -0.1128723993897438, 

    -0.00524233840405941, 0.09986750781536102, 0.2068413645029068, 
    -0.06925990432500839, 0.0197162926197052, -0.0011624176986515522, 
    -0.0977872684597969, -0.14473780989646912, -0.05302322655916214, 

        };
        static filter_t filter11 = {3, 3, weights11, 0.0029919417575001717};
        filters[11]=filter11;
            
        static const float weights12[]={ 
    0.024201607331633568, -0.15579558908939362, -0.2790742516517639, 
    0.10418497025966644, 0.01587107963860035, 0.1958189308643341, 
    -0.044047653675079346, 0.07760028541088104, 0.18907523155212402, 

    -0.1577158272266388, -0.07102401554584503, 0.07753273099660873, 
    0.012336255982518196, 0.058539122343063354, -0.1159052923321724, 
    0.10849414765834808, -0.11320262402296066, -0.0034392722882330418, 

    -0.028334030881524086, 0.18078511953353882, 0.18970459699630737, 
    0.0016941912472248077, 0.04111512750387192, 0.04024588689208031, 
    0.0041425577364861965, 0.08870300650596619, -0.17296920716762543, 

        };
        static filter_t filter12 = {3, 3, weights12, 0.01783960498869419};
        filters[12]=filter12;
            
        static const float weights13[]={ 
    -0.06806991249322891, -0.035185348242521286, 0.03037935495376587, 
    0.19104796648025513, 0.015469346195459366, -0.13309143483638763, 
    -0.1186973825097084, -0.01714441180229187, 0.14246757328510284, 

    -0.15304450690746307, 0.15670901536941528, 0.12913650274276733, 
    0.07254714518785477, 0.05313649773597717, -0.28800398111343384, 
    0.06196538358926773, -0.11445916444063187, 0.06624334305524826, 

    -0.027139998972415924, -0.1979646384716034, 0.08384964615106583, 
    0.12629592418670654, 0.19297295808792114, -0.2256094366312027, 
    -0.12601187825202942, -0.07985392957925797, 0.2352391481399536, 

        };
        static filter_t filter13 = {3, 3, weights13, 0.00626011099666357};
        filters[13]=filter13;
            
        static const float weights14[]={ 
    0.17733575403690338, -0.0821838453412056, 0.2460663914680481, 
    -0.003462556516751647, 0.25706127285957336, 0.011909452266991138, 
    0.08644083142280579, -0.16307291388511658, -0.242851123213768, 

    0.1771826446056366, -0.00733132055029273, -0.04198332130908966, 
    -0.0367438942193985, 0.10649217665195465, 0.14313511550426483, 
    -0.1231454610824585, -0.15262457728385925, -0.05856122449040413, 

    0.0021120780147612095, -0.09502788633108139, 0.10205113887786865, 
    -0.07272452116012573, 0.05839076265692711, -0.03287900611758232, 
    0.03800735995173454, 0.033907562494277954, -0.11066171526908875, 

        };
        static filter_t filter14 = {3, 3, weights14, 0.009596770629286766};
        filters[14]=filter14;
            
        static const float weights15[]={ 
    0.027168063446879387, -0.24669606983661652, 0.14672429859638214, 
    0.05261041224002838, 0.041196685284376144, -0.0628603845834732, 
    -0.07810065895318985, 0.06194629147648811, -0.060423653572797775, 

    0.07941244542598724, -0.26684728264808655, 0.11935469508171082, 
    0.00011926383012905717, 0.2568161189556122, -0.10274422913789749, 
    -0.010405647568404675, -0.05739952623844147, 0.11798129230737686, 

    0.018480174243450165, -0.12393809109926224, -0.10116377472877502, 
    -0.15176820755004883, 0.30285993218421936, -0.04322504997253418, 
    0.042177580296993256, 0.011861360631883144, -0.005204911343753338, 

        };
        static filter_t filter15 = {3, 3, weights15, -0.0149376280605793};
        filters[15]=filter15;
            
        conv2d_layer_t layer = {16,filters};
        return layer;
        }
        
batch_normalization_layer_t init_batch_normalization_394_data(void){

    static const float inv_gamma_dev[] ={
    1.993009/sqrt(0.045040+0.001000), 1.035213/sqrt(0.008997+0.001000), 2.028276/sqrt(0.011683+0.001000), 
    0.897711/sqrt(0.015392+0.001000), 1.075650/sqrt(0.013872+0.001000), 2.406347/sqrt(0.015640+0.001000), 
    0.391448/sqrt(0.018709+0.001000), 1.396050/sqrt(0.060017+0.001000), 1.213796/sqrt(0.008749+0.001000), 
    1.521564/sqrt(0.072834+0.001000), 1.553874/sqrt(0.008040+0.001000), 0.706553/sqrt(0.016825+0.001000), 
    1.242810/sqrt(0.016722+0.001000), 1.978864/sqrt(0.005085+0.001000), 0.654472/sqrt(0.038189+0.001000), 
    2.043886/sqrt(0.007173+0.001000)
    };
    static const float std_beta[] ={
    -0.232981-(0.112848*1.993009/sqrt(0.045040+0.001000)), -0.138733-(0.027667*1.035213/sqrt(0.008997+0.001000)), 
    -0.814378-(0.051500*2.028276/sqrt(0.011683+0.001000)), 0.352311-(0.051183*0.897711/sqrt(0.015392+0.001000)), 
    -0.040302-(0.059344*1.075650/sqrt(0.013872+0.001000)), -1.450951-(0.079281*2.406347/sqrt(0.015640+0.001000)), 
    -0.201123-(0.110224*0.391448/sqrt(0.018709+0.001000)), -0.944747-(0.087707*1.396050/sqrt(0.060017+0.001000)), 
    0.499024-(0.055308*1.213796/sqrt(0.008749+0.001000)), -1.108702-(0.134256*1.521564/sqrt(0.072834+0.001000)), 
    -1.166777-(0.058150*1.553874/sqrt(0.008040+0.001000)), -0.224496-(0.093362*0.706553/sqrt(0.016825+0.001000)), 
    -0.986229-(0.029826*1.242810/sqrt(0.016722+0.001000)), -0.420900-(0.067072*1.978864/sqrt(0.005085+0.001000)), 
    -0.521091-(0.102829*0.654472/sqrt(0.038189+0.001000)), 0.724890-(0.045356*2.043886/sqrt(0.007173+0.001000)), 
  
    };

    static const batch_normalization_layer_t norm = { 16, inv_gamma_dev, std_beta };
    return norm;
}


            quant_separable_conv2d_layer_t init_quant_separable_conv2d_300_data(void){

            
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

    static const float inv_gamma_dev[] ={
    0.059070/sqrt(111.214180+0.001000), 0.561489/sqrt(88.480919+0.001000), 0.757129/sqrt(107.711472+0.001000), 
    0.407886/sqrt(122.712494+0.001000), 0.190249/sqrt(111.245399+0.001000), 0.006115/sqrt(92.301834+0.001000), 
    0.082474/sqrt(95.621750+0.001000), -0.312113/sqrt(109.682640+0.001000), 0.177239/sqrt(82.629814+0.001000), 
    1.131652/sqrt(111.693962+0.001000), 0.652466/sqrt(72.448265+0.001000), 1.028592/sqrt(92.369896+0.001000), 
    0.469044/sqrt(112.853523+0.001000), 1.037238/sqrt(93.279564+0.001000), 0.008873/sqrt(82.735970+0.001000), 
    0.930311/sqrt(140.257690+0.001000), 1.181052/sqrt(131.419418+0.001000), 0.058007/sqrt(86.660683+0.001000), 
    -0.058095/sqrt(114.774933+0.001000), 1.333406/sqrt(86.937416+0.001000), 0.575163/sqrt(137.714920+0.001000), 
    1.292935/sqrt(87.922890+0.001000), 1.658741/sqrt(78.584435+0.001000), 2.093564/sqrt(87.610603+0.001000), 
    0.058879/sqrt(85.720947+0.001000), 0.145276/sqrt(96.405479+0.001000), 0.989487/sqrt(96.214821+0.001000), 
    0.046698/sqrt(79.526947+0.001000), 0.016710/sqrt(77.605019+0.001000), 1.235741/sqrt(81.468002+0.001000), 
    0.989740/sqrt(104.436829+0.001000), 0.357190/sqrt(93.568321+0.001000), 0.880814/sqrt(86.858505+0.001000), 
    0.812150/sqrt(116.355835+0.001000), 1.851705/sqrt(88.815559+0.001000), 1.378206/sqrt(107.456795+0.001000), 
    1.119662/sqrt(99.201775+0.001000), 1.495626/sqrt(77.397293+0.001000), 0.686114/sqrt(96.429108+0.001000), 
    0.770921/sqrt(92.670433+0.001000), 0.133057/sqrt(98.346413+0.001000), -0.121981/sqrt(94.506691+0.001000), 
    1.398119/sqrt(71.244781+0.001000), 0.048120/sqrt(98.693123+0.001000), 1.381215/sqrt(98.208672+0.001000), 
    1.483584/sqrt(81.804260+0.001000), 0.771769/sqrt(92.450035+0.001000), 0.002574/sqrt(78.850533+0.001000), 
    1.232106/sqrt(97.853592+0.001000), 0.618145/sqrt(104.926262+0.001000), 0.243292/sqrt(99.505157+0.001000), 
    1.248834/sqrt(144.135269+0.001000), 1.179480/sqrt(107.974518+0.001000), -0.066047/sqrt(116.037369+0.001000), 
    0.137180/sqrt(95.405350+0.001000), -0.152082/sqrt(81.903305+0.001000), 1.738635/sqrt(103.634247+0.001000), 
    1.143872/sqrt(112.513985+0.001000), 0.119650/sqrt(105.311516+0.001000), 0.855846/sqrt(114.701141+0.001000), 
    -0.014211/sqrt(73.629059+0.001000), 1.059232/sqrt(91.664970+0.001000), 0.131723/sqrt(105.205269+0.001000), 
    0.127938/sqrt(104.160980+0.001000)
    };
    static const float std_beta[] ={
    1.061642-(-2.048846*0.059070/sqrt(111.214180+0.001000)), -0.012498-(-6.092588*0.561489/sqrt(88.480919+0.001000)), 
    -0.206194-(3.113620*0.757129/sqrt(107.711472+0.001000)), 0.068306-(-2.997951*0.407886/sqrt(122.712494+0.001000)), 
    0.005515-(0.193472*0.190249/sqrt(111.245399+0.001000)), -0.010063-(6.830170*0.006115/sqrt(92.301834+0.001000)), 
    0.043570-(2.104798*0.082474/sqrt(95.621750+0.001000)), 0.158878-(-10.721679*-0.312113/sqrt(109.682640+0.001000)), 
    -0.473826-(2.271303*0.177239/sqrt(82.629814+0.001000)), 0.291915-(2.606179*1.131652/sqrt(111.693962+0.001000)), 
    0.286911-(-3.531565*0.652466/sqrt(72.448265+0.001000)), 0.051964-(-1.475825*1.028592/sqrt(92.369896+0.001000)), 
    -0.115362-(6.529369*0.469044/sqrt(112.853523+0.001000)), 0.181447-(5.616878*1.037238/sqrt(93.279564+0.001000)), 
    -0.018144-(7.048623*0.008873/sqrt(82.735970+0.001000)), -0.272764-(6.587653*0.930311/sqrt(140.257690+0.001000)), 
    0.223665-(2.286747*1.181052/sqrt(131.419418+0.001000)), -0.234762-(-2.323969*0.058007/sqrt(86.660683+0.001000)), 
    -0.085592-(-11.006747*-0.058095/sqrt(114.774933+0.001000)), -0.533424-(-5.437778*1.333406/sqrt(86.937416+0.001000)), 
    -0.038635-(11.211671*0.575163/sqrt(137.714920+0.001000)), -0.485194-(2.278106*1.292935/sqrt(87.922890+0.001000)), 
    0.156074-(-4.603124*1.658741/sqrt(78.584435+0.001000)), 0.037224-(5.887838*2.093564/sqrt(87.610603+0.001000)), 
    -0.053322-(7.134128*0.058879/sqrt(85.720947+0.001000)), -0.147035-(-5.167170*0.145276/sqrt(96.405479+0.001000)), 
    0.458379-(0.602243*0.989487/sqrt(96.214821+0.001000)), 0.066260-(-6.813764*0.046698/sqrt(79.526947+0.001000)), 
    -0.019109-(1.940039*0.016710/sqrt(77.605019+0.001000)), 0.269894-(-1.987977*1.235741/sqrt(81.468002+0.001000)), 
    -0.119793-(2.754396*0.989740/sqrt(104.436829+0.001000)), -0.646093-(4.377793*0.357190/sqrt(93.568321+0.001000)), 
    -0.004006-(-0.295933*0.880814/sqrt(86.858505+0.001000)), 0.518981-(4.986172*0.812150/sqrt(116.355835+0.001000)), 
    0.160886-(6.829708*1.851705/sqrt(88.815559+0.001000)), -0.295391-(-0.034581*1.378206/sqrt(107.456795+0.001000)), 
    -0.714845-(-2.796635*1.119662/sqrt(99.201775+0.001000)), -0.101784-(-0.670536*1.495626/sqrt(77.397293+0.001000)), 
    -0.075146-(-2.600365*0.686114/sqrt(96.429108+0.001000)), -0.148982-(-1.390987*0.770921/sqrt(92.670433+0.001000)), 
    0.067334-(3.126426*0.133057/sqrt(98.346413+0.001000)), 0.161643-(0.982735*-0.121981/sqrt(94.506691+0.001000)), 
    0.303123-(1.831571*1.398119/sqrt(71.244781+0.001000)), 0.107539-(2.397360*0.048120/sqrt(98.693123+0.001000)), 
    0.519020-(5.704619*1.381215/sqrt(98.208672+0.001000)), -0.614281-(-4.405377*1.483584/sqrt(81.804260+0.001000)), 
    -0.193529-(-10.206701*0.771769/sqrt(92.450035+0.001000)), 0.027291-(4.094965*0.002574/sqrt(78.850533+0.001000)), 
    -0.486397-(-3.472038*1.232106/sqrt(97.853592+0.001000)), 0.035725-(8.780932*0.618145/sqrt(104.926262+0.001000)), 
    -0.029376-(-1.013927*0.243292/sqrt(99.505157+0.001000)), 0.816273-(5.985423*1.248834/sqrt(144.135269+0.001000)), 
    -0.489608-(-0.254525*1.179480/sqrt(107.974518+0.001000)), 0.097190-(11.482803*-0.066047/sqrt(116.037369+0.001000)), 
    -1.599058-(1.592864*0.137180/sqrt(95.405350+0.001000)), 0.137571-(0.892940*-0.152082/sqrt(81.903305+0.001000)), 
    -0.086796-(-4.618419*1.738635/sqrt(103.634247+0.001000)), -0.537223-(5.918550*1.143872/sqrt(112.513985+0.001000)), 
    0.093183-(4.024277*0.119650/sqrt(105.311516+0.001000)), -0.268694-(-13.273828*0.855846/sqrt(114.701141+0.001000)), 
    0.010863-(-4.494705*-0.014211/sqrt(73.629059+0.001000)), -0.063221-(2.743977*1.059232/sqrt(91.664970+0.001000)), 
    0.452529-(6.000952*0.131723/sqrt(105.205269+0.001000)), 0.174048-(-2.166692*0.127938/sqrt(104.160980+0.001000)), 
  
    };

    static const batch_normalization_layer_t norm = { 64, inv_gamma_dev, std_beta };
    return norm;
}


            quant_separable_conv2d_layer_t init_quant_separable_conv2d_301_data(void){

            
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

    static const float inv_gamma_dev[] ={
    0.762443/sqrt(399.112061+0.001000), 1.262347/sqrt(369.249634+0.001000), 1.459635/sqrt(337.246338+0.001000), 
    1.161113/sqrt(256.917236+0.001000), 1.226943/sqrt(431.714691+0.001000), 0.820192/sqrt(204.333740+0.001000), 
    0.976516/sqrt(331.422668+0.001000), 1.267003/sqrt(514.213074+0.001000), 1.317249/sqrt(347.916565+0.001000), 
    1.168623/sqrt(236.188553+0.001000), 1.240126/sqrt(474.700409+0.001000), 1.635037/sqrt(662.391968+0.001000), 
    1.136566/sqrt(301.266144+0.001000), 1.378003/sqrt(560.968628+0.001000), 0.857839/sqrt(267.379364+0.001000), 
    0.534878/sqrt(251.969727+0.001000), 1.171364/sqrt(225.863251+0.001000), 1.058238/sqrt(389.618073+0.001000), 
    1.170704/sqrt(333.118469+0.001000), 0.328319/sqrt(202.274200+0.001000), 1.214303/sqrt(585.808960+0.001000), 
    1.237607/sqrt(310.484436+0.001000), 0.978781/sqrt(246.082932+0.001000), 1.270001/sqrt(366.183533+0.001000), 
    0.996452/sqrt(426.328949+0.001000), 1.221157/sqrt(405.588196+0.001000), 0.986021/sqrt(274.268188+0.001000), 
    1.337566/sqrt(348.795654+0.001000), 1.191817/sqrt(432.084625+0.001000), 1.053683/sqrt(356.293091+0.001000), 
    1.159655/sqrt(345.870331+0.001000), 0.892138/sqrt(255.517181+0.001000), 1.224881/sqrt(386.710052+0.001000), 
    1.126566/sqrt(312.270447+0.001000), 1.251628/sqrt(298.610077+0.001000), 1.041563/sqrt(302.656311+0.001000), 
    1.038547/sqrt(294.503326+0.001000), 1.110702/sqrt(504.813812+0.001000), 0.831448/sqrt(256.225647+0.001000), 
    1.069132/sqrt(382.874207+0.001000), 0.775360/sqrt(227.221634+0.001000), 1.117724/sqrt(269.513367+0.001000), 
    0.259561/sqrt(284.096527+0.001000), 0.763286/sqrt(235.374298+0.001000), 1.117207/sqrt(282.009399+0.001000), 
    1.234700/sqrt(408.503418+0.001000), 0.866501/sqrt(246.558273+0.001000), 1.316944/sqrt(467.390869+0.001000), 
    0.925648/sqrt(280.473602+0.001000), 0.981331/sqrt(264.542816+0.001000), 1.223963/sqrt(345.022919+0.001000), 
    0.574299/sqrt(605.455505+0.001000), 1.023532/sqrt(326.111206+0.001000), 1.135829/sqrt(280.103973+0.001000), 
    1.103728/sqrt(378.053558+0.001000), 1.022860/sqrt(333.447388+0.001000), 0.698930/sqrt(263.637909+0.001000), 
    1.041979/sqrt(310.893951+0.001000), 1.405731/sqrt(387.218750+0.001000), 1.206824/sqrt(362.268036+0.001000), 
    1.042477/sqrt(613.096802+0.001000), 0.935171/sqrt(273.537628+0.001000), 1.212924/sqrt(315.830475+0.001000), 
    0.698882/sqrt(242.825912+0.001000), 1.347427/sqrt(388.849213+0.001000), 1.000040/sqrt(319.459625+0.001000), 
    1.134390/sqrt(381.867462+0.001000), 1.441635/sqrt(367.112549+0.001000), 1.198713/sqrt(371.567413+0.001000), 
    0.741005/sqrt(281.658051+0.001000), 1.048549/sqrt(226.859711+0.001000), 1.032900/sqrt(298.642670+0.001000), 
    1.074102/sqrt(213.100616+0.001000), 0.973129/sqrt(248.054016+0.001000), 0.777891/sqrt(381.295441+0.001000), 
    0.882307/sqrt(322.076019+0.001000), 1.255814/sqrt(367.616608+0.001000), 1.219126/sqrt(378.783295+0.001000), 
    0.943914/sqrt(273.859558+0.001000), 0.988764/sqrt(252.804916+0.001000), 1.117349/sqrt(299.022644+0.001000), 
    1.091344/sqrt(492.942810+0.001000), 1.103797/sqrt(256.951477+0.001000), 0.871794/sqrt(322.085144+0.001000), 
    0.359337/sqrt(228.776215+0.001000), 1.135305/sqrt(306.282898+0.001000), 1.007227/sqrt(472.956787+0.001000), 
    1.197893/sqrt(328.983246+0.001000), 1.017536/sqrt(299.656830+0.001000), 0.910601/sqrt(255.174896+0.001000), 
    1.326557/sqrt(291.407440+0.001000), 1.137929/sqrt(236.991241+0.001000), 0.849461/sqrt(261.303833+0.001000), 
    0.874149/sqrt(305.278534+0.001000), 1.097307/sqrt(297.145935+0.001000), 1.309676/sqrt(374.207947+0.001000), 
  
    };
    static const float std_beta[] ={
    -0.117390-(3.070204*0.762443/sqrt(399.112061+0.001000)), 0.587567-(20.335627*1.262347/sqrt(369.249634+0.001000)), 
    -0.313133-(2.346892*1.459635/sqrt(337.246338+0.001000)), 0.127915-(4.400020*1.161113/sqrt(256.917236+0.001000)), 
    -0.196079-(25.939053*1.226943/sqrt(431.714691+0.001000)), -0.091702-(18.063860*0.820192/sqrt(204.333740+0.001000)), 
    0.164175-(14.892305*0.976516/sqrt(331.422668+0.001000)), 0.240169-(15.723468*1.267003/sqrt(514.213074+0.001000)), 
    -0.077261-(28.606174*1.317249/sqrt(347.916565+0.001000)), 0.132448-(13.538617*1.168623/sqrt(236.188553+0.001000)), 
    0.005817-(30.711973*1.240126/sqrt(474.700409+0.001000)), -0.145176-(16.512676*1.635037/sqrt(662.391968+0.001000)), 
    0.504411-(3.159323*1.136566/sqrt(301.266144+0.001000)), -0.180361-(20.639782*1.378003/sqrt(560.968628+0.001000)), 
    0.088673-(8.429006*0.857839/sqrt(267.379364+0.001000)), 0.093903-(21.065765*0.534878/sqrt(251.969727+0.001000)), 
    -0.015275-(8.766741*1.171364/sqrt(225.863251+0.001000)), 0.074693-(31.770306*1.058238/sqrt(389.618073+0.001000)), 
    0.190099-(12.911551*1.170704/sqrt(333.118469+0.001000)), 0.099996-(23.423399*0.328319/sqrt(202.274200+0.001000)), 
    0.671714-(19.108603*1.214303/sqrt(585.808960+0.001000)), 0.080058-(7.118558*1.237607/sqrt(310.484436+0.001000)), 
    0.233215-(17.544970*0.978781/sqrt(246.082932+0.001000)), 0.364373-(12.485995*1.270001/sqrt(366.183533+0.001000)), 
    0.287191-(30.439936*0.996452/sqrt(426.328949+0.001000)), -0.316578-(21.143219*1.221157/sqrt(405.588196+0.001000)), 
    -0.001398-(9.725632*0.986021/sqrt(274.268188+0.001000)), -0.431202-(18.579823*1.337566/sqrt(348.795654+0.001000)), 
    0.331774-(19.075445*1.191817/sqrt(432.084625+0.001000)), -0.135018-(26.001064*1.053683/sqrt(356.293091+0.001000)), 
    -0.042610-(6.337032*1.159655/sqrt(345.870331+0.001000)), 0.120388-(8.962132*0.892138/sqrt(255.517181+0.001000)), 
    -0.106472-(35.794441*1.224881/sqrt(386.710052+0.001000)), 0.157530-(21.838844*1.126566/sqrt(312.270447+0.001000)), 
    0.088851-(14.314401*1.251628/sqrt(298.610077+0.001000)), 0.790130-(9.155838*1.041563/sqrt(302.656311+0.001000)), 
    -0.188375-(14.272759*1.038547/sqrt(294.503326+0.001000)), 0.136639-(10.551871*1.110702/sqrt(504.813812+0.001000)), 
    -0.138980-(12.650069*0.831448/sqrt(256.225647+0.001000)), 0.158759-(15.585749*1.069132/sqrt(382.874207+0.001000)), 
    -0.277990-(5.860517*0.775360/sqrt(227.221634+0.001000)), 0.091419-(17.113369*1.117724/sqrt(269.513367+0.001000)), 
    -0.074571-(15.985186*0.259561/sqrt(284.096527+0.001000)), 0.145391-(24.662418*0.763286/sqrt(235.374298+0.001000)), 
    -0.081002-(17.547857*1.117207/sqrt(282.009399+0.001000)), -0.251519-(22.703791*1.234700/sqrt(408.503418+0.001000)), 
    -0.120514-(6.520959*0.866501/sqrt(246.558273+0.001000)), 0.486714-(21.168463*1.316944/sqrt(467.390869+0.001000)), 
    -0.049155-(26.394617*0.925648/sqrt(280.473602+0.001000)), 0.176634-(20.752010*0.981331/sqrt(264.542816+0.001000)), 
    -0.072573-(27.037750*1.223963/sqrt(345.022919+0.001000)), 0.088090-(-11.763868*0.574299/sqrt(605.455505+0.001000)), 
    -0.025988-(10.981718*1.023532/sqrt(326.111206+0.001000)), 0.359166-(30.057110*1.135829/sqrt(280.103973+0.001000)), 
    0.158424-(16.585365*1.103728/sqrt(378.053558+0.001000)), -0.223144-(13.193221*1.022860/sqrt(333.447388+0.001000)), 
    -0.179488-(7.679944*0.698930/sqrt(263.637909+0.001000)), 0.015477-(21.112902*1.041979/sqrt(310.893951+0.001000)), 
    -0.110442-(11.323656*1.405731/sqrt(387.218750+0.001000)), -0.205449-(3.626335*1.206824/sqrt(362.268036+0.001000)), 
    0.377088-(4.209587*1.042477/sqrt(613.096802+0.001000)), 0.219545-(7.451916*0.935171/sqrt(273.537628+0.001000)), 
    -0.319791-(12.846603*1.212924/sqrt(315.830475+0.001000)), -0.371681-(29.561008*0.698882/sqrt(242.825912+0.001000)), 
    -0.083557-(9.177355*1.347427/sqrt(388.849213+0.001000)), -0.226807-(19.000626*1.000040/sqrt(319.459625+0.001000)), 
    0.239124-(18.080269*1.134390/sqrt(381.867462+0.001000)), 0.311088-(19.940750*1.441635/sqrt(367.112549+0.001000)), 
    -0.180102-(5.277673*1.198713/sqrt(371.567413+0.001000)), 0.143751-(16.521990*0.741005/sqrt(281.658051+0.001000)), 
    0.127333-(16.349766*1.048549/sqrt(226.859711+0.001000)), 0.075840-(12.796062*1.032900/sqrt(298.642670+0.001000)), 
    -0.065775-(10.167465*1.074102/sqrt(213.100616+0.001000)), -0.082013-(26.657309*0.973129/sqrt(248.054016+0.001000)), 
    -0.280005-(10.643170*0.777891/sqrt(381.295441+0.001000)), 0.231470-(36.347088*0.882307/sqrt(322.076019+0.001000)), 
    -0.306695-(10.752224*1.255814/sqrt(367.616608+0.001000)), 0.180886-(26.137220*1.219126/sqrt(378.783295+0.001000)), 
    0.030185-(2.360765*0.943914/sqrt(273.859558+0.001000)), -0.033604-(4.916474*0.988764/sqrt(252.804916+0.001000)), 
    -0.202707-(18.484186*1.117349/sqrt(299.022644+0.001000)), 0.029793-(23.910728*1.091344/sqrt(492.942810+0.001000)), 
    0.291738-(7.338949*1.103797/sqrt(256.951477+0.001000)), 0.077632-(16.656076*0.871794/sqrt(322.085144+0.001000)), 
    0.320301-(18.098093*0.359337/sqrt(228.776215+0.001000)), 0.131300-(14.660336*1.135305/sqrt(306.282898+0.001000)), 
    0.164198-(22.277546*1.007227/sqrt(472.956787+0.001000)), 0.752176-(29.169100*1.197893/sqrt(328.983246+0.001000)), 
    -0.244606-(2.741220*1.017536/sqrt(299.656830+0.001000)), -0.243386-(12.332290*0.910601/sqrt(255.174896+0.001000)), 
    -0.416125-(24.266220*1.326557/sqrt(291.407440+0.001000)), -0.023838-(31.131191*1.137929/sqrt(236.991241+0.001000)), 
    0.125715-(19.922379*0.849461/sqrt(261.303833+0.001000)), 0.010621-(1.284655*0.874149/sqrt(305.278534+0.001000)), 
    0.483015-(28.134546*1.097307/sqrt(297.145935+0.001000)), 0.352799-(36.102818*1.309676/sqrt(374.207947+0.001000)), 
  
    };

    static const batch_normalization_layer_t norm = { 96, inv_gamma_dev, std_beta };
    return norm;
}


            quant_separable_conv2d_layer_t init_quant_separable_conv2d_302_data(void){

            
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

    static const float inv_gamma_dev[] ={
    0.750679/sqrt(183.794113+0.001000), 0.883491/sqrt(164.063431+0.001000), 0.767102/sqrt(380.949829+0.001000), 
    1.060649/sqrt(173.529373+0.001000), 1.035712/sqrt(176.211868+0.001000), 0.689433/sqrt(193.719467+0.001000), 
    0.795292/sqrt(388.192474+0.001000), 1.263755/sqrt(250.895554+0.001000), 0.638560/sqrt(265.349518+0.001000), 
    1.242243/sqrt(312.762268+0.001000), 0.830333/sqrt(345.731171+0.001000), 1.369249/sqrt(324.516022+0.001000), 
    1.142395/sqrt(364.642822+0.001000), 0.728114/sqrt(156.604492+0.001000), 1.379342/sqrt(614.154053+0.001000), 
    0.931050/sqrt(134.725571+0.001000), 1.245805/sqrt(272.392303+0.001000), 0.924392/sqrt(189.338379+0.001000), 
    0.797337/sqrt(165.425644+0.001000), 0.761169/sqrt(126.588882+0.001000), 0.524586/sqrt(146.302658+0.001000), 
    0.982591/sqrt(312.185150+0.001000), 1.181889/sqrt(232.108673+0.001000), 0.759982/sqrt(250.576675+0.001000), 
    1.008273/sqrt(444.304047+0.001000), 0.954209/sqrt(337.202209+0.001000), 0.616673/sqrt(426.462097+0.001000), 
    0.741007/sqrt(372.745422+0.001000), 0.831581/sqrt(219.632721+0.001000), 1.008740/sqrt(224.835587+0.001000), 
    1.105117/sqrt(286.654114+0.001000), 0.836408/sqrt(119.280594+0.001000), 1.096259/sqrt(172.371063+0.001000), 
    0.920037/sqrt(311.613800+0.001000), 0.746397/sqrt(233.188690+0.001000), 0.559681/sqrt(176.567230+0.001000), 
    0.919873/sqrt(215.138123+0.001000), 0.678649/sqrt(144.545792+0.001000), 1.056221/sqrt(165.122787+0.001000), 
    1.092808/sqrt(208.811127+0.001000), 0.799793/sqrt(281.772003+0.001000), 1.250921/sqrt(295.080200+0.001000), 
    0.583861/sqrt(170.089203+0.001000), 0.982769/sqrt(187.606247+0.001000), 0.903135/sqrt(238.560028+0.001000), 
    0.739905/sqrt(164.511703+0.001000), 0.845826/sqrt(162.701477+0.001000), 0.978778/sqrt(147.430511+0.001000), 
    1.130841/sqrt(552.508545+0.001000), 0.931714/sqrt(195.784927+0.001000), 0.755018/sqrt(158.225357+0.001000), 
    1.117247/sqrt(228.381180+0.001000), 0.483002/sqrt(154.839417+0.001000), 0.638272/sqrt(215.700760+0.001000), 
    0.700161/sqrt(137.542084+0.001000), 0.968244/sqrt(308.922699+0.001000), 0.791358/sqrt(129.577652+0.001000), 
    0.822475/sqrt(148.368942+0.001000), 0.573975/sqrt(217.432190+0.001000), 0.760092/sqrt(157.014664+0.001000), 
    0.926740/sqrt(129.143829+0.001000), 1.015330/sqrt(171.310257+0.001000), 0.843927/sqrt(145.185669+0.001000), 
    0.951387/sqrt(442.708221+0.001000), 0.976816/sqrt(251.350067+0.001000), 1.007399/sqrt(292.258545+0.001000), 
    0.790463/sqrt(176.712097+0.001000), 0.965520/sqrt(223.962036+0.001000), 0.765308/sqrt(280.928223+0.001000), 
    0.881511/sqrt(162.300522+0.001000), 0.646607/sqrt(343.548370+0.001000), 0.765059/sqrt(260.238800+0.001000), 
    0.756451/sqrt(211.347778+0.001000), 0.904791/sqrt(220.740753+0.001000), 0.707400/sqrt(230.141006+0.001000), 
    0.881214/sqrt(119.680511+0.001000), 0.839123/sqrt(119.981644+0.001000), 0.719657/sqrt(186.148727+0.001000), 
    0.901798/sqrt(171.535294+0.001000), 0.881881/sqrt(336.699768+0.001000), 0.923361/sqrt(183.031204+0.001000), 
    0.738768/sqrt(145.611206+0.001000), 0.915717/sqrt(676.996643+0.001000), 0.851957/sqrt(238.919739+0.001000), 
    0.942640/sqrt(190.029053+0.001000), 0.828030/sqrt(139.950699+0.001000), 0.745401/sqrt(190.544037+0.001000), 
    0.472275/sqrt(131.150238+0.001000), 0.604055/sqrt(177.696762+0.001000), 0.570622/sqrt(196.696060+0.001000), 
    0.734422/sqrt(166.135422+0.001000), 1.482284/sqrt(529.812500+0.001000), 0.777455/sqrt(221.041718+0.001000), 
    0.746683/sqrt(170.835922+0.001000), 0.593853/sqrt(241.865051+0.001000), 1.417866/sqrt(311.951233+0.001000), 
    0.955159/sqrt(216.845001+0.001000), 0.662735/sqrt(291.371674+0.001000), 1.013658/sqrt(286.259796+0.001000), 
    0.535543/sqrt(228.341980+0.001000), 1.058384/sqrt(323.269135+0.001000), 0.765232/sqrt(128.747406+0.001000), 
    1.058068/sqrt(292.964417+0.001000), 1.035321/sqrt(210.046722+0.001000), 0.755284/sqrt(125.695099+0.001000), 
    0.720006/sqrt(169.339279+0.001000), 1.134286/sqrt(228.900742+0.001000), 0.771729/sqrt(176.532211+0.001000), 
    0.791719/sqrt(344.509460+0.001000), 0.969635/sqrt(177.923950+0.001000), 0.593232/sqrt(109.846466+0.001000), 
    0.962448/sqrt(386.825348+0.001000), 0.864769/sqrt(282.304291+0.001000), 0.837655/sqrt(161.619507+0.001000), 
    0.764769/sqrt(188.072937+0.001000), 0.854159/sqrt(317.924988+0.001000), 0.669213/sqrt(134.178055+0.001000), 
    0.854276/sqrt(147.206345+0.001000), 0.884858/sqrt(247.181778+0.001000), 0.868036/sqrt(138.920395+0.001000), 
    1.031286/sqrt(181.047302+0.001000), 0.972297/sqrt(162.536041+0.001000), 0.791167/sqrt(131.247147+0.001000), 
    0.639971/sqrt(156.897018+0.001000), 0.981607/sqrt(371.865753+0.001000), 0.745679/sqrt(116.281151+0.001000), 
    0.864640/sqrt(269.746979+0.001000), 0.920263/sqrt(299.290863+0.001000), 0.772694/sqrt(302.219086+0.001000), 
    0.667799/sqrt(99.489182+0.001000), 0.563844/sqrt(324.647522+0.001000), 1.011787/sqrt(367.474823+0.001000), 
    0.763452/sqrt(171.118851+0.001000), 0.929398/sqrt(187.320526+0.001000), 1.147684/sqrt(267.447937+0.001000), 
    0.838234/sqrt(232.514481+0.001000), 1.160890/sqrt(275.947113+0.001000), 0.757024/sqrt(205.323883+0.001000), 
    0.966467/sqrt(237.733856+0.001000), 0.781832/sqrt(210.660446+0.001000), 1.175303/sqrt(585.964600+0.001000), 
    1.052473/sqrt(240.278839+0.001000), 1.068428/sqrt(148.327393+0.001000), 0.957497/sqrt(343.977417+0.001000), 
    1.022751/sqrt(191.080093+0.001000), 0.904565/sqrt(334.221191+0.001000), 0.852476/sqrt(585.205994+0.001000), 
    0.828229/sqrt(192.988953+0.001000), 0.824247/sqrt(193.600967+0.001000), 0.912234/sqrt(226.316116+0.001000), 
    0.800982/sqrt(175.242310+0.001000), 0.815355/sqrt(149.531067+0.001000), 0.864880/sqrt(117.655594+0.001000), 
    0.787019/sqrt(161.163773+0.001000), 0.758110/sqrt(164.183472+0.001000), 0.936841/sqrt(322.148529+0.001000), 
    0.901713/sqrt(169.413071+0.001000), 0.929386/sqrt(374.353210+0.001000), 0.895230/sqrt(354.231750+0.001000), 
    0.641060/sqrt(117.598351+0.001000), 0.642121/sqrt(152.959808+0.001000), 1.028889/sqrt(302.255798+0.001000), 
    0.693197/sqrt(140.137512+0.001000), 1.410781/sqrt(307.468781+0.001000), 0.909177/sqrt(342.776154+0.001000), 
    0.832613/sqrt(166.124466+0.001000), 0.778213/sqrt(134.412689+0.001000), 0.806682/sqrt(327.656555+0.001000), 
    0.678508/sqrt(129.943268+0.001000), 0.451224/sqrt(269.683075+0.001000), 0.847387/sqrt(144.273300+0.001000), 
    0.849561/sqrt(141.191284+0.001000), 0.939889/sqrt(167.914108+0.001000), 0.483493/sqrt(253.882492+0.001000), 
    0.763394/sqrt(190.296707+0.001000), 1.022586/sqrt(494.586151+0.001000), 0.986355/sqrt(290.245026+0.001000), 
    0.783376/sqrt(171.187531+0.001000), 0.769421/sqrt(199.859833+0.001000), 0.779122/sqrt(248.165573+0.001000), 
    1.092562/sqrt(308.564392+0.001000), 0.821335/sqrt(138.844467+0.001000), 0.735155/sqrt(179.932343+0.001000), 
    1.021632/sqrt(260.388611+0.001000), 0.588541/sqrt(105.271988+0.001000), 0.686936/sqrt(295.504700+0.001000), 
    0.639154/sqrt(221.417404+0.001000), 0.984606/sqrt(151.790100+0.001000), 0.755497/sqrt(167.946686+0.001000), 
    0.858086/sqrt(330.408051+0.001000), 0.917839/sqrt(153.291168+0.001000), 0.742095/sqrt(167.956116+0.001000), 
    0.865658/sqrt(212.055252+0.001000), 0.656918/sqrt(181.683990+0.001000), 0.626508/sqrt(172.164993+0.001000), 
    0.577355/sqrt(125.397110+0.001000), 0.789151/sqrt(293.684082+0.001000), 1.040580/sqrt(249.099350+0.001000), 
    0.816116/sqrt(146.876068+0.001000), 1.225533/sqrt(266.165588+0.001000), 0.957588/sqrt(140.091690+0.001000), 
    0.855418/sqrt(138.797089+0.001000), 0.850987/sqrt(289.454987+0.001000), 0.693183/sqrt(176.870728+0.001000), 
    0.981261/sqrt(271.303680+0.001000), 0.708909/sqrt(231.376266+0.001000), 0.758847/sqrt(154.759399+0.001000), 
    0.896347/sqrt(218.770432+0.001000), 1.079516/sqrt(252.339096+0.001000), 0.989293/sqrt(166.481277+0.001000), 
    1.182887/sqrt(241.511749+0.001000), 0.900826/sqrt(184.146210+0.001000), 1.032980/sqrt(324.961548+0.001000), 
    1.154450/sqrt(219.549362+0.001000), 0.708216/sqrt(166.252991+0.001000), 1.011590/sqrt(255.986542+0.001000), 
    0.873168/sqrt(117.639984+0.001000), 1.164196/sqrt(235.918930+0.001000), 0.855211/sqrt(115.515320+0.001000), 
    1.219751/sqrt(189.480545+0.001000), 0.946711/sqrt(215.599930+0.001000), 0.760800/sqrt(420.280334+0.001000), 
    0.766619/sqrt(190.543182+0.001000), 0.991689/sqrt(287.125061+0.001000), 0.651705/sqrt(169.049850+0.001000), 
    1.116166/sqrt(298.970276+0.001000), 0.643305/sqrt(312.795746+0.001000), 0.868313/sqrt(166.564041+0.001000), 
    0.978477/sqrt(195.337555+0.001000), 0.603936/sqrt(324.610687+0.001000), 0.658405/sqrt(166.048401+0.001000), 
    0.678225/sqrt(215.069962+0.001000), 0.630645/sqrt(165.029099+0.001000), 0.827753/sqrt(118.744423+0.001000), 
    0.853326/sqrt(157.568619+0.001000), 0.778496/sqrt(172.984497+0.001000), 0.773650/sqrt(243.219482+0.001000), 
    0.594038/sqrt(212.615112+0.001000), 1.100457/sqrt(173.186539+0.001000), 0.984096/sqrt(154.892654+0.001000), 
    0.759250/sqrt(283.509277+0.001000), 0.727384/sqrt(461.723816+0.001000), 0.714637/sqrt(342.375580+0.001000), 
    0.775013/sqrt(119.609520+0.001000), 0.622289/sqrt(114.869057+0.001000), 0.932085/sqrt(205.057159+0.001000), 
    0.906272/sqrt(224.416061+0.001000), 0.912606/sqrt(348.997711+0.001000), 1.185480/sqrt(278.326538+0.001000), 
    0.795552/sqrt(229.161758+0.001000), 0.741116/sqrt(162.534195+0.001000), 1.027286/sqrt(232.497894+0.001000), 
    1.164752/sqrt(255.026428+0.001000), 0.819287/sqrt(152.933517+0.001000), 0.864285/sqrt(213.287308+0.001000), 
    0.844032/sqrt(331.315826+0.001000), 1.032779/sqrt(189.215759+0.001000), 0.642900/sqrt(216.955093+0.001000), 
    0.820709/sqrt(177.619186+0.001000), 0.639207/sqrt(139.203415+0.001000), 0.703655/sqrt(122.211098+0.001000), 
    1.013082/sqrt(195.002747+0.001000), 0.728469/sqrt(199.258057+0.001000), 0.838234/sqrt(236.063385+0.001000), 
    0.850901/sqrt(144.646469+0.001000), 0.895061/sqrt(220.762985+0.001000), 1.066899/sqrt(140.404694+0.001000), 
    0.991905/sqrt(187.856583+0.001000), 0.780839/sqrt(152.347382+0.001000), 0.803460/sqrt(182.404785+0.001000), 
    0.928889/sqrt(206.915024+0.001000), 0.752942/sqrt(223.656967+0.001000), 0.929372/sqrt(272.315857+0.001000), 
    0.759226/sqrt(187.482132+0.001000), 1.097139/sqrt(309.978729+0.001000), 0.967777/sqrt(227.527283+0.001000), 
    0.821351/sqrt(198.799713+0.001000), 0.795732/sqrt(206.003098+0.001000), 0.937088/sqrt(234.289017+0.001000), 
    1.000904/sqrt(259.974609+0.001000), 0.959789/sqrt(176.219589+0.001000), 0.642039/sqrt(107.895683+0.001000), 
    0.807803/sqrt(139.320526+0.001000), 0.944208/sqrt(197.539520+0.001000), 0.911537/sqrt(247.027771+0.001000), 
    0.705633/sqrt(236.083633+0.001000), 0.820100/sqrt(148.707275+0.001000), 0.832211/sqrt(153.052582+0.001000), 
    1.108139/sqrt(203.311295+0.001000), 0.973207/sqrt(126.663445+0.001000), 0.783927/sqrt(256.900269+0.001000), 
    0.968620/sqrt(213.466599+0.001000), 0.567436/sqrt(200.764175+0.001000), 0.584065/sqrt(154.437881+0.001000), 
    0.683100/sqrt(167.825409+0.001000), 0.774789/sqrt(150.928970+0.001000), 0.796204/sqrt(283.324463+0.001000), 
    1.143420/sqrt(302.671875+0.001000), 0.734889/sqrt(312.791779+0.001000), 0.568994/sqrt(120.704071+0.001000), 
    1.224349/sqrt(178.434189+0.001000), 1.008015/sqrt(213.925735+0.001000), 0.732257/sqrt(191.541306+0.001000), 
    0.595264/sqrt(213.422714+0.001000), 1.262220/sqrt(339.070129+0.001000), 0.955023/sqrt(210.737091+0.001000), 
    0.865259/sqrt(228.556381+0.001000), 0.580252/sqrt(233.801025+0.001000), 0.946033/sqrt(315.957397+0.001000), 
    0.791448/sqrt(145.878601+0.001000), 0.888759/sqrt(219.888046+0.001000), 1.063231/sqrt(260.517944+0.001000), 
    0.855176/sqrt(305.798523+0.001000), 0.637525/sqrt(265.963318+0.001000), 0.920539/sqrt(140.829468+0.001000), 
    1.217759/sqrt(499.735779+0.001000), 0.972019/sqrt(260.562134+0.001000), 0.807694/sqrt(264.221741+0.001000), 
    1.131782/sqrt(150.449936+0.001000), 0.547888/sqrt(128.066177+0.001000), 0.601016/sqrt(123.907738+0.001000), 
    0.912180/sqrt(187.605148+0.001000), 1.305207/sqrt(214.531418+0.001000), 1.413387/sqrt(496.952911+0.001000), 
    0.722100/sqrt(127.442734+0.001000), 0.597080/sqrt(187.809494+0.001000), 1.021203/sqrt(163.475479+0.001000), 
    0.850185/sqrt(189.675598+0.001000), 1.044190/sqrt(218.386765+0.001000), 1.043707/sqrt(254.593292+0.001000), 
    1.005548/sqrt(162.489182+0.001000), 0.842746/sqrt(213.385971+0.001000), 0.718463/sqrt(131.285583+0.001000), 
    0.691844/sqrt(168.638901+0.001000), 0.858130/sqrt(232.494751+0.001000), 0.834409/sqrt(214.154556+0.001000), 
    0.839877/sqrt(152.938919+0.001000), 0.739728/sqrt(177.528595+0.001000), 1.121459/sqrt(172.980423+0.001000), 
    1.001263/sqrt(173.338684+0.001000), 1.028222/sqrt(188.818085+0.001000), 1.032738/sqrt(336.843140+0.001000), 
    0.728425/sqrt(168.436890+0.001000), 1.170665/sqrt(413.908630+0.001000), 0.733893/sqrt(358.960754+0.001000), 
    0.752217/sqrt(165.759018+0.001000), 0.846335/sqrt(169.316849+0.001000), 0.625165/sqrt(164.740646+0.001000), 
    0.732072/sqrt(181.285980+0.001000), 0.774820/sqrt(133.677734+0.001000), 0.730029/sqrt(131.182907+0.001000), 
    0.655362/sqrt(133.641968+0.001000), 0.550855/sqrt(148.404877+0.001000), 1.249618/sqrt(320.903839+0.001000), 
    1.261876/sqrt(321.601776+0.001000), 0.880822/sqrt(302.192230+0.001000), 1.290730/sqrt(313.298370+0.001000), 
    0.936485/sqrt(445.619629+0.001000), 0.877705/sqrt(226.979065+0.001000), 0.870635/sqrt(261.694885+0.001000), 
    0.812991/sqrt(215.816528+0.001000), 0.659452/sqrt(237.270386+0.001000), 0.721113/sqrt(174.470367+0.001000), 
    0.670038/sqrt(204.275284+0.001000), 1.019209/sqrt(293.082336+0.001000), 1.023655/sqrt(140.578430+0.001000), 
    0.703914/sqrt(151.083664+0.001000), 0.605411/sqrt(113.446045+0.001000), 0.918798/sqrt(241.848846+0.001000), 
    0.695200/sqrt(221.263504+0.001000), 0.822500/sqrt(153.451065+0.001000), 0.662373/sqrt(132.390976+0.001000), 
    0.692816/sqrt(151.326675+0.001000), 0.968449/sqrt(313.220581+0.001000), 1.005165/sqrt(201.573257+0.001000), 
    0.677219/sqrt(111.363449+0.001000), 0.939789/sqrt(431.797089+0.001000), 0.744205/sqrt(281.766632+0.001000), 
    0.854380/sqrt(284.521179+0.001000), 0.736849/sqrt(161.276581+0.001000), 0.939497/sqrt(157.420761+0.001000), 
    0.785954/sqrt(172.549011+0.001000), 0.939585/sqrt(255.853134+0.001000), 0.592331/sqrt(148.325165+0.001000), 
    0.924299/sqrt(192.365005+0.001000), 0.659987/sqrt(132.378586+0.001000), 0.480913/sqrt(254.425690+0.001000), 
    0.911636/sqrt(219.446991+0.001000), 0.755067/sqrt(146.003326+0.001000), 0.957869/sqrt(161.232895+0.001000), 
    0.480797/sqrt(184.927689+0.001000), 0.880447/sqrt(253.899750+0.001000), 0.764348/sqrt(271.511322+0.001000), 
    0.859953/sqrt(236.679382+0.001000), 1.289335/sqrt(156.268494+0.001000), 0.884973/sqrt(203.797302+0.001000), 
    0.908592/sqrt(192.782654+0.001000), 1.149788/sqrt(365.497589+0.001000), 0.745504/sqrt(172.129257+0.001000), 
    0.798645/sqrt(195.216217+0.001000), 0.740087/sqrt(99.041069+0.001000), 0.628827/sqrt(138.007126+0.001000), 
    0.967902/sqrt(115.522758+0.001000), 0.954328/sqrt(167.889160+0.001000), 0.968058/sqrt(218.115967+0.001000), 
    1.038580/sqrt(231.173157+0.001000), 0.715082/sqrt(167.091446+0.001000), 0.728947/sqrt(249.572754+0.001000), 
    1.063136/sqrt(300.927460+0.001000), 0.580270/sqrt(155.068359+0.001000), 0.965664/sqrt(219.821930+0.001000), 
    0.989843/sqrt(172.390121+0.001000), 0.785234/sqrt(328.032684+0.001000), 1.099262/sqrt(254.171219+0.001000), 
    0.939356/sqrt(181.775650+0.001000), 0.981123/sqrt(243.654846+0.001000), 0.833147/sqrt(209.621094+0.001000), 
    0.903759/sqrt(201.587112+0.001000), 0.771366/sqrt(204.177383+0.001000), 0.890480/sqrt(131.167694+0.001000), 
    0.957790/sqrt(240.870239+0.001000), 0.909646/sqrt(184.503494+0.001000), 0.937562/sqrt(215.878983+0.001000), 
    0.887373/sqrt(203.578369+0.001000), 0.841901/sqrt(209.516937+0.001000), 1.283488/sqrt(347.486969+0.001000), 
    0.644061/sqrt(157.482239+0.001000), 0.574426/sqrt(127.698456+0.001000), 0.978431/sqrt(289.202576+0.001000), 
    0.901822/sqrt(127.744148+0.001000), 1.325540/sqrt(267.297150+0.001000), 0.811510/sqrt(160.334991+0.001000), 
    0.963087/sqrt(254.171127+0.001000), 1.005993/sqrt(219.382935+0.001000), 1.143173/sqrt(295.502075+0.001000), 
    1.157817/sqrt(258.521179+0.001000), 0.840435/sqrt(198.735291+0.001000), 0.686701/sqrt(303.727417+0.001000), 
    1.102462/sqrt(243.426437+0.001000), 1.151578/sqrt(360.945770+0.001000), 1.044169/sqrt(738.929382+0.001000), 
    1.212885/sqrt(293.488281+0.001000), 1.073056/sqrt(220.215714+0.001000), 0.749174/sqrt(218.720383+0.001000), 
    0.824449/sqrt(167.610168+0.001000), 0.813008/sqrt(179.341019+0.001000), 1.024933/sqrt(191.288162+0.001000), 
    1.068076/sqrt(196.418900+0.001000), 0.773342/sqrt(279.087128+0.001000), 0.640724/sqrt(132.193268+0.001000), 
    1.193822/sqrt(285.091522+0.001000), 0.915898/sqrt(131.874664+0.001000), 0.996405/sqrt(224.109360+0.001000), 
    0.981799/sqrt(263.265991+0.001000), 0.786942/sqrt(160.689423+0.001000), 1.004326/sqrt(172.114822+0.001000), 
    0.958180/sqrt(282.646149+0.001000), 0.634556/sqrt(191.222473+0.001000), 0.974911/sqrt(191.975204+0.001000), 
    0.935749/sqrt(151.939392+0.001000), 1.039169/sqrt(220.191940+0.001000), 0.755844/sqrt(216.537781+0.001000), 
    1.055384/sqrt(223.891144+0.001000), 0.660030/sqrt(112.261620+0.001000), 0.765314/sqrt(186.519958+0.001000), 
    1.025622/sqrt(216.054016+0.001000), 0.855197/sqrt(200.431610+0.001000), 1.109746/sqrt(429.917542+0.001000), 
    0.998344/sqrt(304.108124+0.001000), 0.993199/sqrt(181.602570+0.001000), 0.931870/sqrt(170.925491+0.001000), 
    0.780714/sqrt(182.214890+0.001000), 0.989847/sqrt(233.119370+0.001000), 1.076850/sqrt(401.552277+0.001000), 
    1.029024/sqrt(276.647064+0.001000), 1.379483/sqrt(313.347229+0.001000), 0.800276/sqrt(128.175842+0.001000), 
    0.678067/sqrt(162.705017+0.001000), 0.772000/sqrt(120.726242+0.001000), 0.966365/sqrt(237.808350+0.001000), 
    0.899828/sqrt(168.053574+0.001000), 1.009023/sqrt(229.830429+0.001000), 0.752082/sqrt(128.490524+0.001000), 
    0.695853/sqrt(272.168213+0.001000), 0.545667/sqrt(104.030243+0.001000), 0.831317/sqrt(167.598801+0.001000), 
    0.697332/sqrt(144.572495+0.001000), 0.697032/sqrt(256.592499+0.001000), 0.615052/sqrt(255.284195+0.001000), 
    1.399968/sqrt(374.555634+0.001000), 0.832306/sqrt(181.555939+0.001000), 0.666253/sqrt(136.597168+0.001000), 
    0.628781/sqrt(253.873840+0.001000), 0.600896/sqrt(271.958710+0.001000), 0.951838/sqrt(211.771896+0.001000), 
    0.501607/sqrt(293.692352+0.001000), 0.595763/sqrt(198.839691+0.001000), 0.806052/sqrt(112.261307+0.001000), 
    1.124653/sqrt(374.547333+0.001000), 0.554314/sqrt(315.283844+0.001000), 0.806175/sqrt(141.371140+0.001000), 
    0.661420/sqrt(140.060989+0.001000), 0.907730/sqrt(246.466736+0.001000), 0.668531/sqrt(152.979126+0.001000), 
    0.659378/sqrt(359.071747+0.001000), 0.718741/sqrt(261.904449+0.001000), 1.088722/sqrt(434.069763+0.001000), 
    0.950709/sqrt(180.714127+0.001000), 0.756283/sqrt(277.995056+0.001000), 1.027728/sqrt(227.660904+0.001000), 
    1.054663/sqrt(288.262970+0.001000), 0.652434/sqrt(204.329834+0.001000)
    };
    static const float std_beta[] ={
    -0.003553-(7.513450*0.750679/sqrt(183.794113+0.001000)), -0.113982-(-1.733959*0.883491/sqrt(164.063431+0.001000)), 
    -0.244931-(6.069007*0.767102/sqrt(380.949829+0.001000)), -0.917127-(-2.817611*1.060649/sqrt(173.529373+0.001000)), 
    0.414260-(-8.384272*1.035712/sqrt(176.211868+0.001000)), 0.497535-(-7.459439*0.689433/sqrt(193.719467+0.001000)), 
    0.022538-(-6.539362*0.795292/sqrt(388.192474+0.001000)), -0.327344-(-1.955641*1.263755/sqrt(250.895554+0.001000)), 
    -0.176730-(6.629707*0.638560/sqrt(265.349518+0.001000)), -0.248331-(-0.240500*1.242243/sqrt(312.762268+0.001000)), 
    -0.085969-(1.381696*0.830333/sqrt(345.731171+0.001000)), -0.096846-(-0.230358*1.369249/sqrt(324.516022+0.001000)), 
    -0.303700-(4.181575*1.142395/sqrt(364.642822+0.001000)), -0.218073-(-5.350793*0.728114/sqrt(156.604492+0.001000)), 
    0.112554-(-11.320426*1.379342/sqrt(614.154053+0.001000)), -0.030382-(3.575962*0.931050/sqrt(134.725571+0.001000)), 
    -0.283472-(-8.592748*1.245805/sqrt(272.392303+0.001000)), -0.157676-(-4.546359*0.924392/sqrt(189.338379+0.001000)), 
    0.268008-(0.410444*0.797337/sqrt(165.425644+0.001000)), 0.015308-(13.611620*0.761169/sqrt(126.588882+0.001000)), 
    -0.040894-(3.830918*0.524586/sqrt(146.302658+0.001000)), 0.488679-(2.931701*0.982591/sqrt(312.185150+0.001000)), 
    0.127482-(5.948641*1.181889/sqrt(232.108673+0.001000)), 0.067942-(4.721095*0.759982/sqrt(250.576675+0.001000)), 
    0.005784-(-5.715681*1.008273/sqrt(444.304047+0.001000)), -0.548234-(-4.792869*0.954209/sqrt(337.202209+0.001000)), 
    0.026822-(2.111840*0.616673/sqrt(426.462097+0.001000)), -0.186497-(0.176706*0.741007/sqrt(372.745422+0.001000)), 
    0.021938-(-1.734860*0.831581/sqrt(219.632721+0.001000)), -0.034292-(-0.044441*1.008740/sqrt(224.835587+0.001000)), 
    -0.252395-(0.283341*1.105117/sqrt(286.654114+0.001000)), -0.549328-(8.549019*0.836408/sqrt(119.280594+0.001000)), 
    0.027767-(-4.271474*1.096259/sqrt(172.371063+0.001000)), -0.467292-(3.314645*0.920037/sqrt(311.613800+0.001000)), 
    -0.218453-(-0.241945*0.746397/sqrt(233.188690+0.001000)), -0.350669-(-1.066879*0.559681/sqrt(176.567230+0.001000)), 
    0.051617-(1.093692*0.919873/sqrt(215.138123+0.001000)), -0.394007-(10.208385*0.678649/sqrt(144.545792+0.001000)), 
    -0.132533-(-9.259041*1.056221/sqrt(165.122787+0.001000)), 0.098381-(-11.360151*1.092808/sqrt(208.811127+0.001000)), 
    0.435479-(7.224193*0.799793/sqrt(281.772003+0.001000)), 0.067288-(-2.370717*1.250921/sqrt(295.080200+0.001000)), 
    -0.042927-(-5.633814*0.583861/sqrt(170.089203+0.001000)), -0.000372-(-12.458466*0.982769/sqrt(187.606247+0.001000)), 
    -0.574752-(7.668047*0.903135/sqrt(238.560028+0.001000)), -0.126595-(3.027940*0.739905/sqrt(164.511703+0.001000)), 
    -0.191479-(7.240787*0.845826/sqrt(162.701477+0.001000)), -0.285105-(5.059854*0.978778/sqrt(147.430511+0.001000)), 
    -0.296607-(-2.629854*1.130841/sqrt(552.508545+0.001000)), 0.428791-(1.317607*0.931714/sqrt(195.784927+0.001000)), 
    -0.155760-(3.295574*0.755018/sqrt(158.225357+0.001000)), 0.093759-(2.046010*1.117247/sqrt(228.381180+0.001000)), 
    -0.088997-(5.932052*0.483002/sqrt(154.839417+0.001000)), 0.129928-(-4.534241*0.638272/sqrt(215.700760+0.001000)), 
    0.050897-(-0.564853*0.700161/sqrt(137.542084+0.001000)), 0.251275-(-5.822942*0.968244/sqrt(308.922699+0.001000)), 
    -0.182580-(4.833571*0.791358/sqrt(129.577652+0.001000)), -0.538454-(-3.978700*0.822475/sqrt(148.368942+0.001000)), 
    0.221872-(-1.320190*0.573975/sqrt(217.432190+0.001000)), -0.052834-(0.804723*0.760092/sqrt(157.014664+0.001000)), 
    -0.346642-(6.294308*0.926740/sqrt(129.143829+0.001000)), 0.273048-(0.048454*1.015330/sqrt(171.310257+0.001000)), 
    0.037117-(4.273716*0.843927/sqrt(145.185669+0.001000)), 0.313350-(-9.278516*0.951387/sqrt(442.708221+0.001000)), 
    -0.384213-(-5.450073*0.976816/sqrt(251.350067+0.001000)), -0.020138-(-9.591713*1.007399/sqrt(292.258545+0.001000)), 
    -0.112758-(-1.835788*0.790463/sqrt(176.712097+0.001000)), 0.151561-(-4.018631*0.965520/sqrt(223.962036+0.001000)), 
    -0.010634-(4.659858*0.765308/sqrt(280.928223+0.001000)), 0.131870-(3.340268*0.881511/sqrt(162.300522+0.001000)), 
    0.103328-(-5.629463*0.646607/sqrt(343.548370+0.001000)), 0.504928-(2.594019*0.765059/sqrt(260.238800+0.001000)), 
    0.309077-(-3.335888*0.756451/sqrt(211.347778+0.001000)), 0.541668-(-10.631449*0.904791/sqrt(220.740753+0.001000)), 
    0.362780-(-2.380172*0.707400/sqrt(230.141006+0.001000)), 0.007999-(-7.236015*0.881214/sqrt(119.680511+0.001000)), 
    -0.416247-(-6.023008*0.839123/sqrt(119.981644+0.001000)), -0.062170-(3.235381*0.719657/sqrt(186.148727+0.001000)), 
    0.175059-(-0.060069*0.901798/sqrt(171.535294+0.001000)), 0.178702-(1.159364*0.881881/sqrt(336.699768+0.001000)), 
    -0.152337-(6.551057*0.923361/sqrt(183.031204+0.001000)), 0.097708-(-9.599564*0.738768/sqrt(145.611206+0.001000)), 
    -0.313825-(4.590837*0.915717/sqrt(676.996643+0.001000)), 0.077984-(-7.883271*0.851957/sqrt(238.919739+0.001000)), 
    0.057091-(-1.241756*0.942640/sqrt(190.029053+0.001000)), -0.035902-(-2.204948*0.828030/sqrt(139.950699+0.001000)), 
    -0.096139-(3.538371*0.745401/sqrt(190.544037+0.001000)), 0.055832-(8.244592*0.472275/sqrt(131.150238+0.001000)), 
    0.061711-(1.891877*0.604055/sqrt(177.696762+0.001000)), 0.054724-(0.836008*0.570622/sqrt(196.696060+0.001000)), 
    -0.701848-(2.869808*0.734422/sqrt(166.135422+0.001000)), -0.046748-(-4.238773*1.482284/sqrt(529.812500+0.001000)), 
    0.328224-(10.136914*0.777455/sqrt(221.041718+0.001000)), 0.079245-(1.469051*0.746683/sqrt(170.835922+0.001000)), 
    0.049276-(-3.878717*0.593853/sqrt(241.865051+0.001000)), 0.680665-(-3.393661*1.417866/sqrt(311.951233+0.001000)), 
    -0.026626-(0.944119*0.955159/sqrt(216.845001+0.001000)), 0.098699-(-3.387503*0.662735/sqrt(291.371674+0.001000)), 
    -0.108685-(8.091837*1.013658/sqrt(286.259796+0.001000)), -0.168596-(-3.601925*0.535543/sqrt(228.341980+0.001000)), 
    0.027008-(4.010341*1.058384/sqrt(323.269135+0.001000)), 0.121510-(-3.371612*0.765232/sqrt(128.747406+0.001000)), 
    0.188289-(1.901745*1.058068/sqrt(292.964417+0.001000)), 0.050207-(-0.135981*1.035321/sqrt(210.046722+0.001000)), 
    -0.198226-(0.828740*0.755284/sqrt(125.695099+0.001000)), 0.022343-(-4.479970*0.720006/sqrt(169.339279+0.001000)), 
    0.212381-(-1.404994*1.134286/sqrt(228.900742+0.001000)), -0.007910-(-2.869821*0.771729/sqrt(176.532211+0.001000)), 
    -0.161225-(1.825177*0.791719/sqrt(344.509460+0.001000)), 0.038957-(-3.846102*0.969635/sqrt(177.923950+0.001000)), 
    -0.096888-(3.680713*0.593232/sqrt(109.846466+0.001000)), 0.465117-(-4.380745*0.962448/sqrt(386.825348+0.001000)), 
    0.067138-(-5.121235*0.864769/sqrt(282.304291+0.001000)), 0.227106-(5.756006*0.837655/sqrt(161.619507+0.001000)), 
    -0.221902-(-5.591243*0.764769/sqrt(188.072937+0.001000)), 0.038525-(2.791533*0.854159/sqrt(317.924988+0.001000)), 
    -0.259189-(-9.026723*0.669213/sqrt(134.178055+0.001000)), 0.196253-(-0.107045*0.854276/sqrt(147.206345+0.001000)), 
    0.158330-(-5.868849*0.884858/sqrt(247.181778+0.001000)), 0.067713-(-0.852693*0.868036/sqrt(138.920395+0.001000)), 
    -0.034784-(10.723008*1.031286/sqrt(181.047302+0.001000)), 0.323401-(-1.405015*0.972297/sqrt(162.536041+0.001000)), 
    -0.078662-(9.127281*0.791167/sqrt(131.247147+0.001000)), 0.246056-(-6.043415*0.639971/sqrt(156.897018+0.001000)), 
    -0.076085-(0.362425*0.981607/sqrt(371.865753+0.001000)), 0.036044-(-3.093765*0.745679/sqrt(116.281151+0.001000)), 
    0.094194-(-5.647485*0.864640/sqrt(269.746979+0.001000)), 0.156475-(-0.621308*0.920263/sqrt(299.290863+0.001000)), 
    -0.178494-(-1.890924*0.772694/sqrt(302.219086+0.001000)), 0.015789-(-1.870800*0.667799/sqrt(99.489182+0.001000)), 
    -0.227906-(0.581506*0.563844/sqrt(324.647522+0.001000)), 0.192492-(4.111344*1.011787/sqrt(367.474823+0.001000)), 
    -0.194092-(-7.451074*0.763452/sqrt(171.118851+0.001000)), 0.422941-(9.162976*0.929398/sqrt(187.320526+0.001000)), 
    -0.063945-(-6.013866*1.147684/sqrt(267.447937+0.001000)), -0.079950-(-2.676824*0.838234/sqrt(232.514481+0.001000)), 
    -0.389762-(1.655926*1.160890/sqrt(275.947113+0.001000)), 0.048013-(2.665063*0.757024/sqrt(205.323883+0.001000)), 
    0.377461-(-0.654256*0.966467/sqrt(237.733856+0.001000)), 0.355987-(-3.113373*0.781832/sqrt(210.660446+0.001000)), 
    0.230894-(1.691332*1.175303/sqrt(585.964600+0.001000)), 0.343178-(-2.142751*1.052473/sqrt(240.278839+0.001000)), 
    0.137063-(-10.442636*1.068428/sqrt(148.327393+0.001000)), 0.013019-(0.495289*0.957497/sqrt(343.977417+0.001000)), 
    -0.099573-(-8.482518*1.022751/sqrt(191.080093+0.001000)), 0.217042-(-1.827713*0.904565/sqrt(334.221191+0.001000)), 
    -0.430693-(-3.964911*0.852476/sqrt(585.205994+0.001000)), -0.027959-(0.865470*0.828229/sqrt(192.988953+0.001000)), 
    -0.167315-(11.204798*0.824247/sqrt(193.600967+0.001000)), 0.500826-(-10.304442*0.912234/sqrt(226.316116+0.001000)), 
    0.007920-(-0.328791*0.800982/sqrt(175.242310+0.001000)), 0.157511-(-4.300971*0.815355/sqrt(149.531067+0.001000)), 
    0.289499-(-1.000435*0.864880/sqrt(117.655594+0.001000)), 0.046251-(-5.528253*0.787019/sqrt(161.163773+0.001000)), 
    -0.138015-(-0.721057*0.758110/sqrt(164.183472+0.001000)), -0.115799-(-2.678993*0.936841/sqrt(322.148529+0.001000)), 
    0.377785-(-10.105843*0.901713/sqrt(169.413071+0.001000)), -0.485388-(2.043805*0.929386/sqrt(374.353210+0.001000)), 
    -0.051751-(0.166692*0.895230/sqrt(354.231750+0.001000)), -0.494289-(-8.065009*0.641060/sqrt(117.598351+0.001000)), 
    0.009198-(0.649268*0.642121/sqrt(152.959808+0.001000)), 0.291349-(-0.778255*1.028889/sqrt(302.255798+0.001000)), 
    -0.221390-(-0.833267*0.693197/sqrt(140.137512+0.001000)), -0.220417-(5.491363*1.410781/sqrt(307.468781+0.001000)), 
    0.085816-(4.556617*0.909177/sqrt(342.776154+0.001000)), 0.172128-(7.818807*0.832613/sqrt(166.124466+0.001000)), 
    -0.033216-(4.559536*0.778213/sqrt(134.412689+0.001000)), -0.183955-(4.780962*0.806682/sqrt(327.656555+0.001000)), 
    0.089116-(-5.032004*0.678508/sqrt(129.943268+0.001000)), -0.047275-(3.344230*0.451224/sqrt(269.683075+0.001000)), 
    0.111531-(-1.789688*0.847387/sqrt(144.273300+0.001000)), -0.270944-(-8.090681*0.849561/sqrt(141.191284+0.001000)), 
    -0.030915-(5.793533*0.939889/sqrt(167.914108+0.001000)), 0.083611-(1.540135*0.483493/sqrt(253.882492+0.001000)), 
    0.178552-(-1.983013*0.763394/sqrt(190.296707+0.001000)), -0.141987-(-4.386359*1.022586/sqrt(494.586151+0.001000)), 
    0.151794-(-4.472772*0.986355/sqrt(290.245026+0.001000)), -0.327638-(-0.722393*0.783376/sqrt(171.187531+0.001000)), 
    -0.014238-(-11.577736*0.769421/sqrt(199.859833+0.001000)), -0.582805-(-1.049136*0.779122/sqrt(248.165573+0.001000)), 
    0.237419-(0.256257*1.092562/sqrt(308.564392+0.001000)), -0.322417-(1.286547*0.821335/sqrt(138.844467+0.001000)), 
    -0.032852-(1.223277*0.735155/sqrt(179.932343+0.001000)), 0.256557-(-4.523726*1.021632/sqrt(260.388611+0.001000)), 
    -0.321093-(8.262412*0.588541/sqrt(105.271988+0.001000)), -0.381425-(11.516202*0.686936/sqrt(295.504700+0.001000)), 
    -0.222849-(3.580236*0.639154/sqrt(221.417404+0.001000)), 0.034837-(-3.989834*0.984606/sqrt(151.790100+0.001000)), 
    0.013683-(-5.191483*0.755497/sqrt(167.946686+0.001000)), -0.003870-(-3.204587*0.858086/sqrt(330.408051+0.001000)), 
    -0.225342-(8.473534*0.917839/sqrt(153.291168+0.001000)), -0.009907-(-0.211300*0.742095/sqrt(167.956116+0.001000)), 
    -0.285905-(-4.676422*0.865658/sqrt(212.055252+0.001000)), -0.133171-(5.129649*0.656918/sqrt(181.683990+0.001000)), 
    -0.019113-(-7.219835*0.626508/sqrt(172.164993+0.001000)), 0.045867-(-7.053872*0.577355/sqrt(125.397110+0.001000)), 
    -0.145246-(-3.960975*0.789151/sqrt(293.684082+0.001000)), -0.411061-(1.929006*1.040580/sqrt(249.099350+0.001000)), 
    -0.094946-(-6.641572*0.816116/sqrt(146.876068+0.001000)), 0.495862-(-8.381732*1.225533/sqrt(266.165588+0.001000)), 
    -0.257639-(-11.810394*0.957588/sqrt(140.091690+0.001000)), -0.239535-(-1.176494*0.855418/sqrt(138.797089+0.001000)), 
    -0.416028-(5.909955*0.850987/sqrt(289.454987+0.001000)), -0.312564-(7.678150*0.693183/sqrt(176.870728+0.001000)), 
    0.074178-(5.575513*0.981261/sqrt(271.303680+0.001000)), 0.348482-(4.606457*0.708909/sqrt(231.376266+0.001000)), 
    0.233243-(1.066112*0.758847/sqrt(154.759399+0.001000)), -0.141891-(7.534442*0.896347/sqrt(218.770432+0.001000)), 
    0.275080-(-2.710631*1.079516/sqrt(252.339096+0.001000)), -0.326018-(0.834380*0.989293/sqrt(166.481277+0.001000)), 
    0.274258-(4.413506*1.182887/sqrt(241.511749+0.001000)), 0.570604-(-3.520494*0.900826/sqrt(184.146210+0.001000)), 
    -0.227774-(2.101460*1.032980/sqrt(324.961548+0.001000)), 0.265216-(-3.697196*1.154450/sqrt(219.549362+0.001000)), 
    0.183903-(3.833255*0.708216/sqrt(166.252991+0.001000)), -0.073657-(1.723403*1.011590/sqrt(255.986542+0.001000)), 
    0.176032-(-2.038438*0.873168/sqrt(117.639984+0.001000)), 0.321076-(14.731457*1.164196/sqrt(235.918930+0.001000)), 
    -0.480890-(5.966892*0.855211/sqrt(115.515320+0.001000)), -0.074736-(-2.917309*1.219751/sqrt(189.480545+0.001000)), 
    -0.656417-(-8.144005*0.946711/sqrt(215.599930+0.001000)), -0.125149-(4.926769*0.760800/sqrt(420.280334+0.001000)), 
    0.095933-(1.225697*0.766619/sqrt(190.543182+0.001000)), 0.034277-(-1.872369*0.991689/sqrt(287.125061+0.001000)), 
    -0.062021-(2.203731*0.651705/sqrt(169.049850+0.001000)), 0.196003-(-0.577535*1.116166/sqrt(298.970276+0.001000)), 
    0.444305-(5.033869*0.643305/sqrt(312.795746+0.001000)), 0.447978-(3.875887*0.868313/sqrt(166.564041+0.001000)), 
    0.199676-(3.651541*0.978477/sqrt(195.337555+0.001000)), 0.203871-(5.787081*0.603936/sqrt(324.610687+0.001000)), 
    0.201816-(-8.798192*0.658405/sqrt(166.048401+0.001000)), 0.393333-(3.066662*0.678225/sqrt(215.069962+0.001000)), 
    -0.306704-(1.671075*0.630645/sqrt(165.029099+0.001000)), -0.170941-(-1.011487*0.827753/sqrt(118.744423+0.001000)), 
    0.309159-(8.695929*0.853326/sqrt(157.568619+0.001000)), -0.580474-(-2.605772*0.778496/sqrt(172.984497+0.001000)), 
    0.257954-(9.615359*0.773650/sqrt(243.219482+0.001000)), 0.285619-(-6.177773*0.594038/sqrt(212.615112+0.001000)), 
    0.121999-(7.675441*1.100457/sqrt(173.186539+0.001000)), -0.178270-(2.364562*0.984096/sqrt(154.892654+0.001000)), 
    0.484037-(-1.809583*0.759250/sqrt(283.509277+0.001000)), 0.123679-(9.049217*0.727384/sqrt(461.723816+0.001000)), 
    -0.115694-(4.925934*0.714637/sqrt(342.375580+0.001000)), -0.105246-(-5.313800*0.775013/sqrt(119.609520+0.001000)), 
    0.191943-(7.621212*0.622289/sqrt(114.869057+0.001000)), 0.224016-(-2.358316*0.932085/sqrt(205.057159+0.001000)), 
    -0.151507-(2.807096*0.906272/sqrt(224.416061+0.001000)), 0.068229-(5.899875*0.912606/sqrt(348.997711+0.001000)), 
    0.114833-(0.198825*1.185480/sqrt(278.326538+0.001000)), -0.183484-(2.120253*0.795552/sqrt(229.161758+0.001000)), 
    0.216326-(5.257945*0.741116/sqrt(162.534195+0.001000)), -0.446479-(-0.058301*1.027286/sqrt(232.497894+0.001000)), 
    -0.134228-(-4.524710*1.164752/sqrt(255.026428+0.001000)), -0.007913-(-5.967572*0.819287/sqrt(152.933517+0.001000)), 
    -0.113129-(-2.052916*0.864285/sqrt(213.287308+0.001000)), -0.290176-(2.523608*0.844032/sqrt(331.315826+0.001000)), 
    0.296697-(8.510181*1.032779/sqrt(189.215759+0.001000)), -0.062658-(0.590420*0.642900/sqrt(216.955093+0.001000)), 
    0.214749-(1.695374*0.820709/sqrt(177.619186+0.001000)), 0.097632-(-5.501805*0.639207/sqrt(139.203415+0.001000)), 
    0.290725-(-2.604642*0.703655/sqrt(122.211098+0.001000)), -0.694809-(1.498556*1.013082/sqrt(195.002747+0.001000)), 
    -0.047048-(-5.135473*0.728469/sqrt(199.258057+0.001000)), -0.276604-(2.732759*0.838234/sqrt(236.063385+0.001000)), 
    -0.511516-(-4.721548*0.850901/sqrt(144.646469+0.001000)), -0.085653-(3.933680*0.895061/sqrt(220.762985+0.001000)), 
    -0.074782-(-6.919011*1.066899/sqrt(140.404694+0.001000)), -0.357029-(3.156869*0.991905/sqrt(187.856583+0.001000)), 
    0.193897-(-5.237599*0.780839/sqrt(152.347382+0.001000)), -0.168579-(3.493845*0.803460/sqrt(182.404785+0.001000)), 
    0.299055-(7.066588*0.928889/sqrt(206.915024+0.001000)), -0.212583-(4.647986*0.752942/sqrt(223.656967+0.001000)), 
    -0.401866-(-3.110532*0.929372/sqrt(272.315857+0.001000)), 0.149506-(-2.131818*0.759226/sqrt(187.482132+0.001000)), 
    0.364783-(-2.492481*1.097139/sqrt(309.978729+0.001000)), 0.498763-(-0.945844*0.967777/sqrt(227.527283+0.001000)), 
    0.151682-(-3.902331*0.821351/sqrt(198.799713+0.001000)), 0.311790-(-6.357635*0.795732/sqrt(206.003098+0.001000)), 
    -0.402466-(-0.808120*0.937088/sqrt(234.289017+0.001000)), 0.095687-(-11.062550*1.000904/sqrt(259.974609+0.001000)), 
    -0.018543-(1.700574*0.959789/sqrt(176.219589+0.001000)), -0.035405-(0.295450*0.642039/sqrt(107.895683+0.001000)), 
    -0.191383-(-0.598134*0.807803/sqrt(139.320526+0.001000)), 0.097167-(3.499047*0.944208/sqrt(197.539520+0.001000)), 
    0.469192-(2.991801*0.911537/sqrt(247.027771+0.001000)), -0.005709-(2.091325*0.705633/sqrt(236.083633+0.001000)), 
    0.219454-(7.651162*0.820100/sqrt(148.707275+0.001000)), 0.249657-(3.151312*0.832211/sqrt(153.052582+0.001000)), 
    0.105334-(-0.879395*1.108139/sqrt(203.311295+0.001000)), 0.038029-(-3.199361*0.973207/sqrt(126.663445+0.001000)), 
    0.140000-(-3.621229*0.783927/sqrt(256.900269+0.001000)), 0.288817-(-2.908979*0.968620/sqrt(213.466599+0.001000)), 
    0.146131-(-8.540620*0.567436/sqrt(200.764175+0.001000)), -0.263874-(-1.056896*0.584065/sqrt(154.437881+0.001000)), 
    0.160094-(-5.287860*0.683100/sqrt(167.825409+0.001000)), 0.064598-(-4.770269*0.774789/sqrt(150.928970+0.001000)), 
    -0.277350-(-0.958932*0.796204/sqrt(283.324463+0.001000)), 0.307417-(-1.341767*1.143420/sqrt(302.671875+0.001000)), 
    0.230806-(0.506983*0.734889/sqrt(312.791779+0.001000)), -0.220182-(-3.687269*0.568994/sqrt(120.704071+0.001000)), 
    -0.230530-(-6.842173*1.224349/sqrt(178.434189+0.001000)), -0.021220-(5.584605*1.008015/sqrt(213.925735+0.001000)), 
    -0.270545-(-2.325237*0.732257/sqrt(191.541306+0.001000)), 0.163005-(-0.472081*0.595264/sqrt(213.422714+0.001000)), 
    -0.334118-(3.400972*1.262220/sqrt(339.070129+0.001000)), -0.008632-(2.737472*0.955023/sqrt(210.737091+0.001000)), 
    0.629093-(10.004645*0.865259/sqrt(228.556381+0.001000)), 0.200721-(-0.676376*0.580252/sqrt(233.801025+0.001000)), 
    -0.147537-(2.631239*0.946033/sqrt(315.957397+0.001000)), -0.414863-(1.955801*0.791448/sqrt(145.878601+0.001000)), 
    -0.245107-(4.205220*0.888759/sqrt(219.888046+0.001000)), -0.113032-(3.791171*1.063231/sqrt(260.517944+0.001000)), 
    0.367285-(-7.041756*0.855176/sqrt(305.798523+0.001000)), 0.062703-(0.169302*0.637525/sqrt(265.963318+0.001000)), 
    0.273126-(-3.202875*0.920539/sqrt(140.829468+0.001000)), 0.091296-(0.531279*1.217759/sqrt(499.735779+0.001000)), 
    -0.351233-(6.365417*0.972019/sqrt(260.562134+0.001000)), 0.127421-(1.526020*0.807694/sqrt(264.221741+0.001000)), 
    0.507559-(-3.831825*1.131782/sqrt(150.449936+0.001000)), 0.067890-(-8.826534*0.547888/sqrt(128.066177+0.001000)), 
    -0.041132-(-1.578627*0.601016/sqrt(123.907738+0.001000)), -0.353552-(6.384361*0.912180/sqrt(187.605148+0.001000)), 
    -0.439426-(-6.975495*1.305207/sqrt(214.531418+0.001000)), 0.393085-(12.992022*1.413387/sqrt(496.952911+0.001000)), 
    -0.127974-(2.907539*0.722100/sqrt(127.442734+0.001000)), 0.433432-(9.029805*0.597080/sqrt(187.809494+0.001000)), 
    0.244715-(4.574963*1.021203/sqrt(163.475479+0.001000)), 0.238574-(0.988321*0.850185/sqrt(189.675598+0.001000)), 
    -0.720361-(-5.751041*1.044190/sqrt(218.386765+0.001000)), 0.423202-(3.709765*1.043707/sqrt(254.593292+0.001000)), 
    0.017547-(5.562915*1.005548/sqrt(162.489182+0.001000)), 0.443394-(3.155526*0.842746/sqrt(213.385971+0.001000)), 
    -0.023302-(7.383412*0.718463/sqrt(131.285583+0.001000)), 0.093213-(-4.600108*0.691844/sqrt(168.638901+0.001000)), 
    0.088011-(-0.844502*0.858130/sqrt(232.494751+0.001000)), -0.159532-(-4.101493*0.834409/sqrt(214.154556+0.001000)), 
    0.050671-(-11.456127*0.839877/sqrt(152.938919+0.001000)), 0.368660-(-0.612629*0.739728/sqrt(177.528595+0.001000)), 
    0.208834-(3.058635*1.121459/sqrt(172.980423+0.001000)), 0.409016-(-3.282882*1.001263/sqrt(173.338684+0.001000)), 
    -0.537321-(-2.220544*1.028222/sqrt(188.818085+0.001000)), 0.129106-(2.604176*1.032738/sqrt(336.843140+0.001000)), 
    -0.077089-(-9.303391*0.728425/sqrt(168.436890+0.001000)), -0.397437-(-3.125941*1.170665/sqrt(413.908630+0.001000)), 
    0.199885-(-9.245988*0.733893/sqrt(358.960754+0.001000)), -0.190797-(-6.672394*0.752217/sqrt(165.759018+0.001000)), 
    -0.172393-(6.333338*0.846335/sqrt(169.316849+0.001000)), -0.151109-(3.922552*0.625165/sqrt(164.740646+0.001000)), 
    -0.072208-(3.310564*0.732072/sqrt(181.285980+0.001000)), -0.032227-(3.903495*0.774820/sqrt(133.677734+0.001000)), 
    0.120534-(-0.365062*0.730029/sqrt(131.182907+0.001000)), 0.096353-(-6.418919*0.655362/sqrt(133.641968+0.001000)), 
    -0.097748-(-3.776734*0.550855/sqrt(148.404877+0.001000)), 0.570217-(4.730480*1.249618/sqrt(320.903839+0.001000)), 
    0.283225-(-4.152275*1.261876/sqrt(321.601776+0.001000)), 0.059758-(-3.336315*0.880822/sqrt(302.192230+0.001000)), 
    0.327270-(5.846108*1.290730/sqrt(313.298370+0.001000)), 0.488123-(-2.351249*0.936485/sqrt(445.619629+0.001000)), 
    -0.305506-(3.564491*0.877705/sqrt(226.979065+0.001000)), 0.025028-(-4.130098*0.870635/sqrt(261.694885+0.001000)), 
    -0.389701-(-1.938751*0.812991/sqrt(215.816528+0.001000)), 0.389921-(6.936353*0.659452/sqrt(237.270386+0.001000)), 
    0.527937-(-2.312906*0.721113/sqrt(174.470367+0.001000)), 0.042261-(0.347469*0.670038/sqrt(204.275284+0.001000)), 
    -0.614316-(-5.761781*1.019209/sqrt(293.082336+0.001000)), 0.165326-(14.592692*1.023655/sqrt(140.578430+0.001000)), 
    0.237274-(4.982136*0.703914/sqrt(151.083664+0.001000)), 0.182600-(-1.448876*0.605411/sqrt(113.446045+0.001000)), 
    0.013136-(0.449195*0.918798/sqrt(241.848846+0.001000)), -0.380925-(-17.866213*0.695200/sqrt(221.263504+0.001000)), 
    -0.209724-(-2.513061*0.822500/sqrt(153.451065+0.001000)), 0.305080-(6.242728*0.662373/sqrt(132.390976+0.001000)), 
    0.316212-(-4.063590*0.692816/sqrt(151.326675+0.001000)), 0.059206-(-0.693703*0.968449/sqrt(313.220581+0.001000)), 
    0.152113-(-2.328233*1.005165/sqrt(201.573257+0.001000)), -0.246359-(4.111910*0.677219/sqrt(111.363449+0.001000)), 
    0.040758-(-0.771021*0.939789/sqrt(431.797089+0.001000)), -0.241134-(1.637438*0.744205/sqrt(281.766632+0.001000)), 
    0.034933-(0.091464*0.854380/sqrt(284.521179+0.001000)), 0.275763-(-6.549645*0.736849/sqrt(161.276581+0.001000)), 
    0.093466-(-2.850569*0.939497/sqrt(157.420761+0.001000)), -0.131781-(7.899288*0.785954/sqrt(172.549011+0.001000)), 
    0.204146-(-9.431077*0.939585/sqrt(255.853134+0.001000)), 0.211239-(5.925895*0.592331/sqrt(148.325165+0.001000)), 
    -0.021372-(12.690865*0.924299/sqrt(192.365005+0.001000)), 0.060790-(-6.618912*0.659987/sqrt(132.378586+0.001000)), 
    0.052059-(-7.259269*0.480913/sqrt(254.425690+0.001000)), -0.120687-(4.964230*0.911636/sqrt(219.446991+0.001000)), 
    0.100153-(1.661494*0.755067/sqrt(146.003326+0.001000)), 0.093736-(8.493638*0.957869/sqrt(161.232895+0.001000)), 
    -0.099904-(-4.855736*0.480797/sqrt(184.927689+0.001000)), 0.384293-(-3.862579*0.880447/sqrt(253.899750+0.001000)), 
    0.253714-(-1.416203*0.764348/sqrt(271.511322+0.001000)), -0.110401-(-6.321619*0.859953/sqrt(236.679382+0.001000)), 
    0.311089-(-2.012561*1.289335/sqrt(156.268494+0.001000)), 0.100556-(0.374074*0.884973/sqrt(203.797302+0.001000)), 
    -0.585672-(7.704712*0.908592/sqrt(192.782654+0.001000)), -0.078545-(-6.384918*1.149788/sqrt(365.497589+0.001000)), 
    -0.598969-(-0.775138*0.745504/sqrt(172.129257+0.001000)), 0.135989-(1.761131*0.798645/sqrt(195.216217+0.001000)), 
    0.411370-(1.184048*0.740087/sqrt(99.041069+0.001000)), -0.179602-(3.570510*0.628827/sqrt(138.007126+0.001000)), 
    0.075127-(-2.611588*0.967902/sqrt(115.522758+0.001000)), 0.123901-(-0.235504*0.954328/sqrt(167.889160+0.001000)), 
    0.439586-(-11.382547*0.968058/sqrt(218.115967+0.001000)), 0.343436-(-1.734790*1.038580/sqrt(231.173157+0.001000)), 
    0.055059-(9.128868*0.715082/sqrt(167.091446+0.001000)), 0.087589-(-4.671193*0.728947/sqrt(249.572754+0.001000)), 
    -0.589806-(-0.436340*1.063136/sqrt(300.927460+0.001000)), -0.132419-(-3.575748*0.580270/sqrt(155.068359+0.001000)), 
    0.056584-(5.002534*0.965664/sqrt(219.821930+0.001000)), -0.418072-(3.486754*0.989843/sqrt(172.390121+0.001000)), 
    -0.197581-(-11.106293*0.785234/sqrt(328.032684+0.001000)), 0.324361-(1.725314*1.099262/sqrt(254.171219+0.001000)), 
    -0.136187-(8.036701*0.939356/sqrt(181.775650+0.001000)), -0.551755-(-4.329738*0.981123/sqrt(243.654846+0.001000)), 
    -0.046660-(-0.765944*0.833147/sqrt(209.621094+0.001000)), 0.117849-(7.268943*0.903759/sqrt(201.587112+0.001000)), 
    -0.365280-(2.131552*0.771366/sqrt(204.177383+0.001000)), -0.043395-(-3.216249*0.890480/sqrt(131.167694+0.001000)), 
    -0.040229-(7.084052*0.957790/sqrt(240.870239+0.001000)), -0.162741-(-9.879046*0.909646/sqrt(184.503494+0.001000)), 
    -0.236131-(0.703210*0.937562/sqrt(215.878983+0.001000)), -0.328251-(-5.575998*0.887373/sqrt(203.578369+0.001000)), 
    -0.230663-(1.106800*0.841901/sqrt(209.516937+0.001000)), -0.014919-(1.174497*1.283488/sqrt(347.486969+0.001000)), 
    -0.248544-(-1.497623*0.644061/sqrt(157.482239+0.001000)), 0.292292-(14.073264*0.574426/sqrt(127.698456+0.001000)), 
    -0.344889-(2.336848*0.978431/sqrt(289.202576+0.001000)), -0.378814-(1.342726*0.901822/sqrt(127.744148+0.001000)), 
    0.388639-(-8.325824*1.325540/sqrt(267.297150+0.001000)), 0.033930-(-3.270352*0.811510/sqrt(160.334991+0.001000)), 
    -0.262849-(-0.543510*0.963087/sqrt(254.171127+0.001000)), -0.410076-(1.617158*1.005993/sqrt(219.382935+0.001000)), 
    0.194985-(9.844654*1.143173/sqrt(295.502075+0.001000)), -0.550085-(-0.054909*1.157817/sqrt(258.521179+0.001000)), 
    0.096695-(3.313095*0.840435/sqrt(198.735291+0.001000)), -0.184490-(-7.027001*0.686701/sqrt(303.727417+0.001000)), 
    -0.020456-(-3.164062*1.102462/sqrt(243.426437+0.001000)), -0.471904-(-6.108077*1.151578/sqrt(360.945770+0.001000)), 
    0.015844-(-1.668238*1.044169/sqrt(738.929382+0.001000)), -0.036652-(-0.035788*1.212885/sqrt(293.488281+0.001000)), 
    0.019934-(11.075277*1.073056/sqrt(220.215714+0.001000)), 0.413812-(0.508582*0.749174/sqrt(218.720383+0.001000)), 
    -0.162602-(-6.067569*0.824449/sqrt(167.610168+0.001000)), -0.141376-(-2.554089*0.813008/sqrt(179.341019+0.001000)), 
    -0.415165-(-0.224507*1.024933/sqrt(191.288162+0.001000)), 0.329483-(-0.662258*1.068076/sqrt(196.418900+0.001000)), 
    0.177553-(0.842794*0.773342/sqrt(279.087128+0.001000)), 0.071964-(5.085286*0.640724/sqrt(132.193268+0.001000)), 
    0.092355-(3.554665*1.193822/sqrt(285.091522+0.001000)), -0.076584-(-4.682446*0.915898/sqrt(131.874664+0.001000)), 
    0.158479-(0.470322*0.996405/sqrt(224.109360+0.001000)), 0.252029-(2.194960*0.981799/sqrt(263.265991+0.001000)), 
    -0.000264-(5.153271*0.786942/sqrt(160.689423+0.001000)), -0.045146-(-4.889134*1.004326/sqrt(172.114822+0.001000)), 
    -0.354842-(-2.117532*0.958180/sqrt(282.646149+0.001000)), -0.321932-(10.013783*0.634556/sqrt(191.222473+0.001000)), 
    0.367778-(-5.044251*0.974911/sqrt(191.975204+0.001000)), -0.323028-(-3.811424*0.935749/sqrt(151.939392+0.001000)), 
    0.040858-(-0.367777*1.039169/sqrt(220.191940+0.001000)), 0.211976-(2.402083*0.755844/sqrt(216.537781+0.001000)), 
    0.316120-(0.192043*1.055384/sqrt(223.891144+0.001000)), 0.224912-(-1.948820*0.660030/sqrt(112.261620+0.001000)), 
    0.002163-(-9.585604*0.765314/sqrt(186.519958+0.001000)), 0.970053-(4.168911*1.025622/sqrt(216.054016+0.001000)), 
    -0.033581-(0.612765*0.855197/sqrt(200.431610+0.001000)), 0.353706-(-3.710711*1.109746/sqrt(429.917542+0.001000)), 
    -0.008153-(2.240007*0.998344/sqrt(304.108124+0.001000)), 0.137312-(7.249766*0.993199/sqrt(181.602570+0.001000)), 
    0.213162-(1.139139*0.931870/sqrt(170.925491+0.001000)), -0.581723-(-6.912237*0.780714/sqrt(182.214890+0.001000)), 
    -0.555457-(-1.788494*0.989847/sqrt(233.119370+0.001000)), 0.357779-(-4.745369*1.076850/sqrt(401.552277+0.001000)), 
    0.116870-(0.531076*1.029024/sqrt(276.647064+0.001000)), -0.393499-(11.286892*1.379483/sqrt(313.347229+0.001000)), 
    0.036607-(4.967017*0.800276/sqrt(128.175842+0.001000)), -0.215395-(-1.572637*0.678067/sqrt(162.705017+0.001000)), 
    0.133433-(-3.703158*0.772000/sqrt(120.726242+0.001000)), 0.372157-(0.451215*0.966365/sqrt(237.808350+0.001000)), 
    -0.111480-(5.369642*0.899828/sqrt(168.053574+0.001000)), -0.401697-(10.226728*1.009023/sqrt(229.830429+0.001000)), 
    0.264840-(-0.258413*0.752082/sqrt(128.490524+0.001000)), 0.248749-(2.249463*0.695853/sqrt(272.168213+0.001000)), 
    -0.001846-(-2.520727*0.545667/sqrt(104.030243+0.001000)), 0.533539-(-2.567457*0.831317/sqrt(167.598801+0.001000)), 
    -0.029693-(1.681528*0.697332/sqrt(144.572495+0.001000)), -0.069524-(15.537286*0.697032/sqrt(256.592499+0.001000)), 
    -0.030082-(1.084416*0.615052/sqrt(255.284195+0.001000)), 0.548257-(3.960269*1.399968/sqrt(374.555634+0.001000)), 
    0.556980-(-2.139799*0.832306/sqrt(181.555939+0.001000)), 0.129244-(6.135177*0.666253/sqrt(136.597168+0.001000)), 
    0.305647-(8.539069*0.628781/sqrt(253.873840+0.001000)), 0.203787-(-0.246193*0.600896/sqrt(271.958710+0.001000)), 
    -0.020855-(-13.500644*0.951838/sqrt(211.771896+0.001000)), -0.232064-(4.327055*0.501607/sqrt(293.692352+0.001000)), 
    -0.235570-(-4.696512*0.595763/sqrt(198.839691+0.001000)), -0.289880-(1.598815*0.806052/sqrt(112.261307+0.001000)), 
    0.703012-(4.720378*1.124653/sqrt(374.547333+0.001000)), 0.074706-(10.207825*0.554314/sqrt(315.283844+0.001000)), 
    -0.076469-(-3.954866*0.806175/sqrt(141.371140+0.001000)), 0.346579-(-2.497632*0.661420/sqrt(140.060989+0.001000)), 
    -0.033319-(-2.829802*0.907730/sqrt(246.466736+0.001000)), 0.158528-(-8.244674*0.668531/sqrt(152.979126+0.001000)), 
    0.303298-(4.590652*0.659378/sqrt(359.071747+0.001000)), 0.061951-(-1.609441*0.718741/sqrt(261.904449+0.001000)), 
    0.031262-(1.753278*1.088722/sqrt(434.069763+0.001000)), -0.032098-(7.899911*0.950709/sqrt(180.714127+0.001000)), 
    0.262103-(-9.825538*0.756283/sqrt(277.995056+0.001000)), -0.113018-(1.336647*1.027728/sqrt(227.660904+0.001000)), 
    0.518811-(2.387105*1.054663/sqrt(288.262970+0.001000)), 0.013690-(2.830867*0.652434/sqrt(204.329834+0.001000)), 
  
    };

    static const batch_normalization_layer_t norm = { 512, inv_gamma_dev, std_beta };
    return norm;
}

dense_layer_t init_dense_80_data(void){

    static neuron_t neurons[10];

    static const float weights0[] ={
    -0.053799521178007126, 0.0391012579202652, 0.044197116047143936, 0.09058086574077606, 
    -0.1996782124042511, -0.0290922112762928, 0.017703037708997726, 0.01927761919796467, 
    -0.03636709600687027, -0.02242385223507881, 0.023856574669480324, -0.003811771748587489, 
    -0.010106515139341354, 0.021893691271543503, -0.027682334184646606, -0.0037294412031769753, 
    -0.10048559308052063, 0.0643405094742775, 0.017663488164544106, -0.09376600384712219, 
    0.003941421862691641, 0.12111914157867432, -0.09634526073932648, 0.030738787725567818, 
    0.08253111690282822, 0.042069002985954285, 0.013901189900934696, -0.10974830389022827, 
    0.0359342060983181, 0.04233838990330696, -0.10916368663311005, 0.09334241598844528, 
    0.049526337534189224, 0.008161039091646671, 0.10018468648195267, 0.01655191369354725, 
    -0.04173513501882553, -0.04202139750123024, 0.06234883889555931, -0.008336440660059452, 
    -0.11769949644804001, 0.007683729752898216, -0.15239480137825012, 0.07206880301237106, 
    -0.03582784906029701, 0.07871296256780624, 0.039223141968250275, -0.1279524266719818, 
    0.04914926737546921, -0.09309644252061844, -0.061997223645448685, -0.0698564201593399, 
    0.10735340416431427, -0.11735689640045166, -0.017755912616848946, -0.021107425913214684, 
    0.00803457386791706, -0.10023607313632965, 0.006335975136607885, -0.14823630452156067, 
    0.026811841875314713, 0.10473156720399857, -0.030973300337791443, 0.07921828329563141, 
    -0.07110051065683365, 0.0895208865404129, -0.08332182466983795, -0.0640491172671318, 
    0.032004550099372864, 0.03586957976222038, 0.11408092826604843, -0.1194971576333046, 
    -0.09010137617588043, 0.036755312234163284, 0.05221736058592796, -0.03011508472263813, 
    0.024419473484158516, 0.045941367745399475, 0.042271342128515244, -0.03258180618286133, 
    -0.12136498093605042, -0.034960273653268814, -0.00033271577558480203, 0.06435837596654892, 
    -0.020520374178886414, -0.030282583087682724, 0.01977018266916275, -0.0016507452819496393, 
    -0.09146970510482788, 0.14862394332885742, -0.11627817898988724, -0.0852411538362503, 
    -0.04035039618611336, 0.02686009183526039, 0.06420226395130157, -0.056125540286302567, 
    0.18525658547878265, -0.13279078900814056, -0.054293982684612274, 0.11131136119365692, 
    0.06014389544725418, 0.17805124819278717, -0.0502728596329689, 0.02090507000684738, 
    0.06370078027248383, -0.036099426448345184, -0.04831624776124954, -0.08118883520364761, 
    0.004358113743364811, 0.07595787942409515, 0.036160290241241455, -0.04479796066880226, 
    0.06345151364803314, 0.02151774801313877, -0.0052743395790457726, -0.10251864045858383, 
    -0.07079245150089264, 0.02435002103447914, 0.0286545567214489, -0.008382554166018963, 
    -0.03127741813659668, 0.03195631876587868, 0.06356824189424515, 0.05997602641582489, 
    0.027685092762112617, 0.026458093896508217, -0.049042876809835434, -0.029776491224765778, 
    0.036233145743608475, -0.05405838042497635, -0.004063318949192762, -0.01761443167924881, 
    0.01568654552102089, -0.041078999638557434, -0.11514264345169067, 0.0038363314233720303, 
    -0.06416471302509308, -0.10462558269500732, -0.04248442128300667, -0.12162234634160995, 
    -0.0412529893219471, 0.0913291871547699, -0.04663960635662079, -0.0630597174167633, 
    -0.08637218177318573, -0.06316196173429489, 0.029353460296988487, 0.003912980202585459, 
    0.009126491844654083, -0.011997519060969353, -0.1530105620622635, -0.009569009765982628, 
    0.022372568026185036, -0.08625045418739319, -0.030976125970482826, 0.11759620904922485, 
    0.06540973484516144, 0.039119966328144073, 0.07082600146532059, -0.02187214232981205, 
    -0.05739060044288635, 0.07883477210998535, -0.012196061201393604, -0.05372656509280205, 
    -0.022837162017822266, -0.0774618536233902, -0.04704437404870987, -0.026506664231419563, 
    -0.01826176419854164, -0.043439317494630814, -0.03029932640492916, -0.05578199028968811, 
    -0.01569783315062523, 0.029030699282884598, 0.046475961804389954, 0.036429792642593384, 
    -0.07860738039016724, 0.0455496720969677, 0.07548016309738159, 0.08068297803401947, 
    0.055906992405653, 0.04029560834169388, 0.034862272441387177, -0.0031703978311270475, 
    0.013154338113963604, -0.042089641094207764, 0.07462447881698608, -0.018388863652944565, 
    -0.028757227584719658, 0.15058302879333496, 0.10620544105768204, 0.025729555636644363, 
    0.043621670454740524, 0.03198304399847984, 0.08353639394044876, 0.09254447370767593, 
    -0.06720534712076187, -0.054356686770915985, 0.06906075030565262, 0.07167427241802216, 
    0.025454316288232803, -0.03126317262649536, -0.052401769906282425, 0.032278407365083694, 
    -0.032501764595508575, -0.0253127571195364, -0.08734314888715744, 0.05449588596820831, 
    -0.10850828140974045, -0.056180767714977264, 0.02761174738407135, 0.0054801516234874725, 
    0.007086826488375664, -0.06901711970567703, -0.1780395358800888, 0.043610621243715286, 
    0.06412102282047272, 0.032517775893211365, -0.06986980885267258, 0.051291368901729584, 
    -0.05447983741760254, -0.0723385289311409, -0.004870396573096514, 0.012411078438162804, 
    0.08318230509757996, -0.027555324137210846, -0.08415310829877853, -0.20144790410995483, 
    0.03629393130540848, -0.057342905551195145, -0.01678437739610672, 0.0014689734671264887, 
    -0.11043484508991241, -0.0031734819058328867, -0.03580990806221962, -0.04195785149931908, 
    0.12593083083629608, 0.04783874377608299, 0.007339440751820803, 0.10006611794233322, 
    0.07292363792657852, -0.034340761601924896, 0.03164313733577728, 0.058675363659858704, 
    -0.028538405895233154, -0.01702818274497986, 0.05373814329504967, -0.0721564069390297, 
    0.13203832507133484, -0.009039206430315971, -0.07234740257263184, 0.03910098224878311, 
    0.11434857547283173, 0.04632768779993057, 0.014253227971494198, 0.022984685376286507, 
    -0.05898333713412285, 0.05660953372716904, 0.05645349621772766, 0.038174163550138474, 
    0.08635770529508591, 0.03878937289118767, -0.04493501037359238, 0.03165312483906746, 
    -0.1955903023481369, -0.03731011599302292, 0.04108760505914688, 0.07742585986852646, 
    -0.04636161029338837, 0.06589840352535248, 0.1060631200671196, 0.0413849800825119, 
    -0.008414228446781635, 0.029589818790555, -0.08502575010061264, 0.12076005339622498, 
    -0.03448090702295303, 0.03265566751360893, -0.037415292114019394, -0.06145671010017395, 
    0.17149843275547028, 0.0966511070728302, -0.01905285008251667, 0.07042571157217026, 
    -0.0667189210653305, -0.04331588000059128, -0.09697817265987396, 0.02167942188680172, 
    -0.14319781959056854, -0.07672872394323349, -0.017181534320116043, 0.048883240669965744, 
    0.08042725175619125, -0.0711820125579834, 0.025525791570544243, 0.1640680432319641, 
    0.07932639122009277, 0.17326174676418304, 0.023528272286057472, 0.052528586238622665, 
    0.05668769031763077, 0.04033545404672623, -0.07158572971820831, -0.025219948962330818, 
    -0.1120457798242569, 0.06467080861330032, -0.03561445698142052, -0.04243319109082222, 
    -0.11421578377485275, 0.13430169224739075, -0.02651282027363777, 0.08486393094062805, 
    0.04910343885421753, 0.031372785568237305, -0.008489816449582577, 0.004138967487961054, 
    0.06493031978607178, 0.09354047477245331, 0.10431460291147232, -0.0738627016544342, 
    0.016663342714309692, -0.09232482314109802, 0.051408492028713226, 0.14977632462978363, 
    -0.05317186191678047, -0.026693567633628845, -0.04245138540863991, 0.022616678848862648, 
    -0.15196332335472107, -0.07941148430109024, -0.12915828824043274, 0.11817729473114014, 
    -0.0012659248895943165, 0.0611608549952507, 0.11861971020698547, -0.06701331585645676, 
    -0.004937085788697004, -0.058826785534620285, -0.046757470816373825, 0.05300065875053406, 
    -0.0749511569738388, 0.044002510607242584, 0.026178697124123573, -0.020443998277187347, 
    -0.12991023063659668, -0.027409087866544724, -0.0328829325735569, 0.010432508774101734, 
    -0.10790299624204636, -0.12153945118188858, 0.05315593630075455, 0.03545274958014488, 
    -0.0349188856780529, -0.09781241416931152, -0.08163963258266449, -0.016394993290305138, 
    0.015373902395367622, -0.013966640457510948, -0.14763817191123962, 0.08389275521039963, 
    0.05042792856693268, -0.06262559443712234, -0.12576769292354584, 0.019883690401911736, 
    -0.04508855193853378, -0.09972033649682999, 0.01635022647678852, -0.05087033286690712, 
    0.06295738369226456, -0.011986308731138706, 0.031825825572013855, 0.12998540699481964, 
    0.1388990581035614, 0.13770204782485962, -0.0907498374581337, 0.04514127969741821, 
    -0.07034607231616974, 0.0033675585873425007, 0.04468878358602524, 0.018392764031887054, 
    -0.09614202380180359, -0.10913891345262527, 0.11364654451608658, 0.09555521607398987, 
    0.05459791421890259, -0.029131755232810974, 0.029551945626735687, 0.04660116508603096, 
    0.04036995396018028, 0.10560791939496994, 0.046961117535829544, 0.007288224995136261, 
    -0.006481604650616646, 0.009966983459889889, 0.0823705866932869, 0.012041456997394562, 
    0.1166994720697403, 0.0013764167670160532, -0.08563106507062912, -0.04971795901656151, 
    -0.051729366183280945, 0.03549068793654442, 0.04174259305000305, -0.09056981652975082, 
    0.04408881813287735, -0.15147387981414795, 0.06606388092041016, -0.03575197979807854, 
    -0.07317773997783661, -0.06508515775203705, -0.03054152801632881, -0.018837934359908104, 
    -0.05096353217959404, 0.06162586063146591, 0.021981945261359215, -0.17148755490779877, 
    0.0023611069191247225, -0.08035817742347717, 0.015498453751206398, 0.0006964735221117735, 
    -0.04838878661394119, 0.17551350593566895, -0.1427503079175949, -0.0038590377662330866, 
    -0.1587243229150772, -0.08668997883796692, 0.02029343508183956, 0.025820931419730186, 
    -0.16122928261756897, 0.020785050466656685, -0.042636822909116745, -0.02708689495921135, 
    0.027778612449765205, -0.00685254018753767, 0.08349283039569855, 0.09002253413200378, 
    -0.07140486687421799, 0.03441760689020157, -0.021388374269008636, 0.10853574424982071, 
    0.012222432531416416, 0.020198743790388107, 0.06932498514652252, 0.007721343077719212, 
    0.0765385851264, 0.12651877105236053, -0.1121596023440361, 0.045178111642599106, 
    -0.030096113681793213, -0.027875883504748344, -0.08021752536296844, 0.050782449543476105, 
    0.012448742054402828, 0.033887602388858795, 0.04822955280542374, 0.09286034852266312, 
    -0.017606688663363457, 0.05002397671341896, 0.011963838711380959, -0.0849597156047821, 
    0.06405353546142578, 0.0025780904106795788, 0.12355940043926239, -0.12866941094398499, 
    0.053860463201999664, 0.09067314863204956, -0.014597387053072453, -0.0026564733125269413, 
    -0.0048669492825865746, -0.04450152814388275, -0.017490310594439507, -0.11132761836051941, 
    0.06697380542755127, 0.030357366427779198, 0.011546105146408081, 0.0102451853454113, 
    -0.010158813558518887, 0.013660854659974575, 0.0800461620092392, -0.0027244980446994305, 
    0.0579618476331234, -0.05241238698363304, 0.07866252958774567, -0.10526323318481445, 
    -0.0986541137099266, -0.05421949177980423, 0.04717548191547394, 0.04802566394209862, 
    -0.045920420438051224, 0.003496473655104637, -0.12045006453990936, 0.054078564047813416, 
    -0.12164346873760223, -0.03201712295413017, -0.015934161841869354, 0.005485369823873043, 
    -0.02185167372226715, 0.09215566515922546, 0.10461903363466263, -0.022656269371509552, 
    -0.048277005553245544, 0.14765536785125732, 0.01660468429327011, 0.040853966027498245, 
    0.004944367799907923, -0.05140126869082451, -0.0043610637076199055, 0.06024584919214249, 
    0.03430555388331413, 0.14764343202114105, -0.037467069923877716, 0.014243228361010551, 
  
    };
    static const neuron_t neuron0 = {weights0, -0.08304399996995926};
    neurons[0]=neuron0;

    static const float weights1[] ={
    -0.09227989614009857, 0.015456286258995533, 0.08550723642110825, -0.05023583769798279, 
    0.06371618062257767, 0.03660036250948906, 0.04176408052444458, 0.011252767406404018, 
    0.159589484333992, 0.06460490077733994, 0.026692179962992668, 0.220413938164711, 
    -0.02396106906235218, 0.11668254435062408, 0.006692431401461363, 0.06458491086959839, 
    -0.00882859155535698, 0.10747349262237549, -0.08750476688146591, 0.0722464844584465, 
    -0.035048458725214005, 0.07838770747184753, 0.012853549793362617, -0.032322049140930176, 
    0.015357156284153461, 0.05755983293056488, 0.01943720132112503, 0.03245225176215172, 
    0.022224079817533493, 0.01616986282169819, -0.015625840052962303, -0.025656666606664658, 
    0.001514900242909789, 0.012787765823304653, 0.047164615243673325, 0.05182059854269028, 
    0.1375219225883484, -0.01639179140329361, -0.011816556565463543, 0.02961392141878605, 
    -0.0029503197874873877, -0.18055564165115356, 0.026590129360556602, -0.09989351034164429, 
    -0.09250397235155106, 0.03029337339103222, 0.0783688873052597, -0.02152307704091072, 
    -0.0011290234979242086, 0.04954420402646065, 0.0199369378387928, -0.009910520166158676, 
    -0.09703176468610764, -0.028708359226584435, -0.04211355373263359, -0.13806989789009094, 
    0.0823623538017273, -0.005686415359377861, 0.029219571501016617, 0.0175746139138937, 
    -0.0011421677190810442, 0.05592397600412369, 0.022038079798221588, 0.18187038600444794, 
    -0.03946206718683243, 0.0036820084787905216, -0.04659496992826462, 0.03662503883242607, 
    -0.01207624189555645, -0.10951143503189087, -0.002122745616361499, -0.0900336429476738, 
    0.013328365050256252, 0.08348260074853897, 0.057856783270835876, 0.06388484686613083, 
    0.038931772112846375, -0.04045478254556656, -0.0737447440624237, -0.019961856305599213, 
    0.014217151328921318, -0.015983860939741135, 0.024728847667574883, -0.09041338413953781, 
    -0.04347332566976547, 0.10641919821500778, -0.024181047454476357, -0.046312008053064346, 
    -0.03022194840013981, 0.093452088534832, -0.032985106110572815, 0.017154552042484283, 
    0.016119334846735, 0.04271882772445679, 0.08576399087905884, -0.10671977698802948, 
    -0.017485765740275383, 0.04780040308833122, 0.09438382834196091, 0.01840038038790226, 
    0.07058867812156677, 0.05009660869836807, 0.045826736837625504, 0.0698864534497261, 
    0.0408051423728466, -0.042083509266376495, 0.14491504430770874, 0.054733410477638245, 
    -0.014179741032421589, 0.05305830016732216, 0.0989520251750946, 0.07784761488437653, 
    0.05561686307191849, -0.0700983926653862, -0.006896122358739376, 0.03447262570261955, 
    0.007830034010112286, -0.07377329468727112, 0.05389589071273804, 0.05035722255706787, 
    -0.09210477024316788, -0.0026539843529462814, -0.05915738642215729, -0.10140791535377502, 
    -0.007683745119720697, -0.025534845888614655, 0.07892245054244995, -0.09161914885044098, 
    0.10661933571100235, -0.08535132557153702, 0.06350904703140259, -0.059939075261354446, 
    -0.1361090987920761, -0.05266881361603737, 0.09036453068256378, 0.08319655060768127, 
    0.02709374390542507, -0.0604928657412529, -9.864907042356208e-05, 0.08189307153224945, 
    0.09274768829345703, 0.045436080545186996, 0.16213558614253998, -0.050947390496730804, 
    -0.16464561223983765, 0.03046426549553871, -0.03836435079574585, -0.09869104623794556, 
    0.04101378843188286, 0.008866977877914906, -0.15596427023410797, -0.045912936329841614, 
    0.0756763368844986, 0.028971176594495773, -0.046103738248348236, 0.01916816085577011, 
    -0.151475191116333, -0.05610090494155884, 0.05543333292007446, 0.04693840071558952, 
    -0.072374127805233, -0.08986208587884903, 0.0022828634828329086, -0.01192310731858015, 
    -0.0016326240729540586, 0.12193146347999573, -0.05337574705481529, -0.002659483579918742, 
    -0.09868083149194717, 0.031868163496255875, 0.07983234524726868, -0.07216271758079529, 
    0.030161652714014053, -0.1701623648405075, 0.09220052510499954, 0.0990925133228302, 
    -0.04916219040751457, 0.09806280583143234, -0.05460836738348007, 0.14652980864048004, 
    -0.087443508207798, -0.07082782685756683, 0.00789587665349245, 0.07689256966114044, 
    0.14686359465122223, -0.0019223265117034316, -0.11812911182641983, 0.028680726885795593, 
    0.08724867552518845, -0.018937811255455017, 0.10622156411409378, -0.1183285191655159, 
    -0.01911860518157482, -0.005395602900534868, 0.030908146873116493, 0.026635559275746346, 
    0.022166920825839043, 0.04872826859354973, -0.0025272942148149014, 0.12522821128368378, 
    0.1092335656285286, 0.02496584691107273, -0.008029492571949959, 0.0522187277674675, 
    0.05128251761198044, 0.11939918994903564, -0.06415676325559616, 0.055787838995456696, 
    0.10242341458797455, 0.15352828800678253, -0.07784106582403183, 0.2681460678577423, 
    0.23037873208522797, -0.07471860945224762, 0.09931060671806335, -0.02653013914823532, 
    0.11574237793684006, -0.14039911329746246, 0.022900372743606567, 0.019525879994034767, 
    0.21363624930381775, -0.09865480661392212, 0.02398274466395378, -0.04064791649580002, 
    -0.013878192752599716, 0.058253783732652664, -0.13827364146709442, 0.07621725648641586, 
    -0.04628516733646393, -0.03331640362739563, 0.03270268812775612, 0.09080987423658371, 
    -0.014472348615527153, -0.03475829213857651, 0.11273334920406342, 0.14531490206718445, 
    0.07597380876541138, -0.021639548242092133, 0.19533663988113403, 0.010851087979972363, 
    0.014090519398450851, 0.0446600578725338, -0.07873403280973434, -0.02974461019039154, 
    0.02792312204837799, 0.051566775888204575, -0.010414469987154007, -0.11972612142562866, 
    0.12773345410823822, 0.05913985148072243, -0.1523543894290924, -0.08904343843460083, 
    -0.017101526260375977, -0.03486380726099014, -0.07520683109760284, -0.019688304513692856, 
    0.0894332155585289, 0.05545342341065407, 0.143667533993721, 0.12591803073883057, 
    -0.11226774007081985, -0.18189851939678192, 0.09581220149993896, -0.10014835745096207, 
    0.0005402651149779558, 0.17164747416973114, 0.052471939474344254, 0.10589462518692017, 
    -0.037914615124464035, 0.07893216609954834, -0.20707879960536957, 0.06849818676710129, 
    -0.14219115674495697, 0.014334416016936302, -0.03257090225815773, -0.0015398412942886353, 
    -0.043224845081567764, -0.1577020287513733, -0.1385306864976883, 0.01605141907930374, 
    -0.04429026320576668, 0.0011459249071776867, 0.03261447325348854, 0.0009041590965352952, 
    0.11199194192886353, 0.07843891531229019, 0.056416116654872894, -0.0817079097032547, 
    0.014963964000344276, 0.04230235889554024, -0.0744003877043724, -0.16944965720176697, 
    -0.05417485162615776, 0.2000218778848648, 0.002059522783383727, 0.08525446802377701, 
    -0.0165391992777586, 0.07685748487710953, -0.024837510660290718, 0.06708547472953796, 
    0.03868477791547775, 0.07250552624464035, -0.008359013125300407, -0.08555910736322403, 
    -0.12300291657447815, -0.1418350338935852, -0.08007648587226868, -0.04604160040616989, 
    0.04044676199555397, 0.1148931086063385, -0.11394419521093369, -0.136611670255661, 
    0.023677466437220573, -0.07661058753728867, -0.0066327061504125595, 0.021808955818414688, 
    0.12689484655857086, -0.0710720419883728, -0.027601424604654312, 0.000566894537769258, 
    -0.0034641146194189787, 0.015648426488041878, -0.1400173306465149, 0.08158650994300842, 
    0.018461329862475395, -0.13640841841697693, -0.11040706187486649, 0.07590950280427933, 
    -0.05045262351632118, 0.09917847812175751, -0.11313016712665558, 0.007354406639933586, 
    -0.012019140645861626, -0.04272998496890068, -0.11369606107473373, -0.08430161327123642, 
    -0.030985746532678604, 0.09481950849294662, -0.014484361745417118, -0.09064052253961563, 
    0.03698989748954773, -0.025379681959748268, -0.009208560921251774, -0.06621181964874268, 
    0.012753463350236416, 0.11671241372823715, -0.060006894171237946, -0.07478350400924683, 
    -0.1421658843755722, 0.11607179045677185, 0.013245770707726479, 0.032699596136808395, 
    -0.027492444962263107, 0.06302420049905777, -0.014404206536710262, -0.0767936110496521, 
    0.11706200987100601, -0.039611827582120895, 0.07571270316839218, 0.05275838077068329, 
    0.0852176621556282, 0.03192039579153061, 0.10724674165248871, -0.06143626943230629, 
    0.1324208378791809, 0.0549355074763298, -0.060521770268678665, 0.05027642846107483, 
    -0.0041153645142912865, 0.033146172761917114, 0.07607854157686234, 0.06474971771240234, 
    0.12577767670154572, -0.012453228235244751, -0.05188082903623581, -0.07199572026729584, 
    -0.12761381268501282, 0.03544497489929199, 0.07937541604042053, -0.028422461822628975, 
    -0.03590510040521622, -0.07505960017442703, -9.822763968259096e-05, -0.03237435221672058, 
    -0.04436279460787773, 0.04906672239303589, 0.06823872029781342, 0.1794668436050415, 
    0.0822702944278717, 0.02203504368662834, -0.09394373744726181, -0.01915658824145794, 
    0.1279338151216507, -0.13937734067440033, 0.051835522055625916, -0.07603287696838379, 
    0.06603018939495087, 0.12042529135942459, 0.03141027316451073, -0.12125639617443085, 
    -0.0543689951300621, 0.07806216180324554, -0.10263028740882874, -0.05309082940220833, 
    0.08324505388736725, 0.08422920852899551, -0.029178299009799957, -0.0543997623026371, 
    0.04008486121892929, 0.09331496059894562, -0.02399449236690998, -0.03491279482841492, 
    0.1017352044582367, 0.030536353588104248, 0.06825310736894608, 0.10835912078619003, 
    0.018486734479665756, -0.08109954744577408, 0.05979990214109421, -0.09600070863962173, 
    -0.03571585938334465, -0.026339896023273468, -0.04395201802253723, 0.06830181926488876, 
    0.06082414835691452, 0.11563130468130112, 0.06754333525896072, -0.051471319049596786, 
    -0.17075318098068237, -0.019507190212607384, -0.05025240033864975, -0.04184495657682419, 
    -0.03279157727956772, -0.0933675467967987, -0.01108905766159296, 0.12718497216701508, 
    -0.03416052833199501, -0.023478439077734947, -0.09890849888324738, -0.04673691466450691, 
    -0.06781424582004547, 0.18263667821884155, 0.0015845996094867587, -0.10135165601968765, 
    0.06308289617300034, 0.10579165071249008, -0.045471809804439545, 0.21195563673973083, 
    -0.10321734100580215, 0.06118578463792801, 0.10086105018854141, -0.07585211098194122, 
    0.1470370590686798, -0.04568623751401901, -0.06078994274139404, -0.07375074923038483, 
    0.12268070131540298, 0.07314462214708328, -0.02741185389459133, -0.07334588468074799, 
    0.004983985796570778, 0.08913467079401016, -0.056755680590867996, -0.01638711243867874, 
    0.079364113509655, 0.02322683483362198, -0.013234340585768223, 0.037965718656778336, 
    -0.09031631797552109, -0.05139344185590744, -0.11094566434621811, 0.08233533054590225, 
    -0.043449051678180695, 0.044433627277612686, -0.0975308045744896, -0.11996351182460785, 
    0.07500241696834564, 0.12834307551383972, -0.07499127089977264, 0.01636934094130993, 
    0.05273064598441124, 0.0949387326836586, -0.019246593117713928, -0.07302618771791458, 
    0.1656760722398758, -0.034360501915216446, -0.039284221827983856, 0.00500902533531189, 
    -0.02590101957321167, 0.10408179461956024, -0.03628448396921158, -0.003776008728891611, 
    -0.09393444657325745, -0.1691320389509201, -0.16877874732017517, -0.09737139195203781, 
    0.006337756756693125, 0.1438952386379242, -0.01933271437883377, -0.02092173509299755, 
    0.11585226655006409, 0.016053546220064163, -0.09260889887809753, -0.07634012401103973, 
    -0.06803104281425476, 0.009829365648329258, 0.004190095234662294, -0.005070582497864962, 
    -0.01217375136911869, -0.09276089817285538, 0.11381316930055618, 0.05624755471944809, 
  
    };
    static const neuron_t neuron1 = {weights1, -0.26595011353492737};
    neurons[1]=neuron1;

    static const float weights2[] ={
    0.01589019224047661, 0.002503864699974656, -0.08211452513933182, 0.09916671365499496, 
    0.044779740273952484, 0.02094140462577343, -0.054392799735069275, -0.06317420303821564, 
    -0.05532919988036156, 0.15561598539352417, -0.015085079707205296, -0.035993147641420364, 
    0.02080145850777626, 0.03220203518867493, 0.059913717210292816, 0.10283590853214264, 
    -0.013877646997570992, 0.01640348508954048, 0.04754554480314255, 0.023544669151306152, 
    -0.10739830136299133, 0.07580918818712234, -0.04493112117052078, -0.0032719355076551437, 
    0.019013606011867523, -0.011681679636240005, 0.024050388485193253, 0.011638112366199493, 
    -0.0037616300396621227, 0.012339204549789429, 0.008592747151851654, -0.09570910781621933, 
    0.1095537543296814, 0.0657709389925003, 0.015678394585847855, 0.050803426653146744, 
    0.030323181301355362, -0.08897224813699722, -0.08121979981660843, -0.08223037421703339, 
    -0.058264702558517456, 0.05558273568749428, 0.05708233267068863, 0.012553184293210506, 
    -0.08509407192468643, 0.010653499513864517, -0.06805086880922318, -0.01806468516588211, 
    0.008635002188384533, 0.1962938904762268, 0.14113233983516693, 0.04666969180107117, 
    0.10233976691961288, 0.07046284526586533, -0.0021870904602110386, 0.09197279065847397, 
    0.012727309949696064, -0.015374835580587387, 0.07755323499441147, -0.0272932518273592, 
    0.06491152197122574, -0.025988245382905006, 0.0154695650562644, 0.03442101553082466, 
    -0.03610026463866234, 0.01994388736784458, -0.057936470955610275, -0.004920697305351496, 
    0.009312441572546959, 0.12501747906208038, -0.010357228107750416, -0.0073243542574346066, 
    -0.01910170167684555, 0.09952377527952194, 0.022197416052222252, -0.051077887415885925, 
    0.1179823949933052, 0.0394914373755455, 0.05376642942428589, 0.13239607214927673, 
    -0.013769138604402542, 0.06476250290870667, -0.03353307768702507, 0.06501582264900208, 
    0.04071815311908722, -0.030752532184123993, 0.07391084730625153, 0.0547499917447567, 
    -0.006568173877894878, 0.023062609136104584, -0.03948720172047615, -0.13895036280155182, 
    0.021385468542575836, -0.03973950445652008, -0.009494123980402946, 0.007444807793945074, 
    0.039842840284109116, -0.04549034312367439, -0.08411062508821487, 0.06917273998260498, 
    0.08423508703708649, 0.07513276487588882, 0.009083368815481663, -0.019631950184702873, 
    0.0885421633720398, -0.08365924656391144, 0.04563368111848831, -0.02533024363219738, 
    0.005668759346008301, 0.029496759176254272, 0.056715019047260284, 0.05474739149212837, 
    -0.016504881903529167, -0.008922429755330086, -0.033655863255262375, 0.05209159478545189, 
    0.07587974518537521, -0.001973970327526331, 0.032175105065107346, 0.019293077290058136, 
    -0.05871133506298065, 0.02891824021935463, -0.007160973735153675, -0.018169302493333817, 
    -0.011420557275414467, -0.08125163614749908, 0.05616340413689613, 0.02413666434586048, 
    -0.07907699793577194, 0.023532215505838394, -0.0733916312456131, 0.0949145182967186, 
    -0.029204970225691795, -0.010710551403462887, 0.01140589639544487, 0.0199721809476614, 
    0.048335615545511246, -0.046950697898864746, -0.017581205815076828, -0.1179019883275032, 
    -0.00020075557404197752, 0.030803272500634193, -0.022801270708441734, -0.1169264167547226, 
    0.05508381873369217, 0.1009521484375, -0.08000728487968445, -0.008116410113871098, 
    -0.05662054568529129, 0.08527521044015884, -0.010132553055882454, 0.10257618129253387, 
    -0.11025848984718323, -0.09273785352706909, 0.08969944715499878, -0.019249064847826958, 
    -0.05265403911471367, -0.05674111098051071, -0.07938574999570847, 0.022118914872407913, 
    -0.03893566131591797, 0.1417800337076187, 0.04440126195549965, 0.011105990968644619, 
    -0.0776444524526596, -0.02545345015823841, -0.0594106949865818, 0.008442922495305538, 
    -0.043971579521894455, 0.026242276653647423, 0.04782221466302872, 0.03810695558786392, 
    0.04653003066778183, 0.06209718808531761, -0.04468680918216705, -0.034821659326553345, 
    0.07488549500703812, 0.013779450207948685, -0.054207902401685715, -0.07558510452508926, 
    0.08321794867515564, -0.033740028738975525, -0.08579573035240173, 0.02638012170791626, 
    -0.024898597970604897, -0.03153592720627785, 0.07044447213411331, 0.050557855516672134, 
    -0.049255967140197754, 0.01600251905620098, 0.11647676676511765, 0.023741817101836205, 
    -0.039292771369218826, 0.03526889532804489, -0.025217292830348015, 0.020166033878922462, 
    0.07168451696634293, 0.006691957823932171, 0.0015486733755096793, 0.16183966398239136, 
    0.007239836268126965, 0.08604635298252106, -0.1287374049425125, -0.06128789111971855, 
    -0.07111998647451401, -0.028204966336488724, -0.08768840879201889, -0.014373481273651123, 
    -0.1049688458442688, -0.003173593897372484, 0.1953451931476593, 0.1761389672756195, 
    -0.07327112555503845, -0.01901710219681263, -0.06932778656482697, -0.043885353952646255, 
    0.07283102720975876, 0.022770890966057777, 0.04378094524145126, -0.044409241527318954, 
    -0.007991439662873745, 0.012741664424538612, -0.03944391757249832, 0.0655938982963562, 
    -0.030161608010530472, -0.01313839852809906, 0.025822144001722336, -0.12679246068000793, 
    0.00974993035197258, -0.05538097396492958, 0.023660393431782722, -0.15957845747470856, 
    -0.11300063133239746, 0.07291711866855621, -0.00524802366271615, 0.06360054761171341, 
    0.03490092232823372, -0.037181176245212555, -0.06171921268105507, -0.022759445011615753, 
    0.15260635316371918, -0.12391109764575958, -0.005943511612713337, -0.08109122514724731, 
    -0.027016431093215942, -0.11393618583679199, -0.07750706374645233, 0.0016150204464793205, 
    -0.0742727518081665, -0.02771795354783535, 0.003726700320839882, 0.006131794303655624, 
    0.029281755909323692, -0.05412473529577255, -0.04189702868461609, -0.14297732710838318, 
    -0.0005690744146704674, 0.013756250031292439, -0.0775861069560051, 0.05318433791399002, 
    0.03817341476678848, -0.06331988424062729, -0.07027606666088104, -0.03976757451891899, 
    -0.019172407686710358, 0.008965625427663326, 0.02648698352277279, -0.07102981954813004, 
    0.00715736486017704, 0.07223383337259293, -0.012655336409807205, 0.022754935547709465, 
    0.013710693456232548, -0.04777190461754799, -0.010426162742078304, 0.032370675355196, 
    -0.13399435579776764, 0.02998155727982521, 0.05004572123289108, 0.04428838565945625, 
    0.023556765168905258, 0.029310960322618484, 0.03410225361585617, 0.0739082470536232, 
    -0.09669631719589233, -0.07129750400781631, 0.040312282741069794, 0.002990254433825612, 
    0.06589280068874359, 0.07868753373622894, -0.030933240428566933, 0.06838531792163849, 
    -0.028122397139668465, 0.027421219274401665, 0.058348461985588074, 0.013496693223714828, 
    0.01811027340590954, 0.06175311654806137, 0.0742608904838562, 0.030764026567339897, 
    0.005545333027839661, -0.07805897295475006, 0.028747892007231712, -0.028175951912999153, 
    -0.026431437581777573, -0.05741460621356964, 0.17972712218761444, 0.031212618574500084, 
    0.08885353058576584, 0.04576299712061882, 0.003813001560047269, -0.05326278880238533, 
    0.10261739790439606, 0.07433756440877914, 0.08147045224905014, 0.012335076928138733, 
    0.08868318051099777, 0.009196368977427483, -0.03334944695234299, -0.01459102425724268, 
    0.1044488400220871, 0.09664438664913177, 0.04388389736413956, -0.04159065708518028, 
    0.00920010730624199, 0.08472701907157898, -0.02569466456770897, -0.11024928092956543, 
    -0.06187629699707031, 0.11008047312498093, 0.04366251453757286, 0.0024053778033703566, 
    0.013972661457955837, -0.029685191810131073, 0.0860549807548523, -0.06696898490190506, 
    0.08622317761182785, -0.06901397556066513, 0.039926040917634964, 0.015536724589765072, 
    -0.018324371427297592, 0.01988849975168705, 0.012251939624547958, -0.011498110368847847, 
    -0.015197467990219593, -0.0649639144539833, -0.008058082312345505, -0.08891218155622482, 
    0.002098443917930126, -0.014814557507634163, -0.035221658647060394, 0.07864708453416824, 
    -0.06580787897109985, -0.054240092635154724, 0.032196540385484695, -0.02179153822362423, 
    0.03802603483200073, -0.19452905654907227, -0.008166064508259296, -0.023596378043293953, 
    0.034543730318546295, -0.006348439026623964, -0.0649116262793541, -0.05928235873579979, 
    0.0491308867931366, -0.04360997676849365, -0.001061632065102458, -0.11018302291631699, 
    -0.03320955112576485, -0.001259722514078021, -0.007142156828194857, -0.06814082711935043, 
    0.08013696223497391, 0.009861140511929989, -0.04604283720254898, 0.06992922723293304, 
    0.04510952904820442, 0.07149911671876907, 0.08323916047811508, -0.0436871200799942, 
    -0.13248690962791443, 0.013907483778893948, 0.11318732053041458, -0.05561695620417595, 
    0.03358374908566475, 0.017649736255407333, -0.08777807652950287, -0.016368359327316284, 
    -0.09544464945793152, -0.047968246042728424, -0.021632743999361992, 0.024997666478157043, 
    -0.07807745784521103, 0.09961080551147461, 0.0595608688890934, 0.02618652954697609, 
    0.06529977917671204, 0.011279009282588959, -0.0046646627597510815, -0.030417829751968384, 
    -0.056100908666849136, 0.12414451688528061, -0.04028075188398361, 0.027689630165696144, 
    -0.04678432270884514, -0.10622237622737885, 0.08426803350448608, 0.0509452149271965, 
    -0.09705301374197006, 0.09112978726625443, -0.03752351924777031, -0.00017802268848754466, 
    0.027326000854372978, -0.021845154464244843, -0.017128532752394676, -0.11227653175592422, 
    -0.01450443547219038, 0.014778267592191696, -0.0663311630487442, 0.023877430707216263, 
    -0.08059083670377731, -0.0460003986954689, -0.017480820417404175, -0.03737358748912811, 
    -0.09796846657991409, 0.028370331972837448, 0.028662655502557755, 0.08662644028663635, 
    -0.06991694867610931, -0.06857487559318542, -0.00807566661387682, -0.01290226075798273, 
    -0.05710677430033684, 0.051808495074510574, -0.055738575756549835, -0.04333735629916191, 
    0.00039093082887120545, 0.08999205380678177, 0.08323702216148376, 0.055037956684827805, 
    -0.06456191837787628, -0.05150511860847473, 0.03171467408537865, 0.029461655765771866, 
    0.10527176409959793, -0.001084606978110969, -0.036024898290634155, 0.01156314741820097, 
    0.03206132724881172, 0.05351291969418526, -0.06411436200141907, 0.021271420642733574, 
    -0.04332137480378151, 0.03069295920431614, -0.11858952045440674, 0.10224819928407669, 
    -0.04697752371430397, 0.04329824820160866, -0.04982630908489227, -0.05356587469577789, 
    -0.008081461302936077, 0.030767982825636864, 0.0689249113202095, -0.12366236746311188, 
    0.024868309497833252, 0.1676790416240692, 0.0010453752474859357, -0.04603419452905655, 
    0.16641829907894135, 0.09245895594358444, 0.0018059442518278956, -0.023025071248412132, 
    -0.05150327831506729, 0.0248073972761631, -0.015575332567095757, -0.018421759828925133, 
    0.04553339257836342, -0.04095909371972084, 0.05276769772171974, 0.13163019716739655, 
    -0.011513518169522285, -0.09339817613363266, -0.08496423065662384, 0.05043042078614235, 
    0.027095865458250046, -0.06736310571432114, 0.0475362092256546, 0.02034379541873932, 
    0.0017469787271693349, 0.11860696971416473, -0.0620911680161953, 0.08731396496295929, 
    0.011286204680800438, -0.025024306029081345, 0.07542484998703003, -0.04039003700017929, 
    0.041579823940992355, -0.04925670474767685, 0.0735635757446289, 0.07013106346130371, 
    -0.0069900997914373875, -0.027542609721422195, -0.032982952892780304, -0.024629540741443634, 
    0.09725972265005112, -0.12256737798452377, 0.06605557352304459, -0.0342080220580101, 
    0.10954764485359192, 0.05444161966443062, 0.05885826796293259, -0.03258107975125313, 
  
    };
    static const neuron_t neuron2 = {weights2, 0.004709299188107252};
    neurons[2]=neuron2;

    static const float weights3[] ={
    -0.003305676858872175, -0.11225482076406479, -0.036598481237888336, 0.060578811913728714, 
    -0.07403171807527542, 0.0072199017740786076, -0.06254488229751587, 0.11540688574314117, 
    -0.10878893733024597, -0.02845325879752636, 0.06261259317398071, -0.03798803687095642, 
    0.07271526753902435, 0.08036654442548752, 0.00046263140393421054, -0.028856556862592697, 
    -0.020493872463703156, -0.005965394899249077, 0.06679980456829071, -0.06725707650184631, 
    -0.06489483267068863, -0.100699283182621, -0.061669331043958664, 0.038206782191991806, 
    -0.03366483375430107, -0.0017137588001787663, -0.022029832005500793, -0.03789706528186798, 
    0.05565003305673599, -0.015565061941742897, 0.0493326336145401, -0.11762446165084839, 
    -0.004940861836075783, -0.02327169105410576, -0.07769689708948135, -0.13598881661891937, 
    0.04910961166024208, -0.05670342221856117, 0.03631305322051048, 0.04029380902647972, 
    0.09331567585468292, -0.0033285890240222216, 0.04548218846321106, 0.09947682172060013, 
    0.028026077896356583, 0.042245566844940186, -0.020507842302322388, 0.08804414421319962, 
    0.009277960285544395, -0.016858212649822235, -0.02247508242726326, -0.05934979021549225, 
    -0.016146987676620483, -0.042226552963256836, 0.05226868763566017, 0.09574788063764572, 
    0.08671610802412033, -0.04809656739234924, 0.039392080157995224, 0.08750941604375839, 
    -0.03865962475538254, -0.09316504746675491, 0.06632231920957565, -0.018641456961631775, 
    0.018057502806186676, -0.03355669230222702, -0.06111161783337593, -0.007285596802830696, 
    -0.021600408479571342, 0.052647121250629425, 0.016200244426727295, 0.11473962664604187, 
    0.10595837980508804, -0.07879600673913956, 0.07670345902442932, 0.024389643222093582, 
    -0.06264643371105194, -0.08869540691375732, 0.052028268575668335, 0.08822982758283615, 
    -0.00994125660508871, 0.12150344252586365, -0.09678354859352112, 0.006155568175017834, 
    0.04332248121500015, -0.07708846777677536, -0.05072728917002678, -0.031219687312841415, 
    -0.01677146926522255, -0.09838535636663437, -0.08738907426595688, 0.09487160295248032, 
    0.0255486648529768, 0.03468501195311546, -0.015601668506860733, 0.03260599076747894, 
    0.007852081209421158, -0.01191446091979742, -0.12930046021938324, -0.0690804198384285, 
    -0.019696541130542755, 0.029469119384884834, -0.0025649305898696184, 0.08334258943796158, 
    0.010882368311285973, -0.03032935969531536, -0.07140389084815979, -0.06895777583122253, 
    0.02017434500157833, -0.031043533235788345, -0.04628394544124603, -0.008717414923012257, 
    -0.02237289771437645, -0.013449279591441154, 0.01452572364360094, 0.02562473900616169, 
    -0.05134394392371178, 0.034453023225069046, 0.004091734532266855, -0.010340066626667976, 
    0.014503369107842445, -0.019372064620256424, -0.02411954291164875, -0.025348622351884842, 
    0.002854045480489731, 0.010307575576007366, 0.06415478885173798, -0.04126889258623123, 
    -0.013548798859119415, 0.08165983855724335, -0.02139134146273136, 0.00611121766269207, 
    -0.01647566631436348, -0.1082644835114479, -0.027093730866909027, -0.14159804582595825, 
    0.030039427801966667, 0.11038865149021149, -0.05227426066994667, -0.06238482892513275, 
    0.0072906664572656155, -0.08695053309202194, -0.005875150673091412, -0.05077498406171799, 
    0.02844107709825039, 0.08977507054805756, -0.026392197236418724, -0.07859191298484802, 
    -0.0072006890550255775, -0.09347599744796753, 0.0557221919298172, 0.0718982145190239, 
    -0.05059846490621567, -0.010889732278883457, -0.03793102130293846, 0.017667628824710846, 
    0.057472847402095795, -0.04294126480817795, 0.07163447886705399, -0.043413031846284866, 
    0.0010068854317069054, -0.005986569449305534, -0.053163111209869385, 0.06471344083547592, 
    0.008910362608730793, 0.007429605815559626, -0.054971951991319656, -0.02034832537174225, 
    -0.04384408891201019, 0.02864532172679901, 0.02615123800933361, 0.010498752817511559, 
    -0.00786967296153307, 0.04916505515575409, 0.02062634564936161, -0.02706761658191681, 
    -0.08169806003570557, 0.07479643076658249, -0.0031339051201939583, -0.05807499587535858, 
    0.04080963507294655, 0.037148136645555496, 0.0988742858171463, 0.029167424887418747, 
    -0.04189560189843178, -0.04623229056596756, -0.041821837425231934, -0.015057732351124287, 
    -0.015965968370437622, -0.01612662710249424, -0.04177075996994972, 0.0051123155280947685, 
    0.03155995160341263, -0.002369111170992255, -0.11641022562980652, -0.017512895166873932, 
    -0.06395628303289413, 0.05484475567936897, -0.009976113215088844, -0.04147130995988846, 
    -0.01918385736644268, 0.00023878319188952446, -0.005520286504179239, 0.11335275322198868, 
    -0.004368099384009838, 0.002991270273923874, 0.05970248207449913, -0.07549698650836945, 
    -0.05666197091341019, 0.015775209292769432, 0.007721475325524807, -0.015914015471935272, 
    0.04444107413291931, -0.003629208542406559, 0.02880224399268627, -0.06409530341625214, 
    -0.0038766132201999426, 0.008755318820476532, -0.07576097548007965, -0.040832024067640305, 
    0.019449647516012192, 0.09141607582569122, -0.009200654923915863, 0.05100904032588005, 
    -0.02730751782655716, -0.01041590329259634, 0.18931268155574799, 0.07819049805402756, 
    -0.014595509506762028, -0.057855162769556046, -0.015066584572196007, -0.05056244507431984, 
    -0.07147455960512161, 0.021562648937106133, 0.028869889676570892, 0.02370217628777027, 
    -0.048560626804828644, 0.11753582209348679, -0.09236964583396912, -0.06279352307319641, 
    -0.11891527473926544, -0.054813843220472336, -0.03917459771037102, 0.0624145045876503, 
    -0.12860222160816193, 0.09033669531345367, -0.01776934787631035, 0.016967549920082092, 
    -0.01097804307937622, 0.0365169532597065, 0.032887525856494904, -0.06602581590414047, 
    -0.007246077060699463, -0.057772353291511536, 0.025353718549013138, -0.014735148288309574, 
    0.0013515674509108067, -0.0552433580160141, 0.017661768943071365, -0.00482544070109725, 
    0.009188725613057613, 0.0550931952893734, 0.0425901859998703, 0.11640851199626923, 
    -0.02316833846271038, 0.009447924792766571, 0.029605146497488022, -0.010388270951807499, 
    -0.004578901920467615, -0.0399925522506237, -0.11428231745958328, 0.015367209911346436, 
    -0.0014812361914664507, 0.0009262637468054891, 0.004720412194728851, 0.0048871589824557304, 
    0.02462402544915676, -0.06537991017103195, 0.12296829372644424, -0.05609126761555672, 
    -0.06315869837999344, -0.002023342764005065, 0.11492279917001724, 0.0027511767111718655, 
    0.03566336631774902, -0.10084547102451324, 0.047281429171562195, -0.04414728283882141, 
    -0.11267513781785965, -0.042715057730674744, -0.15821503102779388, -0.0019385057967156172, 
    0.11692242324352264, -0.028875721618533134, -0.045594725757837296, -0.057574160397052765, 
    -0.016741806641221046, -0.08472908288240433, 0.10085572302341461, -0.11011165380477905, 
    -0.0019839671440422535, -0.021227896213531494, -0.07239967584609985, -0.06604047119617462, 
    0.03992191329598427, 0.043499305844306946, 0.1217329278588295, 0.05649029091000557, 
    0.03364522382616997, -0.07991016656160355, -0.0013531427830457687, -0.1350669413805008, 
    0.025623580440878868, -0.005847846157848835, -0.07197223603725433, 0.0037058296147733927, 
    0.014796431176364422, -0.039982691407203674, -0.03605346754193306, 0.01837223395705223, 
    -0.00033842999255284667, 0.02197766862809658, 0.10246308892965317, 0.025948476046323776, 
    -0.030774375423789024, 0.09492750465869904, -0.03938605263829231, 0.01925399713218212, 
    -0.030224543064832687, -0.07915376126766205, -0.019623342901468277, -0.046959150582551956, 
    -0.02417021431028843, 0.05719808116555214, 0.020692642778158188, 0.0880572646856308, 
    0.045975543558597565, -0.04877966269850731, 0.061644669622182846, -0.05764184147119522, 
    -0.0652429610490799, 0.07689842581748962, 0.031006893143057823, 0.021489636972546577, 
    0.0362858772277832, -0.13586629927158356, -0.04485023021697998, 0.01084392610937357, 
    0.09435270726680756, -0.03310755640268326, -0.03767063841223717, -0.004529818892478943, 
    -0.04075800999999046, 0.05068669095635414, -0.1880842000246048, -0.0037950468249619007, 
    0.016617247834801674, 0.031946830451488495, -0.016025640070438385, 0.009792277589440346, 
    -0.01506041456013918, 0.05321439355611801, 0.09640128910541534, -0.0635000467300415, 
    0.07706701755523682, -0.04961780831217766, -0.042831435799598694, -0.05734468623995781, 
    0.03978138044476509, 0.09786821156740189, 0.05604151636362076, -0.0457371324300766, 
    -0.09221766144037247, 0.024698281660676003, 0.02166467346251011, -0.04965413361787796, 
    -0.03253374621272087, 0.04041625186800957, -0.05174768716096878, 0.008423786610364914, 
    -0.08469419181346893, 0.04150962084531784, -0.05093960091471672, 0.05613842234015465, 
    0.002104491926729679, 0.055468011647462845, -0.00257085170596838, -0.10483089834451675, 
    -0.016118036583065987, -0.04276260733604431, -0.06122748926281929, 0.11044757813215256, 
    -0.022422296926379204, -0.0057369861751794815, 0.01885085180401802, -0.019505202770233154, 
    -0.05274921655654907, -0.03869166225194931, 0.03312741965055466, -0.0056586796417832375, 
    -0.038010142743587494, 0.0026298954617232084, 0.024466026574373245, 0.01916377618908882, 
    -0.06606273353099823, -0.05386985093355179, 0.04293425753712654, 0.06273526698350906, 
    -0.030087847262620926, 0.018779372796416283, -0.06676323711872101, -0.03313065692782402, 
    0.0433223620057106, -0.04396712779998779, -0.040002454072237015, 0.02836710959672928, 
    0.02408467046916485, -0.021670369431376457, -0.013337068259716034, 0.1065991222858429, 
    -0.016261281445622444, -0.08093592524528503, 0.022022560238838196, -0.07359690219163895, 
    0.03700198978185654, -0.00201670010574162, -0.03382350504398346, -0.06192031502723694, 
    0.05788186192512512, 0.05339078605175018, 0.013497737236320972, -0.09686171263456345, 
    0.024777358397841454, 0.11871447414159775, -0.13121755421161652, -0.08883078396320343, 
    0.0673268660902977, 0.10733155906200409, 0.02577969618141651, -0.05878765136003494, 
    0.001315389876253903, 0.03714984282851219, 0.003756539896130562, 0.04610588774085045, 
    0.06205661594867706, -0.013019806705415249, -0.033940210938453674, -0.04385889321565628, 
    0.03768083080649376, 0.07764457911252975, -0.03657069802284241, 0.00253546005114913, 
    0.01402497198432684, -0.0032078055664896965, -0.024183519184589386, -0.0038010042626410723, 
    -0.04173174872994423, -0.07634717971086502, -0.041782163083553314, -0.004472118336707354, 
    -0.06223955377936363, 0.008245336823165417, -0.05294548720121384, 0.011775396764278412, 
    0.03162002190947533, -0.036343108862638474, -0.06726258248090744, -0.05489090457558632, 
    0.0830436646938324, 0.08957980573177338, -0.04350787773728371, 0.05294812098145485, 
    0.13432404398918152, -0.04420844092965126, -0.03668621927499771, 0.059198006987571716, 
    0.018988195806741714, -0.039311546832323074, 0.11282464116811752, 0.01765717938542366, 
    0.0012444722233340144, 0.03675800561904907, -0.04800596088171005, -0.009137097746133804, 
    0.04426441714167595, -0.057017214596271515, 0.014791500754654408, -0.12076956778764725, 
    0.00699799507856369, -0.05304134264588356, 0.03044707328081131, 0.09923426061868668, 
    0.0601990669965744, 0.054939884692430496, -0.013144116848707199, -0.10535474866628647, 
    0.013604238629341125, 0.015648718923330307, -0.016751261427998543, 0.11557091027498245, 
    -0.004152535926550627, -0.018926208838820457, 0.05906696990132332, 0.04125835373997688, 
    -0.07188177853822708, -0.02254371903836727, -0.003982506692409515, 0.03622831404209137, 
    0.0227184034883976, -0.0032298360019922256, -0.05761343240737915, -0.04481041431427002, 
  
    };
    static const neuron_t neuron3 = {weights3, 0.2800866961479187};
    neurons[3]=neuron3;

    static const float weights4[] ={
    0.017434218898415565, -0.05648639798164368, -0.0434228777885437, -0.07252722978591919, 
    0.03361699357628822, -0.09844181686639786, -0.04719642922282219, -0.042745448648929596, 
    0.005149203818291426, 0.028629854321479797, 0.10954034328460693, 0.13596878945827484, 
    -0.041014596819877625, -0.017884451895952225, -0.04545006901025772, -0.05816761776804924, 
    -0.06340484321117401, 0.021236684173345566, -0.04682927578687668, -0.03215775638818741, 
    0.023327630013227463, 0.10739070922136307, 0.05287974700331688, -0.03094159997999668, 
    -0.010389364324510098, 0.07530231028795242, -0.04689553380012512, 0.05185224860906601, 
    -0.07751677185297012, 0.03851886838674545, 0.09503577649593353, -0.0028595689218491316, 
    0.004518724512308836, 0.12152143567800522, 0.010175267234444618, -0.03127884492278099, 
    -0.07297181338071823, 0.013659127987921238, -0.07085863500833511, -0.06974048912525177, 
    -0.023303592577576637, -0.03769670054316521, 0.04291762039065361, 0.06638514250516891, 
    -0.05125178024172783, 0.024250874295830727, -0.10230712592601776, 0.04012919217348099, 
    0.06914444267749786, -0.03489775210618973, 0.03277960047125816, 0.001520925317890942, 
    -0.07149992138147354, -0.04789622128009796, 0.04932418093085289, 0.11634604632854462, 
    -0.11938545107841492, 0.035079970955848694, -0.03610578924417496, 0.08853139728307724, 
    -0.009390480816364288, -0.005975742824375629, 0.05722206458449364, 0.008882405236363411, 
    -0.002194548025727272, -0.05693366006016731, 0.09880201518535614, -0.002100670011714101, 
    -0.00407291017472744, 0.0443718321621418, -0.06031070649623871, -0.05867389962077141, 
    0.021465592086315155, 0.013865673914551735, -0.14176391065120697, 0.10533206909894943, 
    0.048611272126436234, 0.024006063118577003, 0.1057400107383728, 0.023632338270545006, 
    -0.07742926478385925, 0.05151151493191719, 0.06287141889333725, -0.057412177324295044, 
    0.014209306798875332, 0.011846042238175869, -0.029402337968349457, -0.04785048961639404, 
    0.03410257399082184, 0.11701234430074692, 0.06550590693950653, 0.057864103466272354, 
    -0.06126517057418823, 0.02380092814564705, -0.005711894948035479, -0.061435725539922714, 
    -0.044733062386512756, 0.05108126997947693, -0.03726444020867348, -0.06154090166091919, 
    -0.02048359252512455, 0.033270303159952164, -0.020867502316832542, -0.08344732969999313, 
    0.08148355036973953, -0.06412076950073242, 0.028752397745847702, -0.013810801319777966, 
    0.02494736760854721, -0.04072985798120499, -0.034387897700071335, -0.15307088196277618, 
    -0.02956440858542919, 0.033190805464982986, 0.011561035178601742, -0.010261652991175652, 
    0.02006426267325878, 0.16649577021598816, -0.06824059784412384, 0.050260212272405624, 
    0.05225655436515808, 0.09864235669374466, -0.019338279962539673, -0.015595857985317707, 
    0.027906298637390137, 0.022888945415616035, -0.022268544882535934, 0.06739439070224762, 
    -0.05658452957868576, 0.0930645614862442, 0.020589541643857956, -0.07562942057847977, 
    -0.03077647276222706, -0.007755391299724579, -0.09249647706747055, -0.03622307628393173, 
    -0.07522455602884293, -0.017458055168390274, -0.009372679516673088, -0.09459461271762848, 
    -0.07395882159471512, -0.09053133428096771, -0.04624729976058006, 0.019225753843784332, 
    0.12968309223651886, 0.12061592191457748, 0.004133167676627636, 0.09784514456987381, 
    0.03867564722895622, 0.029563333839178085, 0.01302677858620882, 0.13634838163852692, 
    -0.03989332169294357, -0.10401938855648041, 0.06894408911466599, 0.04439018666744232, 
    -0.07136929035186768, 0.14755108952522278, -0.08074924349784851, 0.021928904578089714, 
    0.0180564783513546, -0.024084264412522316, -0.09259801357984543, -0.0064735752530395985, 
    -0.01291777566075325, -0.11498978734016418, 0.008991830050945282, -0.05028238147497177, 
    -0.0867953896522522, 0.018146200105547905, 0.022788506001234055, 0.030241120606660843, 
    0.016591764986515045, 0.04578763619065285, 0.004495740402489901, 0.09385630488395691, 
    -0.07594843208789825, -0.04768471419811249, -0.008734582923352718, -0.05326022952795029, 
    -0.01875418610870838, 0.018200702965259552, -0.012746619060635567, -0.05359669029712677, 
    0.032220952212810516, 0.1385960578918457, 0.17586791515350342, -0.029868274927139282, 
    -0.0414600633084774, 0.015192369930446148, -0.04008051007986069, 0.0958731546998024, 
    -0.03279979154467583, 0.03472603112459183, -0.07377221435308456, -0.04918312281370163, 
    -0.002679392695426941, -0.02651866525411606, -0.07014413923025131, -0.06334009766578674, 
    -0.011620559729635715, -0.041793398559093475, -0.01867641508579254, -0.17191267013549805, 
    -0.018581802025437355, -0.03484351560473442, 0.08403732627630234, -0.004558964166790247, 
    -0.04410208761692047, 0.09643830358982086, 0.13755345344543457, 0.036793000996112823, 
    0.042356159538030624, -0.0885162279009819, 0.07555712759494781, 0.009978462010622025, 
    -0.017222417518496513, 0.07165732234716415, -0.08732494711875916, 0.022836245596408844, 
    -0.004392670467495918, -0.014318941161036491, 0.09541875123977661, 0.0040954044088721275, 
    -0.022061895579099655, 0.03145040571689606, -0.026411762461066246, 0.03695204481482506, 
    -0.05154309794306755, -0.129629448056221, 0.07150670140981674, -0.14906670153141022, 
    -0.046420030295848846, -0.11796766519546509, -0.09363023191690445, -0.04600292816758156, 
    0.003051092615351081, 0.029348131269216537, 0.05155596509575844, 0.16282542049884796, 
    0.008429880253970623, -0.016483673825860023, 0.007033411879092455, 0.06475883722305298, 
    0.018721913918852806, -0.09537764638662338, 0.05109633877873421, -0.005980716086924076, 
    -0.1366243213415146, -0.08761773258447647, -0.039743199944496155, 0.05160249024629593, 
    -0.01144913025200367, -0.057199627161026, 0.02339247427880764, -0.0816768929362297, 
    0.05680873617529869, 0.08324133604764938, -0.18298126757144928, 0.003447422757744789, 
    0.07338759303092957, -0.018612533807754517, -0.037729278206825256, 0.07903836667537689, 
    -0.09162139147520065, -0.04841252788901329, -0.02916504628956318, 0.008541418239474297, 
    0.04680061712861061, -0.011960011906921864, 0.07523868978023529, 0.0057346937246620655, 
    0.09727848321199417, -0.07661482691764832, -0.0012711964081972837, 0.005331188905984163, 
    -0.034504711627960205, 0.06275814771652222, 0.04137752950191498, 0.050602395087480545, 
    -0.1561478078365326, -0.012208234518766403, -0.128505140542984, 0.03932444751262665, 
    0.0735003650188446, -0.017025494948029518, 0.11055479198694229, 0.04587186872959137, 
    0.06993672251701355, -0.02782249264419079, 0.0837337076663971, -0.051921892911195755, 
    -0.05338641628623009, -0.08165769279003143, 0.08708296716213226, -0.06237678602337837, 
    -0.007852557115256786, 0.007866187021136284, 0.01765119656920433, 0.1176481768488884, 
    0.018075767904520035, 0.046215273439884186, 0.05495651438832283, 0.0592474527657032, 
    -0.04193644970655441, -0.01767101138830185, 0.01522397343069315, 0.06064427271485329, 
    0.0031398527789860964, -0.02004547230899334, 0.15747182071208954, 0.027301141992211342, 
    -0.06006558984518051, 0.05565975606441498, 0.028154633939266205, -0.021700894460082054, 
    -0.013239272870123386, 0.00934345182031393, -0.04912290349602699, 0.009854082949459553, 
    0.05268760398030281, -0.041965141892433167, 0.030502786859869957, -0.024966835975646973, 
    -0.010793343186378479, -0.09345081448554993, -0.02551421895623207, -0.051228467375040054, 
    0.05448944866657257, 0.11444412916898727, 0.012010928243398666, 0.07806382328271866, 
    0.037686917930841446, -0.03192724287509918, 0.12384229153394699, -0.020830366760492325, 
    0.05719354376196861, 0.06385661661624908, -0.018399950116872787, 0.05129845812916756, 
    0.057234637439250946, -0.016773106530308723, -0.011947834864258766, -0.022516783326864243, 
    -0.016434235498309135, -0.21382617950439453, -0.05533382669091225, -0.06279309839010239, 
    0.09930291026830673, -0.09664762020111084, -0.00950335618108511, -0.08546928316354752, 
    -0.015504524111747742, -0.0676274448633194, -0.002121745143085718, 0.03170335292816162, 
    0.09206297248601913, -0.017196878790855408, -0.11714127659797668, -0.018751531839370728, 
    0.088257797062397, 0.046735286712646484, 0.09871987998485565, 0.0027545420452952385, 
    0.009965197183191776, 0.09548608213663101, -0.01618233695626259, -0.02905815839767456, 
    -0.040544524788856506, -0.031094146892428398, -0.06856153160333633, 0.0247094314545393, 
    0.012110133655369282, 0.04182400554418564, -0.0921807512640953, 0.06742706894874573, 
    0.04653989151120186, 0.04821169376373291, -0.052290305495262146, 0.07831558585166931, 
    0.026596644893288612, 0.049338508397340775, -0.01778753474354744, 0.057930782437324524, 
    -0.030156409367918968, -0.026719387620687485, 0.10116636753082275, -0.11512494832277298, 
    0.06602278351783752, -0.07516942173242569, -0.03856270760297775, -0.03715949133038521, 
    -0.0712890475988388, 0.033998094499111176, 0.0473131388425827, 0.026441948488354683, 
    0.020145311951637268, 0.08758105337619781, -0.03311425447463989, 0.060770198702812195, 
    0.06700154393911362, 0.006014946382492781, -0.02268064022064209, 0.06520665436983109, 
    -0.01649688184261322, 0.0277695395052433, 0.03863289952278137, -0.10693162679672241, 
    -0.11228437721729279, 0.005301464349031448, 0.060617025941610336, -0.04882222041487694, 
    -0.09034600108861923, -0.1509772539138794, 0.09291837364435196, 0.0072900112718343735, 
    -0.004946152679622173, 0.11921866983175278, -0.039079148322343826, 0.1402861475944519, 
    -0.03127492591738701, -0.040620192885398865, 0.016291556879878044, -0.09163649380207062, 
    -0.08468260616064072, -0.031618811190128326, -0.07878613471984863, 0.023855945095419884, 
    -0.015670403838157654, -0.040235619992017746, 0.06858944892883301, -0.0808589905500412, 
    -0.04718652367591858, 0.05658801645040512, 0.01894701085984707, -0.010198459029197693, 
    0.009333024732768536, 0.0021396914962679148, 0.03610900789499283, 0.08234893530607224, 
    0.03714732080698013, -0.007068205159157515, 0.05162801966071129, -0.0006447809864766896, 
    -0.09370841830968857, -0.06161877512931824, -0.012278473004698753, -0.01826128363609314, 
    0.012936880812048912, 0.08471260964870453, -0.003953786566853523, 0.0014535632217302918, 
    -0.05210453271865845, 0.09628152847290039, -0.06484933197498322, -0.0841275230050087, 
    0.0126457205042243, -0.09599240869283676, 0.13304907083511353, -0.11201254278421402, 
    0.008002613671123981, 0.04040682315826416, -0.09136143326759338, 0.058736611157655716, 
    0.05344509705901146, 0.1380334198474884, -0.10344915837049484, -0.020947279408574104, 
    0.058667104691267014, -0.06597565114498138, 0.0009139972971752286, -0.1388961672782898, 
    0.002906249137595296, 0.057866860181093216, -0.04530937224626541, 0.041167329996824265, 
    0.013422698713839054, -0.1318880319595337, 0.03484870120882988, 0.015487991273403168, 
    -0.0006319772219285369, -0.04429081827402115, 0.04003980755805969, 0.07618117332458496, 
    -0.01989157870411873, 0.028428727760910988, -0.02314179763197899, -0.09222353249788284, 
    -0.13402140140533447, -0.0478542223572731, 0.05879896506667137, 0.00047808195813558996, 
    0.012392268516123295, 0.1252688318490982, 0.013990717008709908, 0.059829484671354294, 
    0.13810381293296814, -0.1117832139134407, 0.04174310341477394, 0.09644519537687302, 
    -0.08388844132423401, 0.051303084939718246, -0.09229395538568497, 0.033049825578927994, 
    0.062159668654203415, -0.07749707251787186, -0.02690162882208824, -0.03289502114057541, 
    0.033472076058387756, 0.006664606276899576, -0.03716803714632988, -0.049444761127233505, 
  
    };
    static const neuron_t neuron4 = {weights4, 0.0996442586183548};
    neurons[4]=neuron4;

    static const float weights5[] ={
    -0.09589487314224243, -0.059369802474975586, -0.04828272759914398, -0.05127500742673874, 
    0.041235264390707016, 0.04931139573454857, 0.04257531091570854, -0.04192989319562912, 
    -0.06515727192163467, -0.12265881896018982, 0.005795917473733425, 0.008049746043980122, 
    -0.04353925585746765, 0.08612403273582458, -0.034760843962430954, -0.021500607952475548, 
    -0.11719224601984024, -0.08500215411186218, -0.07623783499002457, -0.014116572216153145, 
    0.003756302874535322, 0.062412481755018234, -0.043232716619968414, -0.042051952332258224, 
    -0.011561558581888676, -0.10644061863422394, 0.023164408281445503, -0.034290529787540436, 
    0.06741362065076828, 0.09398335963487625, -0.03027184307575226, -0.14469365775585175, 
    -0.019304128363728523, 0.013434913009405136, -0.01880820281803608, -0.05284569784998894, 
    -0.00736061530187726, 0.040360789746046066, 0.005872049368917942, -0.03160829842090607, 
    0.03212800994515419, -0.09555936604738235, 0.040423519909381866, 0.017410220578312874, 
    0.05260709673166275, -0.013749179430305958, -0.06133732944726944, -0.0064874449744820595, 
    0.08408492803573608, -0.10017509013414383, 0.047544486820697784, -0.061323899775743484, 
    -0.042201824486255646, -0.0233494583517313, 0.07612563669681549, 0.0648815706372261, 
    0.015435640700161457, -0.02306530438363552, -0.06837965548038483, 0.08232004940509796, 
    -0.13023419678211212, -0.02688090316951275, 0.0010446907253935933, -0.06843622773885727, 
    -0.010785677470266819, -0.11569390445947647, -0.03449290618300438, 0.051502615213394165, 
    -0.066285640001297, 0.0089974794536829, -0.0631248876452446, 0.10560573637485504, 
    -0.011839808896183968, -0.03390445560216904, 0.07476916164159775, -0.05254219099879265, 
    -0.06264274567365646, -0.021824708208441734, 0.021603265777230263, 0.08326716721057892, 
    0.04320721700787544, -0.027680039405822754, -0.15409064292907715, 0.01975484937429428, 
    0.0042400420643389225, -0.013313050381839275, 0.07695454359054565, 0.004231753293424845, 
    0.014933626167476177, 0.009676591493189335, -0.011569789610803127, -0.05728267878293991, 
    0.006602277513593435, 0.08559934794902802, -0.01324683241546154, 0.1924753338098526, 
    0.030931968241930008, -0.016740988940000534, -0.0810893177986145, -0.03761844336986542, 
    0.022420598194003105, -0.1071169450879097, -0.056669510900974274, -0.03250587359070778, 
    0.0034253806807100773, 0.018009981140494347, -0.03139426186680794, -0.026623373851180077, 
    -0.010608210228383541, 0.13036860525608063, -0.040045879781246185, -0.0834505707025528, 
    -0.0439746268093586, -0.0023876202758401632, 0.03230535238981247, -0.03432336077094078, 
    0.04640306532382965, 0.04467803239822388, 0.027788987383246422, -0.05200008302927017, 
    0.09510733932256699, -0.033395107835531235, -0.055952463299036026, 0.003705862443894148, 
    0.10813391953706741, 0.08171240240335464, 0.11493363976478577, 0.06823277473449707, 
    -0.04011866822838783, 0.05576591566205025, 0.04804510250687599, -0.09714999049901962, 
    0.014663690701127052, -0.03826224058866501, -0.013760175555944443, -0.18882907927036285, 
    0.13071538507938385, 0.06515287607908249, 0.0429004430770874, 0.0039748540148139, 
    -0.1324337273836136, -0.06268839538097382, 0.017864400520920753, -0.024628309532999992, 
    0.12204942852258682, 0.027334505692124367, 0.005887849256396294, 0.04293473809957504, 
    -0.03406000882387161, -0.03920925408601761, -0.037735000252723694, 0.09259501844644547, 
    -0.0736028403043747, -0.07531216740608215, 0.0074395304545760155, 0.05104343220591545, 
    0.059409577399492264, -0.05236354470252991, -0.020957911387085915, -0.06291163712739944, 
    0.06400023400783539, -0.02299157902598381, -0.02571258507668972, -0.031949885189533234, 
    0.03461787849664688, 0.035481344908475876, -0.08199508488178253, 0.040512338280677795, 
    0.03591570258140564, 0.0036084111779928207, 0.08500288426876068, 0.07488913834095001, 
    -0.10643696039915085, 0.019360294565558434, -0.06692756712436676, 0.0548740029335022, 
    0.0222898218780756, 0.012871567159891129, 0.036590658128261566, -0.025969313457608223, 
    -0.009914141148328781, 0.030320407822728157, 0.06223602592945099, -0.02136177383363247, 
    -0.040657661855220795, -0.09297844767570496, 0.008479210548102856, 0.017259033396840096, 
    -0.09266669303178787, 0.029599063098430634, -0.06617723405361176, -0.07464532554149628, 
    0.0820431187748909, 0.04096371307969093, 0.03775429725646973, -0.00047506202827207744, 
    -0.07294131815433502, 0.0041193594224750996, -0.04333806410431862, -0.1359645575284958, 
    0.040664155036211014, -0.0431705042719841, 0.08362208306789398, 0.06968586891889572, 
    0.06260350346565247, -0.10087311267852783, -0.007328232750296593, 0.002795698819682002, 
    -0.11098190397024155, -0.08250239491462708, 0.005327966529875994, -0.026637721806764603, 
    -0.014115076512098312, -0.060143016278743744, -0.04858347401022911, 0.018419865518808365, 
    -0.13912057876586914, 0.007574961520731449, -0.04209725931286812, 0.023524966090917587, 
    -0.12159851938486099, 0.0361284501850605, 0.014651648700237274, 0.09723695367574692, 
    -0.03419185429811478, 0.0640062540769577, 0.11937953531742096, 0.09966801851987839, 
    0.0005324649391695857, -0.041768498718738556, 0.03130365163087845, -0.0013802312314510345, 
    -0.03247584402561188, 0.05058114230632782, -0.04822884872555733, 0.02536838687956333, 
    -0.05843159928917885, 0.10057152062654495, 0.022676492109894753, -0.0232265442609787, 
    0.0045992536470294, -0.0732375755906105, -0.04189198091626167, 0.07136832922697067, 
    -0.14718636870384216, -0.025622691959142685, 0.028954217210412025, 0.03326563164591789, 
    -0.00808948278427124, -0.0009042638703249395, -0.00790128018707037, -0.01617579720914364, 
    0.07481994479894638, -0.03379763290286064, 0.08010590821504593, -0.02186492830514908, 
    0.039430055767297745, -0.15509459376335144, -0.03735765814781189, 0.0396868921816349, 
    0.03827080503106117, -0.008040111511945724, 0.042533840984106064, 0.058256786316633224, 
    -0.08032871037721634, -0.026565957814455032, -0.004344999324530363, 0.038145795464515686, 
    -0.07428243011236191, -0.09156831353902817, -0.11415805667638779, 0.09863100945949554, 
    0.026424504816532135, -0.06539769470691681, -0.09574846178293228, 0.10732315480709076, 
    0.025001803413033485, -0.0402393713593483, 0.13254083693027496, -0.08508861809968948, 
    -0.05014897882938385, 0.00546257896348834, 0.05296977236866951, 0.039269205182790756, 
    0.027565952390432358, -0.06067674234509468, 0.013746273703873158, -0.06461438536643982, 
    -0.06628061085939407, -0.02509274147450924, -0.013439045287668705, 0.02364533022046089, 
    0.004155605565756559, -0.004129769746214151, -0.13324178755283356, -0.020593438297510147, 
    -0.07124218344688416, 0.04790131747722626, 0.10617194324731827, 0.007828854024410248, 
    -0.11592745035886765, 0.09987051039934158, -0.12096098065376282, 0.018711742013692856, 
    -0.06139048933982849, 0.023234685882925987, 0.05064857751131058, 0.011606505140662193, 
    0.05918610841035843, -0.1531563103199005, -0.031116746366024017, -0.18070241808891296, 
    -0.0612187534570694, 0.011829892173409462, 0.09179610759019852, -0.04471652954816818, 
    0.0632876306772232, 0.004206209909170866, -0.0035077552311122417, 0.02008841000497341, 
    -0.009816814213991165, -0.021100901067256927, 0.11605926603078842, 0.008306109346449375, 
    0.02505459263920784, 0.018370289355516434, -0.10540014505386353, -0.0590386763215065, 
    -0.0415073037147522, 0.01885656639933586, 0.004332123324275017, -0.16591912508010864, 
    -0.015526541508734226, -0.0912833884358406, -0.01615677773952484, 0.055045273154973984, 
    0.08396286517381668, -0.014453599229454994, 0.1480107307434082, -0.0014137087855488062, 
    -0.09619391709566116, 0.11677925288677216, 0.07601461559534073, 0.0065560354851186275, 
    -0.012723173014819622, -0.08641504496335983, 0.017158012837171555, -0.08762101829051971, 
    0.11333537846803665, 0.0633556991815567, 0.05602988228201866, -0.004439961165189743, 
    -0.0611000694334507, 0.18007080256938934, -0.07364943623542786, 0.09356260299682617, 
    0.06911841034889221, -0.057564329355955124, 0.006293523591011763, 0.04636457562446594, 
    -0.0848548412322998, -0.028308963403105736, 0.14322620630264282, 0.024058951064944267, 
    0.028342902660369873, -0.05216396600008011, -0.0055509028024971485, -0.07429799437522888, 
    0.018467415124177933, 0.11680757999420166, 0.01958649232983589, -0.07401474565267563, 
    -0.08509796857833862, 0.04224468767642975, 0.021202052012085915, 0.03607012704014778, 
    0.03262125328183174, 0.019087444990873337, -0.0655708909034729, 0.06520795822143555, 
    -0.10795358568429947, 0.02723636105656624, -0.06750403344631195, 0.016261866316199303, 
    -0.060825783759355545, 0.09037698805332184, 0.017743444070219994, -0.11400678753852844, 
    -0.057623784989118576, -0.03155118227005005, 0.006397104822099209, -0.0033626891672611237, 
    -0.10182541608810425, 0.06290602684020996, 0.0244599599391222, -0.016730766743421555, 
    0.03622671216726303, 0.006209461018443108, -0.021194826811552048, -0.019171923398971558, 
    0.10648254305124283, 0.07445333153009415, 0.10440005362033844, 0.0007038132753223181, 
    -0.09723848849534988, -0.11286504566669464, 0.09685157984495163, 0.003806221531704068, 
    0.00407051108777523, 0.08927595615386963, -0.027187148109078407, 0.07284975796937943, 
    0.0043411944061517715, -0.10012535750865936, 0.0247051864862442, 0.048606228083372116, 
    -0.05250541865825653, -0.05010427162051201, -0.019780751317739487, 0.059593867510557175, 
    -0.004745300859212875, -0.14435043931007385, 0.0196093637496233, -0.05018245056271553, 
    0.03535671904683113, 0.018451794981956482, 0.04594375938177109, -0.0287797711789608, 
    0.047874752432107925, 0.09560590982437134, -0.07644784450531006, -0.011317756958305836, 
    0.03671128302812576, 0.036492541432380676, -0.038793329149484634, -0.10406947880983353, 
    0.02997301146388054, 0.020701764151453972, -0.04281354323029518, -0.06925386190414429, 
    0.06488616019487381, -0.00770528195425868, 0.005027616396546364, 0.037395279854536057, 
    0.02186676673591137, 0.05895388126373291, -0.09282264113426208, 0.030275242403149605, 
    -0.0034376797266304493, 0.018637293949723244, -0.09966516494750977, -0.018919486552476883, 
    0.07748949527740479, -0.05359559878706932, -0.03369076922535896, 0.06342247128486633, 
    -0.04593401774764061, 0.07629768550395966, -0.1934162974357605, -0.065736785531044, 
    -0.08918250352144241, -0.018668241798877716, 0.043107740581035614, 0.047572772949934006, 
    0.048439063131809235, -0.043060459196567535, -0.13046756386756897, -0.013822630979120731, 
    0.044413477182388306, -0.027312826365232468, 0.022724078968167305, 0.1391318440437317, 
    0.04676740616559982, -0.017698384821414948, -0.061824094504117966, 0.0820324495434761, 
    0.0824718326330185, 0.031467046588659286, -0.01910424791276455, -0.01754966750741005, 
    -0.000911004317458719, 0.06958635151386261, -0.07949740439653397, 0.06481939554214478, 
    -0.04043015465140343, 0.04329811781644821, -0.09235388040542603, -0.20047296583652496, 
    -0.01761634834110737, -0.031720153987407684, 0.05760842561721802, 0.03848356753587723, 
    0.014229852706193924, -0.02058335393667221, 0.021736830472946167, -0.08039069920778275, 
    -0.024066494777798653, -0.01445010770112276, -0.10901020467281342, 0.047200221568346024, 
    0.02848672680556774, -0.08891391009092331, 0.036274462938308716, 0.018649516627192497, 
    0.05677790939807892, 0.011558383703231812, -0.0020031786989420652, -0.015006056986749172, 
    -0.042730335146188736, -0.06995728611946106, -0.02223687246441841, -0.02081908844411373, 
  
    };
    static const neuron_t neuron5 = {weights5, -0.04839489236474037};
    neurons[5]=neuron5;

    static const float weights6[] ={
    0.01598341017961502, -0.09749697893857956, 0.0050590019673109055, 0.12565864622592926, 
    0.02085825614631176, 0.08767274022102356, -0.07556460797786713, -0.0017978358082473278, 
    0.08229193091392517, 0.08228274434804916, 0.143389031291008, 0.003589339554309845, 
    -0.12640362977981567, -0.04370268061757088, 0.04819175601005554, -0.12962917983531952, 
    -0.08306820690631866, 0.04899617284536362, -0.04140833765268326, -0.025639768689870834, 
    -0.08404805511236191, 0.036991048604249954, -0.03942066431045532, 0.038785532116889954, 
    0.09129087626934052, 0.13328363001346588, -0.0079429242759943, -0.022497029975056648, 
    0.05943240597844124, 0.06778101623058319, 0.07457788288593292, 0.11912494152784348, 
    -0.11350508779287338, -0.006316307000815868, -0.09777297079563141, -0.10088137537240982, 
    0.031140819191932678, -0.12935103476047516, 0.0807327851653099, -0.0927983820438385, 
    -0.1552605926990509, 0.08853664249181747, 0.02200382389128208, 0.0493280403316021, 
    0.09148461371660233, 0.0464581735432148, 0.02828201651573181, 0.13810178637504578, 
    0.02400483563542366, 0.023967258632183075, -0.012526172213256359, 0.00569629343226552, 
    0.027349665760993958, -0.00556704169139266, -0.11405805498361588, -0.04908410459756851, 
    0.006605187430977821, -0.12091278284788132, 0.011516740545630455, 0.13171762228012085, 
    -0.07025115191936493, -0.08795555680990219, 0.1044674664735794, -0.07147982716560364, 
    0.06714698672294617, 0.02788984775543213, 0.008789236657321453, 0.04939555749297142, 
    0.12002476304769516, 0.00032386387465521693, -0.12303261458873749, -0.018940534442663193, 
    -0.10370316356420517, -0.14732520282268524, 0.04548555985093117, -0.06475264579057693, 
    0.05771004781126976, -0.017784487456083298, -0.0841294378042221, -0.002310382202267647, 
    0.0495634600520134, 0.024162981659173965, -0.08693484216928482, 0.015152277424931526, 
    -0.12791629135608673, 0.05668150261044502, -0.01600916124880314, 0.009781943634152412, 
    0.10091378539800644, -0.1552411913871765, -0.07331479340791702, 0.05508539080619812, 
    0.04076642170548439, -0.009612813591957092, -0.0026207431219518185, 0.04193578287959099, 
    -0.09405060112476349, 0.04661943018436432, -0.02846515364944935, -0.16704194247722626, 
    0.07899141311645508, -0.07858259975910187, 0.08929037302732468, -9.655137546360493e-06, 
    -0.06694712489843369, -0.06874871999025345, 0.012237852439284325, -0.18303078413009644, 
    -0.0648297667503357, -0.12336113303899765, -0.04628222808241844, -0.03644693270325661, 
    -0.11699403077363968, -0.0825275108218193, 0.06422808766365051, -0.07620802521705627, 
    -0.03147648274898529, 0.12093304097652435, -0.08333110809326172, 0.044326115399599075, 
    -0.05202016234397888, -0.05482499301433563, -0.007535681128501892, 0.17709335684776306, 
    0.14369022846221924, 0.05959094688296318, 0.12341907620429993, -0.12211813032627106, 
    0.04892550781369209, 0.03154947608709335, 0.04180622845888138, -0.019645486027002335, 
    0.03905249014496803, -0.0759962648153305, 0.03384972736239433, -0.08940854668617249, 
    -0.08249411731958389, -0.02881491184234619, -0.042784519493579865, 0.06838095188140869, 
    -0.05451880395412445, -0.13472339510917664, 0.09659269452095032, -0.062416739761829376, 
    0.07771972566843033, 0.061319027096033096, 0.01615375094115734, 0.09938394278287888, 
    0.023349570110440254, -0.1725061982870102, 0.02057218737900257, 0.031021563336253166, 
    0.030899902805685997, -0.040694527328014374, -0.06559427082538605, 0.027447214350104332, 
    0.06062819063663483, 0.049065910279750824, -0.12788866460323334, -0.14171560108661652, 
    0.02203015610575676, 0.07497061789035797, -0.13869139552116394, 0.020033299922943115, 
    -0.010531127452850342, -0.08165892213582993, -0.0582144558429718, 0.017836114391684532, 
    0.01437835581600666, 0.12933234870433807, -0.05633072555065155, 0.0785856619477272, 
    -0.04233618453145027, -0.09253336489200592, 0.025455497205257416, 0.02758125774562359, 
    0.013939376920461655, -0.06740553677082062, -0.04876779019832611, 0.03287208825349808, 
    -0.014882514253258705, 0.07955652475357056, -0.0076124477200210094, -0.006034491583704948, 
    -0.04711874574422836, 0.1029089093208313, 0.0913255587220192, 0.024000298231840134, 
    -0.009984654374420643, 0.003581942990422249, -0.12305489927530289, 0.12089285254478455, 
    -0.1256939023733139, -0.1338079422712326, -0.06038225069642067, 0.056771419942379, 
    0.0739644393324852, 0.08473812788724899, -0.07789556682109833, -0.019162438809871674, 
    -0.07980397343635559, 0.06892410665750504, 0.09684522449970245, -0.09037066251039505, 
    0.031816836446523666, 0.022302160039544106, -0.027262505143880844, -0.026461275294423103, 
    -0.1314978450536728, 0.10810158401727676, 0.130192369222641, 0.00282076233997941, 
    -0.0026317830197513103, 0.08151053637266159, 0.08255105465650558, -0.005291427485644817, 
    0.11253557354211807, -0.028181754052639008, -0.03497698903083801, -0.032691072672605515, 
    0.02701171673834324, 0.04434918239712715, 0.04541191831231117, 0.0007714479579590261, 
    -0.014094267040491104, 0.018943926319479942, 0.16369663178920746, 0.06120910868048668, 
    0.020767701789736748, -0.013473497703671455, 0.14483767747879028, 0.1989278495311737, 
    -0.016206692904233932, 0.03697087615728378, 0.02607678808271885, 0.08424083888530731, 
    -0.0033402235712856054, 0.054158151149749756, 0.023578815162181854, -0.08203097432851791, 
    -0.09256776422262192, 0.041902318596839905, -0.016053615137934685, -0.038511186838150024, 
    0.0658460333943367, -0.07825195044279099, 0.0058993264101445675, -0.09232316166162491, 
    -0.06562674790620804, 0.10965035110712051, -0.1102157011628151, -0.09575330466032028, 
    0.013691688887774944, 0.016677765175700188, -0.015044555068016052, -0.03416356071829796, 
    0.02124764584004879, 0.022505085915327072, -0.028532739728689194, 0.08831755816936493, 
    0.11949629336595535, -0.0373421311378479, -0.07986948639154434, -0.020678415894508362, 
    0.10308212786912918, -0.11400696635246277, 0.005915145855396986, 0.058574650436639786, 
    -0.041791751980781555, -0.017574340105056763, 0.03856094554066658, 0.0013567717978730798, 
    0.03409639373421669, -0.013091281056404114, -0.20265409350395203, -0.025925831869244576, 
    0.06858127564191818, -0.09819556027650833, 0.1132001206278801, -0.08158967643976212, 
    -0.004007501527667046, 0.11675002425909042, 0.0811571553349495, -0.14725089073181152, 
    -0.15382717549800873, -0.03167333826422691, -0.011348049156367779, -0.006543534807860851, 
    -0.08566223084926605, 0.05646675080060959, 0.08649188280105591, -0.08021725714206696, 
    -0.017830250784754753, -0.07475509494543076, 0.024881744757294655, -0.09261594712734222, 
    -0.06651609390974045, -0.032288577407598495, 0.06382445245981216, -0.03680616244673729, 
    -0.009554947726428509, 0.07627912610769272, 0.1427793800830841, -0.0987226590514183, 
    0.005039389245212078, 0.10885470360517502, 0.011697807349264622, 0.12926322221755981, 
    -0.11666552722454071, 0.07681569457054138, -0.019146978855133057, -0.07603783905506134, 
    -0.0012823061551898718, -0.0064028422348201275, -0.05079973116517067, 0.12935268878936768, 
    0.031433623284101486, 0.048679668456315994, -0.05814139172434807, -0.0027704655658453703, 
    -0.008921870030462742, 0.09932272136211395, 0.13994824886322021, 0.047131262719631195, 
    0.13942161202430725, 0.015579145401716232, -0.015964558348059654, -0.11463548243045807, 
    -0.006938347592949867, -0.020450882613658905, -0.007940312847495079, -0.06496236473321915, 
    -0.03569662198424339, 0.11347898840904236, 0.014967282302677631, 0.04475434496998787, 
    0.04557270184159279, -0.005592328496277332, -0.08124322444200516, -0.016827793791890144, 
    0.05075128749012947, 0.021834595128893852, 0.05486072972416878, 0.13766993582248688, 
    -0.011660370975732803, -0.1311953216791153, -0.00832001119852066, -0.01883147656917572, 
    -0.09289205074310303, -0.03394196182489395, -0.049750953912734985, 0.03427493944764137, 
    0.011508719995617867, -0.04689004644751549, 0.017225516960024834, 0.049106236547231674, 
    -0.12629495561122894, -0.11611922085285187, -0.07054571807384491, -0.07697916030883789, 
    0.12160684913396835, 0.0944424644112587, -0.03764817491173744, 0.00044614647049456835, 
    0.06886857748031616, -0.05405793339014053, 0.019494595006108284, 0.04686662554740906, 
    0.04899797588586807, -0.05504217371344566, 0.1401938498020172, 0.020668702200055122, 
    -0.05058061704039574, 0.026401519775390625, -0.027075132355093956, 0.04933931678533554, 
    0.02796190232038498, 0.12427309155464172, 0.041032273322343826, -0.02425037883222103, 
    0.05363551899790764, 0.05824919044971466, -0.01861693523824215, -0.049775805324316025, 
    0.03823054954409599, 0.08839499950408936, -0.10999029874801636, -0.1278725266456604, 
    0.016232820227742195, -0.06957101076841354, -0.11210960894823074, 0.024534327909350395, 
    -0.033217210322618484, 0.04463088512420654, 0.04534366354346275, 0.0946803167462349, 
    0.20450837910175323, 0.06599606573581696, -0.08554575592279434, 0.13615909218788147, 
    0.039769742637872696, -0.151332288980484, -0.08841795474290848, -0.14229567348957062, 
    0.048544742166996, -0.1024891659617424, 0.021286427974700928, -0.00021592080884147435, 
    0.06074732542037964, 0.07126423716545105, 0.12049801647663116, 0.023523692041635513, 
    0.04653413966298103, -0.011586056090891361, 0.030994955450296402, 0.09870848059654236, 
    0.020105071365833282, -0.08714151382446289, -0.02681129053235054, 0.048560623079538345, 
    0.04232550412416458, -0.05651256814599037, 0.06962413340806961, -0.005041108001023531, 
    -0.0310991108417511, 0.025060363113880157, -0.05571538954973221, 0.05845414474606514, 
    0.05018402636051178, 0.0875021293759346, -0.087933748960495, 0.047469135373830795, 
    -0.010638002306222916, 0.14972257614135742, -0.05487850308418274, -0.03154478222131729, 
    -0.03051821142435074, 0.04823338985443115, 0.09631841629743576, 0.1411927342414856, 
    0.09249185770750046, -0.04973398521542549, 0.035966865718364716, 0.22781416773796082, 
    -0.09986331313848495, -0.07859023660421371, 0.08730098605155945, -0.06102313473820686, 
    -0.006539092864841223, 0.014339552260935307, -0.0020898934453725815, -0.01696304976940155, 
    0.00011078472016379237, -0.028580641373991966, 0.017799533903598785, -0.079940065741539, 
    0.0025246271397918463, -0.10168062895536423, 0.010915004648268223, 0.010588173754513264, 
    -0.06963532418012619, 0.09127188473939896, -0.02940620481967926, 0.17521390318870544, 
    0.11214010417461395, -0.1369474232196808, 0.13041216135025024, -0.15760651230812073, 
    0.015379169024527073, 0.07318306714296341, -0.004063223022967577, 0.026777883991599083, 
    0.18395096063613892, -0.1469748467206955, -0.14972949028015137, -0.11199720203876495, 
    0.04790029302239418, 0.06416390836238861, 0.03880739584565163, 0.011949306353926659, 
    -0.16248787939548492, -0.01628970541059971, -0.0541357696056366, 0.056034356355667114, 
    0.0869663655757904, -0.07642485201358795, -0.034594032913446426, -0.0992913618683815, 
    0.027589157223701477, -0.020568255335092545, 0.07395827770233154, -0.021899458020925522, 
    0.023257756605744362, 0.03313352167606354, 0.03639412671327591, -0.06298191845417023, 
    0.07687830179929733, 0.03844646364450455, -0.07585196942090988, 0.058859556913375854, 
    -0.09670945256948471, 0.0937381461262703, 0.07814078032970428, -0.03946799784898758, 
    0.026301834732294083, -0.1231667771935463, 0.06205655634403229, -0.020504914224147797, 
    -0.1589631289243698, 0.07945435494184494, 0.003996868152171373, 0.03571746125817299, 
  
    };
    static const neuron_t neuron6 = {weights6, 0.22412531077861786};
    neurons[6]=neuron6;

    static const float weights7[] ={
    0.06536511331796646, 0.032338257879018784, 0.009127707220613956, -0.031259141862392426, 
    -0.020528459921479225, -0.13482987880706787, 0.03632061928510666, -0.08756350725889206, 
    -0.05399984121322632, -0.06550879776477814, -0.0023334284778684378, -0.06782632321119308, 
    -0.12226646393537521, 0.10614258795976639, 0.07423325628042221, 0.10821191221475601, 
    0.07656992971897125, 0.09282910823822021, 0.04335102066397667, 0.06167195737361908, 
    0.016557443886995316, 0.06739100068807602, 0.03462592884898186, 0.013236550614237785, 
    -0.09187478572130203, -0.13201133906841278, -0.00468409014865756, 0.01932571269571781, 
    -0.034791212528944016, -0.058815110474824905, -0.0430716797709465, 0.12638695538043976, 
    -0.045584920793771744, -0.003737796563655138, -0.04287739470601082, 0.08356957137584686, 
    -0.1060335785150528, 0.06598708033561707, 0.03806227818131447, 0.012898986227810383, 
    0.1820000559091568, -0.1013985425233841, 0.07197479158639908, 0.08650103956460953, 
    -0.025549110025167465, 0.012445979751646519, -0.16618673503398895, -0.04035990312695503, 
    0.08603387326002121, -0.05904049426317215, -0.02782549150288105, 0.06373297423124313, 
    -0.048978012055158615, -0.06849619746208191, -0.031684670597314835, 0.108045294880867, 
    -0.05176691338419914, -0.03960239514708519, -0.10395892709493637, 0.04317162185907364, 
    -0.1536962240934372, 0.10449109971523285, 0.06088389456272125, 0.10216889530420303, 
    0.015281607396900654, -0.10178021341562271, -0.016688304021954536, -0.06317844241857529, 
    0.10957174003124237, 0.09395434707403183, -0.07374539971351624, 0.10221749544143677, 
    0.006290323566645384, -0.015224689617753029, -0.010061180219054222, 0.00835973396897316, 
    0.013747802935540676, -0.060089852660894394, 0.0468522347509861, 0.10005811601877213, 
    -0.1530967354774475, -0.03527745231986046, -0.04801991954445839, -0.04973282665014267, 
    -0.06936600059270859, -0.051546305418014526, 0.07445985823869705, -0.03832913935184479, 
    0.03244960680603981, 0.061862438917160034, 0.1733390986919403, 0.006211013533174992, 
    -0.11052103340625763, 0.012528317980468273, -0.026107989251613617, 0.14984813332557678, 
    0.007952157407999039, 0.10466279834508896, 0.041511040180921555, 0.01959086023271084, 
    -0.009960182942450047, 0.054391615092754364, -0.015430072322487831, -0.07735105603933334, 
    0.06684187054634094, 0.07379710674285889, 0.0031686932779848576, 0.08734525740146637, 
    0.060952119529247284, 0.05901266634464264, -0.03640855848789215, 0.0009066051570698619, 
    -0.07624403387308121, 0.11558555066585541, -0.09367861598730087, 0.05132361501455307, 
    0.15443482995033264, 0.006851624231785536, -0.08014378696680069, 0.10417810082435608, 
    0.1135156899690628, 0.11631646007299423, -0.09658153355121613, -0.07690159976482391, 
    0.06298389285802841, -0.05558519810438156, 0.1477499008178711, 0.18359321355819702, 
    -0.028079383075237274, -0.02989921160042286, -0.007556322030723095, -0.016540629789233208, 
    0.14252713322639465, 0.09387970715761185, 0.006220147479325533, 0.02163049206137657, 
    -0.006926513742655516, -0.10143361240625381, 0.02822623960673809, 0.013505501672625542, 
    -0.06523943692445755, -0.0009102183394134045, 0.1130354255437851, -0.09853454679250717, 
    0.06635891646146774, 0.019585315138101578, 0.07225777953863144, -0.07870905846357346, 
    -0.10755061358213425, -0.12170586735010147, 0.06460274755954742, -0.023663941770792007, 
    -0.09426898509263992, -0.07307852804660797, 0.03551410138607025, -0.019884595647454262, 
    -0.0017559855477884412, -0.036606092005968094, -0.10509386658668518, 0.17469611763954163, 
    0.038518037647008896, 0.003988555632531643, 0.017757723107933998, -0.1364770084619522, 
    0.03945789858698845, -0.009715130552649498, -0.030840933322906494, 0.014310955069959164, 
    0.08702170103788376, 0.12667818367481232, -0.020048800855875015, -0.050485119223594666, 
    -0.01024097204208374, -0.021226149052381516, -0.03725827485322952, -0.027912456542253494, 
    -0.11634088307619095, 0.009833506308495998, -0.1893942803144455, -0.08699429035186768, 
    -0.022422298789024353, 0.10226752609014511, 0.05297547206282616, -0.07038120925426483, 
    -0.03310756757855415, 0.019299838691949844, 0.0001504568208474666, 0.047134120017290115, 
    -0.05574781820178032, 0.11753782629966736, 0.027495626360177994, 0.07353425025939941, 
    0.017394891008734703, -0.04613756760954857, -0.015998918563127518, 0.12512025237083435, 
    -0.16108818352222443, -0.11690960079431534, 0.008567041717469692, -0.08717145025730133, 
    -0.05033654347062111, -0.11507880687713623, 0.1260506510734558, -0.04664253070950508, 
    0.024566689506173134, -0.08739110827445984, 0.02318735606968403, -0.02253524214029312, 
    0.032455939799547195, -0.0062185898423194885, -0.07874799519777298, -0.054163575172424316, 
    -0.03841328248381615, 0.11688834428787231, -0.0625559464097023, -0.019400721415877342, 
    -0.04428384453058243, 0.1219000294804573, -0.0735463798046112, -0.020875118672847748, 
    -0.09316465258598328, 0.002513073617592454, -0.04501726105809212, 0.048552725464105606, 
    0.0068769496865570545, 0.05745447799563408, -0.03880137577652931, 0.10542017221450806, 
    -0.12141643464565277, 0.015175140462815762, 0.08798545598983765, -0.010693555697798729, 
    0.16274312138557434, -0.057538535445928574, -0.08145873248577118, -0.02591537870466709, 
    -0.061078015714883804, -0.1078171581029892, 0.11032853275537491, 0.023080799728631973, 
    -0.09340153634548187, 0.08848658204078674, -0.06221462041139603, 0.0642906054854393, 
    0.011782889254391193, -0.03971691429615021, 0.06293096393346786, 0.14611278474330902, 
    0.012383170425891876, -0.07905401289463043, 0.015514752827584743, -0.19908155500888824, 
    0.11223725974559784, 0.0063877105712890625, 0.07671382278203964, 0.021176928654313087, 
    0.044341348111629486, -0.038930706679821014, -0.11140391230583191, -0.05767150968313217, 
    0.048787921667099, 0.03378821536898613, 0.0792590007185936, -0.02382648177444935, 
    -0.18046845495700836, 0.08313903212547302, -0.11831630766391754, 0.014064407907426357, 
    0.1389847844839096, -0.014709270559251308, -0.04873873293399811, -0.045192863792181015, 
    0.04207710549235344, -0.05213514342904091, -0.02535928040742874, -0.08473773300647736, 
    -0.010144736617803574, 0.09638676047325134, -0.015761682763695717, -0.06875191628932953, 
    -0.08455941826105118, -0.0015549629461020231, -0.11031949520111084, 0.0025708917528390884, 
    0.09823547303676605, 0.00980337429791689, 0.04149968922138214, -0.04043646901845932, 
    0.0399806946516037, 0.06239423155784607, 0.0758068636059761, 0.02102341689169407, 
    -0.2136131227016449, -0.001631016843020916, 0.03204606473445892, -0.007936486974358559, 
    -0.07999013364315033, -0.08861912041902542, -0.008669382892549038, -0.04524853453040123, 
    -0.05435200408101082, -0.04120376333594322, -0.04292495921254158, -0.0675143301486969, 
    -0.14903351664543152, -0.008535879664123058, -0.09330256283283234, -0.03731226176023483, 
    0.019433965906500816, 0.005423509515821934, -0.00911210011690855, 0.030861254781484604, 
    -0.029279867187142372, 0.03364723175764084, 0.10430499166250229, -0.059847258031368256, 
    -0.09204035997390747, -0.026311123743653297, -0.03842438384890556, 0.028701771050691605, 
    0.03661257401108742, 0.051480814814567566, -0.04943472519516945, 0.05768284574151039, 
    0.11052674800157547, -0.14064982533454895, 0.11067280173301697, -0.1262948364019394, 
    0.1911115050315857, 0.006316155660897493, -0.01606096513569355, -0.02116612158715725, 
    -0.030393341556191444, -0.0735369473695755, -0.03572816029191017, -0.030980069190263748, 
    -0.04783133417367935, -0.016530241817235947, 0.048709936439991, 0.1543113738298416, 
    -0.03948911651968956, 0.11866386234760284, -0.06040911376476288, 0.04382221773266792, 
    0.01680760085582733, -0.09915581345558167, -0.04066560044884682, -0.03672724589705467, 
    0.06859445571899414, -0.05030020698904991, 0.07159452140331268, 0.06981520354747772, 
    0.04814915731549263, 0.16296181082725525, 0.07486507296562195, -0.0351940281689167, 
    0.11052924394607544, 0.013592240400612354, -0.07468100637197495, -0.027826745063066483, 
    -0.048641689121723175, -0.15983958542346954, 0.06189733371138573, 0.032452140003442764, 
    -0.20039555430412292, 0.02627335675060749, -0.02065211907029152, 0.12139502167701721, 
    -0.119253970682621, -0.030799707397818565, -0.0010905254166573286, -0.010442412458360195, 
    -0.0670396164059639, -0.06836193054914474, 0.026297491043806076, 0.005486070644110441, 
    0.05044545978307724, -0.04532306268811226, -0.02871593087911606, -0.07712417095899582, 
    -0.020313633605837822, 0.18491972982883453, -0.07510969042778015, -0.023885536938905716, 
    0.02231929637491703, -0.0031049014069139957, 0.0354643240571022, -0.04381344094872475, 
    0.0035507886204868555, 0.024291906505823135, 0.04107121750712395, 0.06506361812353134, 
    0.11300869286060333, 0.028645990416407585, 0.1229357123374939, 0.040628887712955475, 
    -0.07621470093727112, -0.09086927771568298, 0.05471594259142876, 0.08619444072246552, 
    0.02905440703034401, 0.027578940615057945, -0.04141613468527794, 0.07289544492959976, 
    -0.042708870023489, 0.11661891639232635, -0.034879520535469055, -0.1246974989771843, 
    -0.030120566487312317, 0.1685510277748108, 0.09807946532964706, -0.08782442659139633, 
    -0.058195989578962326, -0.04300928860902786, 0.04602700099349022, -0.03855552896857262, 
    0.06945431232452393, 0.046530984342098236, 0.14874756336212158, -0.01837052032351494, 
    0.1449732780456543, 0.034252751618623734, -0.0724123865365982, -0.07086540758609772, 
    0.06802178174257278, -0.060360413044691086, 0.03995148465037346, -0.1261930614709854, 
    -0.0014673119876533747, -0.006618486251682043, 0.1095956340432167, -0.10596801340579987, 
    -0.02728653885424137, 0.1253238022327423, 0.10928232222795486, -0.13505756855010986, 
    0.030310239642858505, 0.16545510292053223, 0.05231089890003204, -0.06851456314325333, 
    -0.04355157911777496, 0.041008174419403076, -0.1116199865937233, 0.03601895272731781, 
    -0.0993473008275032, -0.08859124779701233, 0.03198172152042389, 0.042719289660453796, 
    0.06869219243526459, -0.09500418603420258, -0.040005896240472794, 0.09088163822889328, 
    0.0654507577419281, 0.07423879951238632, -0.06665778160095215, 0.001675362465903163, 
    -0.18331992626190186, 0.06973017007112503, -0.07143370062112808, -0.02105000428855419, 
    -0.011774460785090923, -0.07251125574111938, 0.022217154502868652, 0.0021289365831762552, 
    0.16169607639312744, -0.12273067981004715, -0.05407734215259552, 0.019196568056941032, 
    0.07474011182785034, -0.0770266056060791, 0.03218929469585419, 0.00024944503093138337, 
    -0.06174375116825104, 0.1007956862449646, 0.04124745354056358, 0.11901553720235825, 
    -0.024558495730161667, 0.044324882328510284, -0.02410684898495674, 0.011808168143033981, 
    0.07955776900053024, 0.15586581826210022, -0.02915896289050579, 0.0990392342209816, 
    0.043708086013793945, 0.07329888641834259, -0.06333645433187485, -0.0035015929024666548, 
    0.026511654257774353, 0.06801807135343552, 0.05006946250796318, -0.0321754552423954, 
    -0.10028358548879623, 0.005797581281512976, 0.019131572917103767, -0.004945253022015095, 
    -0.02671992778778076, -0.0326586589217186, 0.18379825353622437, -0.04570845887064934, 
    0.04899111017584801, -0.003742292057722807, 0.010319871827960014, -0.00946173444390297, 
    0.03385849669575691, 0.019756777212023735, -0.028275098651647568, -0.05791034549474716, 
    -0.019509317353367805, -0.07174400240182877, -0.048453278839588165, 0.020468641072511673, 
  
    };
    static const neuron_t neuron7 = {weights7, -0.32285812497138977};
    neurons[7]=neuron7;

    static const float weights8[] ={
    0.011429224163293839, 0.09462155401706696, -0.08517656475305557, 0.26133275032043457, 
    -0.02598045952618122, -0.16369417309761047, -0.07098282128572464, -0.09907487779855728, 
    0.0017000382067635655, 0.07352475821971893, -0.16078580915927887, 0.05219466611742973, 
    0.1105215921998024, 0.026447569951415062, 0.11351875960826874, 0.06278631836175919, 
    -0.08997522294521332, 0.048028334975242615, -0.05371857061982155, -0.027525994926691055, 
    -0.062444038689136505, -0.07484236359596252, -0.04347413405776024, 0.02932884357869625, 
    0.060180503875017166, 0.1311294138431549, 0.04694544896483421, -0.08762070536613464, 
    -0.009679599665105343, 0.03403400257229805, 0.009863889776170254, -0.045628756284713745, 
    -0.03183382377028465, -0.041925281286239624, 0.039148714393377304, 0.008048136718571186, 
    0.04703662544488907, 0.0055494485422968864, 0.024415520951151848, -0.05671294033527374, 
    0.0371006615459919, 0.11822649836540222, -0.0908576250076294, -0.01915588416159153, 
    -0.07800152897834778, -0.05626773461699486, 0.07024809718132019, 0.021599719300866127, 
    0.048937540501356125, 0.057537227869033813, 0.009588567540049553, -0.005093434825539589, 
    0.09226066619157791, -0.07437926530838013, -0.11385791748762131, -0.03945682570338249, 
    0.15439996123313904, 0.06416931748390198, 0.08458305895328522, -0.08564646542072296, 
    0.1523435413837433, -0.05441707372665405, -0.004789224825799465, -0.044955506920814514, 
    0.08768343180418015, 0.05584504455327988, 0.012412499636411667, -0.09110980480909348, 
    -0.02335299737751484, 0.011096091009676456, 0.12220119684934616, -0.005567295476794243, 
    0.12708792090415955, 0.08559797704219818, 0.10696876794099808, -0.10687918215990067, 
    -0.036269862204790115, -0.060333382338285446, -0.07014068961143494, 0.049522656947374344, 
    -0.0694471076130867, -0.0439189188182354, 0.15132130682468414, 0.05163625627756119, 
    -0.013972793705761433, 0.003865469479933381, -0.058685168623924255, 0.15119768679141998, 
    -0.11120468378067017, -0.10679790377616882, 0.14857473969459534, 0.021747486665844917, 
    -0.00599910831078887, -0.04483633488416672, 0.14886009693145752, -0.019917063415050507, 
    -0.04213660955429077, -0.13391442596912384, -0.09025081247091293, 0.11019277572631836, 
    -0.02946382388472557, 0.08631279319524765, 0.08009417355060577, 0.02397572249174118, 
    0.09965554624795914, 0.12728416919708252, -0.01976665109395981, -0.08171823620796204, 
    -0.10460172593593597, 0.024897202849388123, 0.02153189666569233, -0.08969740569591522, 
    0.048715200275182724, -0.10525073856115341, 0.04666294530034065, -0.0575227253139019, 
    -0.08656913787126541, -0.07513740658760071, -0.09916689246892929, -0.009161976166069508, 
    -0.04051157087087631, 0.0270463228225708, -0.05909815430641174, 0.059851549565792084, 
    0.06551478803157806, -0.0260117556899786, -0.04956579580903053, 0.0038056045304983854, 
    -0.051849544048309326, -0.09317471086978912, -0.03964696824550629, -0.08478686213493347, 
    -0.0021865456365048885, 0.12839503586292267, -0.0896381214261055, 0.12209059298038483, 
    -0.13848666846752167, -0.034082598984241486, -0.18416845798492432, 0.11464328318834305, 
    0.03327268734574318, 0.0775977224111557, -0.1276390105485916, -0.03562067449092865, 
    -0.10422820597887039, -0.03626793622970581, -0.04839346185326576, 0.0023860232904553413, 
    -0.12745480239391327, 0.04612705484032631, -0.0023192951921373606, -0.0655927062034607, 
    0.10383942723274231, 0.11097127944231033, -0.10707312822341919, -0.12975484132766724, 
    -0.024257758632302284, 0.06309456378221512, 0.09628206491470337, -0.07755082100629807, 
    -0.049579259008169174, -0.1118621900677681, 0.06553104519844055, -0.05790954455733299, 
    0.15913912653923035, 0.0028316345997154713, -0.013818989507853985, -0.09643222391605377, 
    0.1291668564081192, -0.10058021545410156, 0.081022709608078, 0.018272103741765022, 
    0.07165279239416122, 0.19538700580596924, 0.04448794201016426, -0.0015699556097388268, 
    -0.013576100580394268, 0.006836859043687582, 0.028370484709739685, -0.01097728032618761, 
    -0.029887929558753967, -0.029264070093631744, -0.10213889181613922, 0.03435080498456955, 
    0.09550582617521286, -0.11749545484781265, 0.1362236738204956, -0.06024007871747017, 
    0.023948542773723602, 0.048903945833444595, -0.04483601823449135, -0.16539965569972992, 
    0.15102314949035645, 0.14468462765216827, 0.0043339114636182785, 0.07099078595638275, 
    -0.04625273123383522, 0.0970185250043869, 0.09472331404685974, -0.009277747012674809, 
    0.018665658310055733, 0.1212020218372345, -0.10566433519124985, 0.05806142836809158, 
    -0.012782366015017033, 0.08155342936515808, -0.04604749754071236, 0.041234537959098816, 
    -0.16190209984779358, -0.050720151513814926, -0.06550006568431854, -6.478358409367502e-05, 
    0.0473816953599453, -0.13635703921318054, -0.13492682576179504, 0.07601011544466019, 
    0.15075372159481049, 0.031064338982105255, 0.040242355316877365, -0.15879808366298676, 
    0.08618892729282379, -0.026510879397392273, -0.02930041402578354, 0.0605299137532711, 
    0.06145016849040985, -0.10493154078722, -0.18208611011505127, 0.029198145493865013, 
    -0.13218821585178375, -0.0010556718334555626, -0.016654763370752335, 0.04038611426949501, 
    -0.05234555900096893, -0.07990313321352005, 0.15522438287734985, -0.029668215662240982, 
    -0.005770362447947264, 0.08501084893941879, 0.03509964048862457, 0.02066403068602085, 
    -0.03984055668115616, 0.009681652300059795, 0.1039309874176979, -0.016125446185469627, 
    -0.027031948789954185, 0.011164527386426926, -0.05577842518687248, -0.02891218662261963, 
    0.0034022803883999586, 0.025991195812821388, 0.0380086675286293, -0.06469722837209702, 
    -0.02588094025850296, 0.05413634702563286, -0.0334843173623085, -0.0950673371553421, 
    0.06387904286384583, -0.05987895280122757, -0.033292025327682495, -0.015302354469895363, 
    -0.03503458574414253, 0.09829460084438324, -0.03462189435958862, -0.0283190980553627, 
    0.07227214425802231, 0.08179336786270142, -0.011828562244772911, -0.03701234981417656, 
    -0.05589453876018524, 0.02852710150182247, -0.0044661457650363445, -0.09156830608844757, 
    -0.12654080986976624, -0.045674849301576614, 0.012298437766730785, -0.15961433947086334, 
    -0.01443216297775507, -0.05165080726146698, -0.03003392368555069, 0.05451984331011772, 
    0.20637083053588867, -0.010769289918243885, 0.1479889154434204, 0.13664719462394714, 
    0.07674296200275421, -0.02580311894416809, -0.07002530992031097, 0.035540901124477386, 
    0.015287782065570354, -0.06931424885988235, -0.01115534920245409, -0.04539386183023453, 
    0.09982553124427795, -0.10773779451847076, 0.05993812158703804, 0.04970439150929451, 
    0.037210863083601, 0.16053587198257446, -0.13728904724121094, 0.1517874002456665, 
    -0.03965461999177933, 0.027416029945015907, -0.038862451910972595, -0.01445151399821043, 
    0.10703258216381073, 0.021938353776931763, 0.09915231168270111, -0.2055802345275879, 
    0.049845319241285324, 0.022933999076485634, 0.009957757778465748, -0.035572994500398636, 
    0.042067304253578186, 0.06559746712446213, -0.0004470741259865463, 0.007940531708300114, 
    -0.1199365183711052, -0.05754001811146736, -0.15467162430286407, 0.12041143327951431, 
    -0.05739468336105347, -0.01036947499960661, 0.015984894707798958, 0.05806894227862358, 
    -0.08875839412212372, 0.12991857528686523, -0.07736168801784515, 0.007249108981341124, 
    0.059988271445035934, 0.06608785688877106, 0.03469308838248253, 0.17915822565555573, 
    0.1229308620095253, 0.02600902132689953, 0.005866583902388811, 0.002449967199936509, 
    0.05436631292104721, 0.10873880237340927, -0.032708581537008286, -0.16327379643917084, 
    0.01120341382920742, -0.060226984322071075, 0.04367837309837341, -0.1261330544948578, 
    0.04933705925941467, 0.14450909197330475, -0.03839836269617081, -0.05882943421602249, 
    -0.0007582925027236342, -0.018312828615307808, -0.10560859739780426, -0.048502709716558456, 
    0.005437387153506279, 0.020334560424089432, -0.0019713693764060736, -0.15156090259552002, 
    -0.04046117886900902, 0.04077159985899925, -0.04497256502509117, 0.02031646855175495, 
    0.166362926363945, 0.09310176223516464, -0.17166157066822052, 0.1601843237876892, 
    0.05593167990446091, 0.06595388799905777, -0.06074090674519539, -0.008766929619014263, 
    0.044330112636089325, -0.023094430565834045, 0.020929666236042976, 0.013435084372758865, 
    -0.07148179411888123, 0.12075145542621613, -0.012258474715054035, -0.016407573595643044, 
    0.09587007015943527, -0.045571159571409225, 0.0790935754776001, -0.01287958212196827, 
    0.03157312795519829, -0.20616301894187927, 0.14213253557682037, 0.03466775268316269, 
    0.034136902540922165, 0.020106075331568718, -0.0062223393470048904, 0.02349732629954815, 
    -0.08188632130622864, -0.017690075561404228, 0.014229310676455498, 0.02304016426205635, 
    -0.12673738598823547, 0.02185829170048237, -0.13247539103031158, -0.0354650653898716, 
    -0.031261175870895386, 0.11566731333732605, 0.2226082980632782, -0.21180732548236847, 
    0.12982630729675293, -0.01885039173066616, 0.034615617245435715, 0.18204030394554138, 
    -0.0741444006562233, 0.0904405266046524, -0.0825686976313591, -0.058024872094392776, 
    -0.008535000495612621, -0.08240818977355957, 0.010139582678675652, 0.15819379687309265, 
    0.07872327417135239, 0.015006877481937408, 0.07405650615692139, -0.07285263389348984, 
    -0.05761502683162689, -0.08735597878694534, -0.12960225343704224, -0.09655944257974625, 
    0.05570696294307709, 0.11252762377262115, -0.04322012513875961, 0.06191108748316765, 
    0.17062056064605713, -0.06987505406141281, 0.0434034988284111, 0.07484559714794159, 
    -0.12547239661216736, 0.09722864627838135, -0.0019214957719668746, 0.10020621865987778, 
    -0.057615604251623154, 0.038832493126392365, -0.08895208686590195, 0.08950532972812653, 
    0.027684343978762627, -0.13075479865074158, -0.0719989463686943, 0.08074445277452469, 
    0.03856217488646507, -0.00904078222811222, 0.11616946756839752, -0.1850464642047882, 
    0.006898340303450823, 0.12498419731855392, 0.15338125824928284, 0.027477027848362923, 
    0.06352118402719498, 0.034283243119716644, -0.057253703474998474, -0.12957043945789337, 
    -0.02351074106991291, -0.15063358843326569, -0.003941800445318222, -0.015563528053462505, 
    0.19162292778491974, -0.019556961953639984, 0.12022743374109268, -0.017411841079592705, 
    -0.07692014425992966, -0.04441501945257187, -0.06556893140077591, -0.13277557492256165, 
    -0.03760749101638794, 0.05387972667813301, 0.06532391160726547, 0.16274546086788177, 
    -0.0186898335814476, -0.01019582524895668, -0.04782894253730774, -0.179364413022995, 
    0.10445520281791687, -0.0066908239386975765, 0.09075263887643814, -0.007485235575586557, 
    0.003099784255027771, 0.02108660154044628, -0.13674235343933105, -0.1075453981757164, 
    -0.10224378108978271, 0.03016434796154499, -0.12647201120853424, -0.06087620183825493, 
    -0.019917014986276627, 0.023109454661607742, 0.11652950942516327, 0.14235956966876984, 
    0.07987285405397415, -0.11303116381168365, 0.031553562730550766, 0.04480253905057907, 
    -0.023049497976899147, 0.0338950976729393, 0.0140374219045043, -0.03456893935799599, 
    -0.060442086309194565, 0.12032665312290192, -0.025797177106142044, -0.06039269641041756, 
    0.034857649356126785, -0.1405060887336731, -0.0330173559486866, 0.10548018664121628, 
    -0.10154020041227341, 0.12981675565242767, -0.05855828896164894, 0.023396816104650497, 
    0.17146039009094238, 0.0484713539481163, -0.10505083203315735, -0.015120095573365688, 
  
    };
    static const neuron_t neuron8 = {weights8, 0.08081910014152527};
    neurons[8]=neuron8;

    static const float weights9[] ={
    -0.0654483363032341, 0.07435867190361023, 0.012872880324721336, -0.15053758025169373, 
    0.026831233873963356, 0.02764960192143917, -0.07809837907552719, 0.08861347287893295, 
    0.11653727293014526, -0.005165977403521538, -0.06730680167675018, -0.02172263152897358, 
    -0.010304405353963375, 0.02983274683356285, -0.07584535330533981, 0.11167073249816895, 
    -0.14769437909126282, 0.07627911865711212, -0.09219823032617569, 0.1539171189069748, 
    0.06157458946108818, -0.06193919852375984, 0.0873766541481018, 0.09622270613908768, 
    0.02040480077266693, -0.10455147922039032, 0.020274516195058823, -0.026843741536140442, 
    0.018542319536209106, 0.025047490373253822, 0.04081932082772255, -0.049237459897994995, 
    -0.0022468676324933767, -0.008126032538712025, -0.01992814987897873, -0.05371078848838806, 
    -0.01372026652097702, 0.09171457588672638, 0.03713316097855568, 0.18574535846710205, 
    0.04546652361750603, -0.04865522310137749, 0.02739807404577732, -0.10537438094615936, 
    0.06581398099660873, 0.028598066419363022, -0.03290748596191406, -0.07869908958673477, 
    0.050873737782239914, -0.1212916225194931, -0.05248899757862091, 0.028837163001298904, 
    -0.08943711221218109, 0.002643806394189596, -0.09947303682565689, -0.04301905632019043, 
    0.12514112889766693, 0.08023377507925034, 0.04357784613966942, 0.04548586905002594, 
    -0.040221020579338074, 0.14862680435180664, 0.055231478065252304, -0.04461759701371193, 
    0.051752544939517975, -0.011894992552697659, 0.03564201295375824, 0.13284702599048615, 
    0.08861597627401352, -0.16586750745773315, 0.024176016449928284, -0.046298086643218994, 
    0.10664277523756027, 0.0164400152862072, 0.044487278908491135, 0.12047064304351807, 
    0.023992732167243958, -0.04236404597759247, -0.04775206372141838, 0.01485764142125845, 
    0.1299258917570114, -0.04060116410255432, 0.13394545018672943, -0.06337836384773254, 
    -0.04800998046994209, 0.03903752937912941, 0.04825321212410927, -0.1197793111205101, 
    -0.02194780670106411, -0.02399395778775215, 0.08841248601675034, 0.12955522537231445, 
    0.10190525650978088, -0.1672704815864563, 0.1557791829109192, 0.015083459205925465, 
    -0.05957261100411415, 0.04107862710952759, 0.07677268236875534, 0.03993740677833557, 
    -0.14216585457324982, 0.1345304399728775, 0.0402352474629879, -0.02603834681212902, 
    -0.07778377085924149, -0.1328894942998886, -0.06115270033478737, -0.05005965754389763, 
    -0.00016444854554720223, 0.0665343627333641, 0.025082802399992943, 0.06670691817998886, 
    -0.06549488753080368, -0.036190662533044815, 0.0220914576202631, -0.09334865212440491, 
    -0.0066163730807602406, -0.1101488396525383, -0.039896439760923386, 0.028349682688713074, 
    -0.17970751225948334, -0.017783770337700844, -0.002237158827483654, -0.19526983797550201, 
    0.10910153388977051, -0.15507157146930695, -0.019700661301612854, -0.0034110904671251774, 
    0.05991499125957489, 0.01445357408374548, 0.01006048358976841, -0.050298478454351425, 
    -0.02498604543507099, -0.07698256522417068, -0.002402982674539089, 0.054991692304611206, 
    0.021468235179781914, -0.011226974427700043, -0.1645510494709015, 0.05862753838300705, 
    0.057594068348407745, 0.06861120462417603, 0.03524600341916084, 0.01881588250398636, 
    -0.09301181882619858, -0.05046812444925308, -0.03559224307537079, -0.03947075456380844, 
    0.021660955622792244, 0.020805325359106064, -0.02815389633178711, -0.08950437605381012, 
    0.025576641783118248, 0.1659366637468338, 0.0034851497039198875, -0.10298367589712143, 
    -0.09843461215496063, -0.11259336024522781, -0.036523398011922836, -0.034073591232299805, 
    0.023537559434771538, -0.08233103156089783, 0.06923193484544754, -0.07513909786939621, 
    -0.027431804686784744, 0.06853688508272171, 0.11400116980075836, -0.050809040665626526, 
    -0.005221547558903694, 0.06695809960365295, 0.07658971846103668, -0.07350750267505646, 
    0.08011766523122787, -0.06845783442258835, 0.059564974159002304, 0.05340585112571716, 
    -0.017775341868400574, 0.12819479405879974, -0.044501129537820816, -0.031277287751436234, 
    0.15143102407455444, -0.03526043891906738, 0.013096410781145096, 0.08261221647262573, 
    0.043156661093235016, 0.03883407637476921, -0.08432454615831375, -0.07524526119232178, 
    0.03313321992754936, 0.043302204459905624, -0.03281640261411667, -0.10657727718353271, 
    0.036241862922906876, -0.12746599316596985, -0.016712414100766182, 0.020158875733613968, 
    -0.0400230772793293, 0.012745005078613758, 0.04067595303058624, -0.01758669689297676, 
    -0.2145557403564453, -0.06143121048808098, 0.04852566868066788, 0.010200276039540768, 
    0.037812739610672, 0.11289069056510925, 0.14039094746112823, -0.05171509459614754, 
    0.12671439349651337, 0.012211899273097515, -0.08512948453426361, 0.11774736642837524, 
    0.0798749253153801, -0.04344113543629646, 0.1104954183101654, -0.014417779631912708, 
    -0.024405479431152344, -0.018234282732009888, 0.05059482529759407, -0.03539741784334183, 
    -0.04085641726851463, -0.08399182558059692, 0.0659298449754715, -0.034420568495988846, 
    -0.034579530358314514, -0.1290750950574875, -0.07110467553138733, 0.049465276300907135, 
    0.07304824888706207, 0.08440086990594864, 0.0049932291731238365, 0.02453446574509144, 
    0.07905557751655579, -0.06000113859772682, 0.10456541925668716, 0.14284279942512512, 
    0.06634115427732468, 0.011216913349926472, 0.01269595604389906, -0.0033200134057551622, 
    -0.03700774163007736, 0.1489851325750351, 0.04097895324230194, 0.06811293214559555, 
    0.10657687485218048, 0.0290486179292202, 0.010330341756343842, 0.04424034804105759, 
    0.1604224592447281, 0.10878539830446243, -0.11607152968645096, -0.07390173524618149, 
    -0.09168308228254318, 0.046752601861953735, -0.09421879053115845, 0.014188208617269993, 
    0.03659392148256302, 0.025557685643434525, 0.17458316683769226, -0.02607649378478527, 
    -0.10835704952478409, -0.02210678718984127, 0.11889129132032394, -0.08385393023490906, 
    -0.04097530245780945, 0.08841095119714737, -0.09839583933353424, -0.03882596269249916, 
    -0.069517120718956, -0.004296042490750551, -0.17884817719459534, -0.013227012008428574, 
    -0.08402030915021896, -0.15725716948509216, -0.06361300498247147, -0.02371666580438614, 
    0.06204581633210182, -0.13040611147880554, -0.0014239274896681309, 0.005112750455737114, 
    0.04216109588742256, -0.022390278056263924, 0.03549961745738983, -0.004236990585923195, 
    0.10786467790603638, 0.04275602847337723, -0.01926281861960888, 0.006714035291224718, 
    -0.012966194190084934, 0.015960806980729103, -0.09920723736286163, -0.16651883721351624, 
    -0.043033476918935776, 0.24126049876213074, -0.01952110044658184, 0.10145176202058792, 
    -0.058769334107637405, -0.017529061064124107, -0.0051022302359342575, 0.04997154325246811, 
    -0.06807321310043335, 0.009671797975897789, -0.0044845100492239, -0.08894442021846771, 
    -0.016158776357769966, -0.015893880277872086, -0.08579892665147781, -0.09829149395227432, 
    0.020586632192134857, -0.0888282060623169, -0.1419052630662918, -0.005904592573642731, 
    -0.05766946077346802, 0.15618501603603363, 0.06610165536403656, 0.025835435837507248, 
    0.007097054738551378, -0.06545183062553406, -0.10363825410604477, 0.04220784455537796, 
    -0.018934037536382675, -0.019827602431178093, -0.20948517322540283, -0.02284586802124977, 
    -0.0029577543027698994, -0.07897847145795822, 0.03690529614686966, 0.08446606993675232, 
    0.1658756285905838, 0.049233272671699524, -0.12138225138187408, -0.08933953940868378, 
    -0.04830179363489151, 0.01917252317070961, -0.052249759435653687, 0.058928895741701126, 
    -0.07847201824188232, 0.03437230363488197, 0.07447919249534607, 0.014928742311894894, 
    0.05196419730782509, 0.03127085044980049, 0.0030740960501134396, -0.005825479980558157, 
    -0.041369739919900894, 0.11921024322509766, 0.07253581285476685, 0.00738207483664155, 
    0.000624930951744318, -0.01410888321697712, -0.03522713854908943, 0.047445762902498245, 
    0.15717896819114685, 0.10164625942707062, -0.15678346157073975, -0.09805622696876526, 
    0.06852688640356064, 0.04728860408067703, 0.002717592054978013, -0.0826038047671318, 
    0.03267045319080353, 0.030266957357525826, 0.2152923047542572, 0.04483018070459366, 
    -0.018414800986647606, 0.06865643709897995, -0.047856349498033524, 0.14111295342445374, 
    -0.05330509692430496, -0.20032243430614471, 0.009254911914467812, -0.02637406811118126, 
    0.007520894054323435, -0.0064972639083862305, -0.031784601509571075, 0.007961688563227654, 
    -0.00317551102489233, -0.11827447265386581, 0.06722503155469894, -0.18087852001190186, 
    -0.002534787403419614, -0.09166847169399261, -0.1285320669412613, -0.04988834634423256, 
    -0.0878223404288292, 0.01730293408036232, 0.08570142835378647, 0.05874406173825264, 
    0.09117676317691803, -0.022204700857400894, 0.0028162687085568905, 0.0757482722401619, 
    0.020226579159498215, 0.002450674306601286, 0.041712068021297455, 0.03033597767353058, 
    0.05565846338868141, 0.020540039986371994, -0.06698276102542877, -0.02586965821683407, 
    -0.07800173759460449, 0.03652801364660263, -0.1092197597026825, 0.08867216110229492, 
    0.10143934190273285, 0.11011971533298492, -0.08491598069667816, -0.02239530347287655, 
    0.1286592185497284, 0.07737811654806137, 0.07348281890153885, 0.03421100974082947, 
    0.11237315833568573, 0.010874941945075989, -0.08382309228181839, -0.02034744620323181, 
    0.09949453920125961, -0.02350548654794693, 0.08494248241186142, 0.00020706135546788573, 
    -0.0825120359659195, -0.08151953667402267, -0.03562531992793083, 0.012494048103690147, 
    0.015949856489896774, -0.057017020881175995, 0.1630813479423523, -0.15187765657901764, 
    0.06937916576862335, 0.02809012681245804, -0.004811141639947891, -0.06558021157979965, 
    0.12507888674736023, 0.044575560837984085, 0.06749186664819717, 0.05201728269457817, 
    -0.014429062604904175, -0.044067833572626114, -0.017043305560946465, 0.006014664191752672, 
    0.002546073868870735, 0.07659593969583511, -0.057538993656635284, 0.0935104489326477, 
    -0.032780956476926804, 0.015305458568036556, -0.063868448138237, 0.04333346337080002, 
    0.1348167359828949, 0.0704989805817604, 0.06216273829340935, -0.09236256033182144, 
    -0.043686117976903915, 0.0128850769251585, 0.03695268556475639, 0.020196763798594475, 
    0.1483333259820938, 0.03991202637553215, -0.061138447374105453, 0.06025345250964165, 
    0.16528351604938507, 0.044062286615371704, -0.06705543398857117, 0.0334828644990921, 
    -0.11057091504335403, -0.015526145696640015, -0.17278850078582764, -0.015339067205786705, 
    -0.11586017161607742, -0.09973359107971191, -0.020209502428770065, 0.22084932029247284, 
    0.08106568455696106, -0.06718726456165314, -0.0923486053943634, 0.025440717115998268, 
    -0.0073777372017502785, 0.08342615514993668, 0.011481693014502525, -0.00764328520745039, 
    0.007179643493145704, 0.15879742801189423, -0.05002584308385849, 0.08268631249666214, 
    -0.01364156138151884, -0.04449564591050148, 0.004234594292938709, 0.07664255797863007, 
    -0.005220342893153429, -0.12908214330673218, 0.06160757690668106, -0.03111652098596096, 
    -0.06978048384189606, -0.060015417635440826, -0.16259054839611053, -0.0366058424115181, 
    -0.005838611163198948, 0.01820540614426136, 0.015259949490427971, -0.09036529809236526, 
    0.07208187878131866, 0.02553694322705269, -0.06834210455417633, -0.0012356145307421684, 
    -0.10680412501096725, 0.04799650236964226, 0.030546553432941437, -9.540271275909618e-05, 
    -0.03781159594655037, -0.03290998190641403, -0.03131594508886337, 0.05838727578520775, 
  
    };
    static const neuron_t neuron9 = {weights9, 0.027055202051997185};
    neurons[9]=neuron9;

    dense_layer_t layer= {10, neurons};
    return layer;
}

