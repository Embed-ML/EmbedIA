#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/fixed32/embedia.h"
#include "../embedia/fixed32/fixed.h"
#include "../embedia/debug/embedia_debug.h"

dense_layer_t init_dense_2_data(void){

    static neuron_t neurons[10];

    /* [-0.33099228  0.31067067  0.33671921 -0.16315621  0.4674796   0.49909818
  0.28363717 -0.07607123 -0.23682088 -0.0430029 ] 0.0*/
    static const fixed weights0[] ={
    -43384, 40720, 44134, -21385, 61273, 65418, 37177, -9971, -31041, -5636
    };
    
    static const neuron_t neuron0 = {weights0, 0  };
    neurons[0]=neuron0;

    /* [-0.11082089  0.03124863 -0.46378607  0.22500181 -0.35611427 -0.0663673
 -0.07871079  0.2077924   0.38137442 -0.13164279] 0.0*/
    static const fixed weights1[] ={
    -14526, 4096, -60789, 29491, -46677, -8699, -10317, 27236, 49988, -17255
    };
    
    static const neuron_t neuron1 = {weights1, 0  };
    neurons[1]=neuron1;

    /* [-0.05128148  0.16662228 -0.23600471 -0.4259005   0.10838348  0.22218227
 -0.00260574  0.17292583 -0.0773215   0.18073511] 0.0*/
    static const fixed weights2[] ={
    -6722, 21840, -30934, -55824, 14206, 29122, -342, 22666, -10135, 23689
    };
    
    static const neuron_t neuron2 = {weights2, 0  };
    neurons[2]=neuron2;

    /* [-0.41105154 -0.32729656 -0.44713202  0.34362608 -0.4896161  -0.26883054
 -0.52095675  0.42101485 -0.08238474 -0.03364944] 0.0*/
    static const fixed weights3[] ={
    -53877, -42899, -58606, 45040, -64175, -35236, -68283, 55183, -10798, -4410
    };
    
    static const neuron_t neuron3 = {weights3, 0  };
    neurons[3]=neuron3;

    /* [ 0.38658118  0.05543077 -0.40747985 -0.20434761 -0.40893918  0.3213492
 -0.3029492   0.1873048   0.01084608 -0.03555328] 0.0*/
    static const fixed weights4[] ={
    50670, 7265, -53409, -26784, -53600, 42120, -39708, 24550, 1422, -4660
    };
    
    static const neuron_t neuron4 = {weights4, 0  };
    neurons[4]=neuron4;

    /* [ 0.31272405 -0.49896902  0.4216807  -0.48942608 -0.42777652  0.1516537
  0.15430212  0.26617193  0.07286078 -0.39797363] 0.0*/
    static const fixed weights5[] ={
    40989, -65401, 55271, -64150, -56070, 19878, 20225, 34888, 9550, -52163
    };
    
    static const neuron_t neuron5 = {weights5, 0  };
    neurons[5]=neuron5;

    /* [ 0.39081812  0.28402096 -0.50731456  0.0859862  -0.34968793  0.04691684
 -0.42188478 -0.05198756 -0.43279433 -0.48334712] 0.0*/
    static const fixed weights6[] ={
    51225, 37227, -66495, 11270, -45834, 6149, -55297, -6814, -56727, -63353
    };
    
    static const neuron_t neuron6 = {weights6, 0  };
    neurons[6]=neuron6;

    /* [ 0.30396503 -0.30337557  0.49672496  0.02618444 -0.4500215   0.31415957
  0.15170687  0.36075926 -0.5153751   0.37026525] 0.0*/
    static const fixed weights7[] ={
    39841, -39764, 65107, 3432, -58985, 41178, 19885, 47285, -67551, 48531
    };
    
    static const neuron_t neuron7 = {weights7, 0  };
    neurons[7]=neuron7;

    /* [-0.32114732 -0.26955557 -0.4509244  -0.34185165  0.35337925 -0.5009176
 -0.1671356   0.23642874 -0.41233522 -0.32824147] 0.0*/
    static const fixed weights8[] ={
    -42093, -35331, -59104, -44807, 46318, -65656, -21907, 30989, -54046, -43023
    };
    
    static const neuron_t neuron8 = {weights8, 0  };
    neurons[8]=neuron8;

    /* [ 0.37558877 -0.4840361  -0.2978857   0.02451265  0.24979407 -0.4709079
 -0.0188418   0.05425406  0.21260947  0.49883664] 0.0*/
    static const fixed weights9[] ={
    49229, -63444, -39044, 3213, 32741, -61723, -2470, 7111, 27867, 65384
    };
    
    static const neuron_t neuron9 = {weights9, 0  };
    neurons[9]=neuron9;

    dense_layer_t layer= { 10, neurons};
    return layer;
}


typedef struct{
    float acc_error;
    int match;
    int total;
} measures_info_t;

float measure_error(data1d_t o_real, data1d_t o_pred, float bnd_error, measures_info_t* info){
    int i;
    float error;

    info->total=o_real.length;
    info->match=0;
    info->acc_error=0;

    for (i=0; i<o_real.length; i++){
        printf("%f   %f\n", FX2FL(o_real.data[i]), FX2FL(o_pred.data[i]));
        error = fabs(FX2FL(o_real.data[i])-FX2FL(o_pred.data[i]));
        info->acc_error += error;
        if (error <= bnd_error)
            info->match++;
    }
}

data1d_t input = { 10, (fixed[]){      0,      0,      0, 262144,      0,      0, 131072,      0, 262144,
 262144 } };

data1d_t real_output = { 10, (fixed[]){  -70609,   91981,  -74716,   -8609,  -84109, -118028, -127061,  -11263,
 -128623,  117480 } };

dense_layer_t dense_2_data;

data1d_t output;


#define ERROR_BOUND 0.0005

measures_info_t info; 
  
int main(){

        dense_2_data = init_dense_2_data();

    
    //************************ LAYER  0 ***********************//
    dense_layer(dense_2_data, input, &output);


// Debug function for layer dense_2
print_data1d_t("dense_2", output);
    //************************ LAYER  1 ***********************//
    tanh_activation(output.data, 10);

// Debug function for layer dense_21
print_data1d_t("dense_21", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}