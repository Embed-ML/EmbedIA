#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/float/embedia.h"
#include "../embedia/debug/embedia_debug.h"

normalization_layer_t init_standard_normalization_data(void){
    /*[0.4 1.2 1.  1.  1. ]*/
    static const float sub_val[] ={
    0.4, 1.2, 1.0, 1.0, 1.0
    };
    /*[1.25       2.5        1.58113883 1.11803399 1.11803399]*/
    static const float inv_div_val[] ={
    1.2499999999999998, 2.5, 1.5811388300841895, 1.118033988749895, 1.118033988749895,

    };

    static const normalization_layer_t norm = { sub_val, inv_div_val  };
    return norm;
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
        printf("%f   %f\n", o_real.data[i], o_pred.data[i]);
        error = fabs(o_real.data[i]-o_pred.data[i]);
        info->acc_error += error;
        if (error <= bnd_error)
            info->match++;
    }
}

data1d_t input = { 5, (float[]){ 0, 1, 1, 2, 0, 0, 1, 2, 2, 1, 2, 1, 1, 0, 2, 0, 2, 1, 1, 2, 0, 1, 0, 0, 0 } };

data1d_t real_output = { 5, (float[]){ -0.5       , -0.5       ,  0.        ,  1.11803399, -1.11803399,
 -0.5       , -0.5       ,  1.58113883,  1.11803399,  0.        ,
  2.        , -0.5       ,  0.        , -1.11803399,  1.11803399,
 -0.5       ,  2.        ,  0.        ,  0.        ,  1.11803399,
 -0.5       , -0.5       , -1.58113883, -1.11803399, -1.11803399 } };

normalization_layer_t standard_normalization_data;

data1d_t output;


#define ERROR_BOUND 1e-05

measures_info_t info;

int main(){

        standard_normalization_data = init_standard_normalization_data();


    //************************ LAYER  0 ***********************//


// Debug function for layer dummy_layer

    //************************ LAYER  1 ***********************//
    //  input = output;
    standard_norm_layer(standard_normalization_data, input, &output);

// Debug function for layer standard_normalization
print_data1d_t("standard_normalization", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}
