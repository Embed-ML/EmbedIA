#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/fixed32/embedia.h"
#include "../embedia/fixed32/fixed.h"
#include "../embedia/debug/embedia_debug.h"


typedef struct{
    float acc_error;
    int match;
    int total;
} measures_info_t;

float measure_error(data2d_t o_real, data2d_t o_pred, float bnd_error, measures_info_t* info){
    int i;
    float error;

    info->total=o_real.width*o_real.height;
    info->match=0;
    info->acc_error=0;

    for (i=0; i<o_real.width*o_real.height; i++){
        printf("%f   %f\n", FX2FL(o_real.data[i]), FX2FL(o_pred.data[i]));
        error = fabs(FX2FL(o_real.data[i])-FX2FL(o_pred.data[i]));
        info->acc_error += error;
        if (error <= bnd_error)
            info->match++;
    }
}

data2d_t input = { 1, 10, (fixed[]){ 262144, 131072, 262144, 262144, 262144, 131072,      0, 262144,      0,
      0 } };

data2d_t real_output = { 1, 10, (fixed[]){ 126357,  99824, 126357, 126357, 126357,  99824,      0, 126357,      0,
      0 } };


data2d_t output;


#define ERROR_BOUND 0.0005

measures_info_t info; 
  
int main(){

    
    
    //************************ LAYER  0 ***********************//
    output = input;
    tanh_activation(output.data, 10);

// Debug function for layer activation_2
print_data2d_t("activation_2", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}