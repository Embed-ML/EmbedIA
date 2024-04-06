#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/fixed16/embedia.h"
#include "../embedia/fixed16/fixed.h"
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

data2d_t input = { 1, 10, (fixed[]){ 512, 256, 512, 512, 512, 256,   0, 512,   0,   0 } };

data2d_t real_output = { 1, 10, (fixed[]){ 247, 195, 247, 247, 247, 195,   0, 247,   0,   0 } };


data2d_t output;


#define ERROR_BOUND 0.025

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