#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/quant8/embedia.h"
#include "../embedia/quant8/quant8.h"
#include "../embedia/debug/embedia_debug.h"

depthwise_conv2d_layer_t init_depthwise_conv2d_28_data(void){

    static const quant8 weights[]={
        170, 60, 69, /* [ 0.14578205 -0.19547093 -0.16793659] */
        202, 255, 40, /* [ 0.24666691  0.41000789 -0.25910509] */
        126, 0, 194 /* [ 0.00795501 -0.38187781  0.22090387] */
    };
    static const quant8 biases[]={
        0 /* 0.06513813138008118 */
    };

    depthwise_conv2d_layer_t layer = {weights, biases, 1, { 3, 3 }, 0, {2, 2},{ 0.003105434132557289, 123 },{ 1, 0 } };
        
    return layer;
}


typedef struct{
    float acc_error;
    int match;
    int total;
} measures_info_t;

float measure_error(data3d_t o_real, data3d_t o_pred, float bnd_error, measures_info_t* info){
    int x, y, c, pr, pp;
    float error;

    info->total=o_real.channels*o_real.height*o_real.width;
    info->match=0;
    info->acc_error=0;

    for (c=0, pp=0; c<o_real.channels; c++){
        for (y=0; y<o_real.height; y++){
            for (x=0; x<o_real.width; x++, pp++){
               pr = (y*o_real.width+x)*o_real.channels + c;
               printf("%f   %f\n", o_real.data[pp], o_pred.data[pr]);
               error = fabs(o_real.data[pr]-o_pred.data[pp]);
               info->acc_error += error;
               if (error <= bnd_error)
                    info->match++;
            }
        }
    }
}

data3d_t input = { 1, 6, 6, (float[]){ 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 2, 1,
 1, 0, 1, 2, 2, 1, 1, 1, 1, 2, 1 } };

data3d_t real_output = { 1, 2, 2, (float[]){ 0.3908644 , 0.29864538, 0.18454915, 0.58167636 } };

depthwise_conv2d_layer_t depthwise_conv2d_28_data;

data3d_t output;


#define ERROR_BOUND 0.05

measures_info_t info; 
  
int main(){

        depthwise_conv2d_28_data = init_depthwise_conv2d_28_data();

    
    //************************ LAYER  0 ***********************//
    depthwise_conv2d_layer(depthwise_conv2d_28_data, input, &output);

// Debug function for layer depthwise_conv2d_28
print_data3d_t("depthwise_conv2d_28", output);
    //************************ LAYER  1 ***********************//
    

// Debug function for layer depthwise_conv2d_281
print_data3d_t("depthwise_conv2d_281", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}