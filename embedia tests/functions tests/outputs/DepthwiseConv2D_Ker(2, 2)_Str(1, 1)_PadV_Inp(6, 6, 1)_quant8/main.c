#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/quant8/embedia.h"
#include "../embedia/quant8/quant8.h"
#include "../embedia/debug/embedia_debug.h"

depthwise_conv2d_layer_t init_depthwise_conv2d_data(void){

    static const quant8 weights[]={
        255, 238, /* [0.75796109 0.66658932] */
        0, 87 /* [-0.64392889 -0.1633572 ] */
    };
    static const quant8 biases[]={
        0 /* -0.08797811716794968 */
    };

    depthwise_conv2d_layer_t layer = {weights, biases, 1, { 2, 2 }, 0, {1, 1},{ 0.005497607764075784, 117 },{ 1, 0 } };
        
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

data3d_t input = { 1, 6, 6, (float[]){ 1, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 2, 1, 2, 2, 1, 2, 1, 0, 0, 0, 2, 1,
 2, 2, 1, 1, 1, 2, 2, 0, 1, 1, 1 } };

data3d_t real_output = { 1, 5, 5, (float[]){ -0.781232  ,  0.43791443,  1.2872474 ,  0.365929  ,  0.04871453,
  1.767819  , -0.11464267,  0.365929  ,  0.3885895 , -0.0232709 ,
 -0.20601445,  1.4506046 ,  2.0031617 ,  2.7611227 ,  1.767819  ,
  1.1238902 , -0.9445892 , -1.5391932 , -0.8952642 ,  0.43791443,
  0.3885895 ,  1.4732649 ,  1.9311762 ,  0.5292862 ,  0.5292862  } };

depthwise_conv2d_layer_t depthwise_conv2d_data;

data3d_t output;


#define ERROR_BOUND 0.05

measures_info_t info; 
  
int main(){

        depthwise_conv2d_data = init_depthwise_conv2d_data();

    
    //************************ LAYER  0 ***********************//
    depthwise_conv2d_layer(depthwise_conv2d_data, input, &output);

// Debug function for layer depthwise_conv2d
print_data3d_t("depthwise_conv2d", output);
    //************************ LAYER  1 ***********************//
    

// Debug function for layer depthwise_conv2d1
print_data3d_t("depthwise_conv2d1", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}