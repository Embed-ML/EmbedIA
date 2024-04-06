#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/quant8/embedia.h"
#include "../embedia/quant8/quant8.h"
#include "../embedia/debug/embedia_debug.h"

depthwise_conv2d_layer_t init_depthwise_conv2d_20_data(void){

    static const quant8 weights[]={
        218, 255, /* [0.32178944 0.48125213] */
        234, 251, /* [0.38951427 0.46553653] */
        0, 178 /* [-0.61435741  0.15014863] */
    };
    static const quant8 biases[]={
        0 /* -0.05482698604464531 */
    };

    depthwise_conv2d_layer_t layer = {weights, biases, 1, { 3, 2 }, 0, {2, 2},{ 0.0042965080223831475, 143 },{ 1, 0 } };
        
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

data3d_t input = { 1, 6, 6, (float[]){ 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 1, 1, 0, 2, 1, 1, 2, 1, 2,
 0, 0, 1, 0, 2, 1, 0, 2, 2, 0, 1 } };

data3d_t real_output = { 1, 3, 2, (float[]){ 2.3329403, 1.10157  , 2.4076347, 1.2536143, 1.5939515, 2.293077  } };

depthwise_conv2d_layer_t depthwise_conv2d_20_data;

data3d_t output;


#define ERROR_BOUND 0.05

measures_info_t info; 
  
int main(){

        depthwise_conv2d_20_data = init_depthwise_conv2d_20_data();

    
    //************************ LAYER  0 ***********************//
    depthwise_conv2d_layer(depthwise_conv2d_20_data, input, &output);

// Debug function for layer depthwise_conv2d_20
print_data3d_t("depthwise_conv2d_20", output);
    //************************ LAYER  1 ***********************//
    

// Debug function for layer depthwise_conv2d_201
print_data3d_t("depthwise_conv2d_201", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}