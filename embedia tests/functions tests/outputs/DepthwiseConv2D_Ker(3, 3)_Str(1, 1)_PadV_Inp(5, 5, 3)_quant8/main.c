#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/quant8/embedia.h"
#include "../embedia/quant8/quant8.h"
#include "../embedia/debug/embedia_debug.h"

depthwise_conv2d_layer_t init_depthwise_conv2d_25_data(void){

    static const quant8 weights[]={
        3, 173, 44, /* [-0.37845272  0.14387363 -0.25054109] */
        131, 0, 243, /* [ 0.01558638 -0.38763005  0.36006182] */
        143, 255, 125, /* [ 0.0518862   0.39493984 -0.00274539] */

        22, 94, 109, /* [-0.31911704 -0.09865752 -0.05239993] */
        76, 188, 250, /* [-0.15333918  0.19085675  0.37974679] */
        211, 125, 122, /* [ 0.26041251 -0.00377831 -0.01153436] */

        60, 153, 53, /* [-0.20346579  0.08365786 -0.22302602] */
        250, 97, 76, /* [ 0.38037789 -0.08947095 -0.15479723] */
        142, 153, 218 /* [0.05054134 0.08235225 0.28200495] */
    };
    static const quant8 biases[]={
        49, /* 0.010618316009640694 */
        146, /* 0.03144432231783867 */
        48 /* 0.06550581008195877 */
    };

    depthwise_conv2d_layer_t layer = {weights, biases, 3, { 3, 3 }, 0, {1, 1},{ 0.0030689015107996324, 126 },{ 0.000215245082097895, 0 } };
        
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

data3d_t input = { 3, 5, 5, (float[]){ 0, 0, 0, 0, 2, 1, 2, 2, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 2, 2, 1, 1, 2,
 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 1, 1, 2, 1, 2, 0, 0, 1, 0, 2, 1, 1, 0, 0,
 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 2, 2, 0, 2, 2, 2, 0, 1, 0, 1, 2, 1 } };

data3d_t real_output = { 3, 3, 3, (float[]){  0.22691439,  0.73588955,  0.7014908 ,  0.3848825 , -0.30661407,
  0.8827698 , -1.150216  , -0.3279507 ,  0.18870336, -0.02662964,
  0.45940623, -0.6316653 , -0.23028482,  0.0391639 ,  0.6060509 ,
 -0.8305835 ,  0.22179092,  0.76412696,  0.5756389 , -0.36228222,
  0.8322399 ,  0.79760295,  0.01060395, -0.16420278, -0.12293376,
 -0.04340966, -0.20260632 } };

depthwise_conv2d_layer_t depthwise_conv2d_25_data;

data3d_t output;


#define ERROR_BOUND 0.05

measures_info_t info; 
  
int main(){

        depthwise_conv2d_25_data = init_depthwise_conv2d_25_data();

    
    //<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//
    channel_adapt_layer(input, &output);

// Debug function for layer channels_adapter
print_data3d_t("channels_adapter", output);
    //************************ LAYER  0 ***********************//
    input = output;
    depthwise_conv2d_layer(depthwise_conv2d_25_data, input, &output);

// Debug function for layer depthwise_conv2d_25
print_data3d_t("depthwise_conv2d_25", output);
    //************************ LAYER  1 ***********************//
    

// Debug function for layer depthwise_conv2d_251
print_data3d_t("depthwise_conv2d_251", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}