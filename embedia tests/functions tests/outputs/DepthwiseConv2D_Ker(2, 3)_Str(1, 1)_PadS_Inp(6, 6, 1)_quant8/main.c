#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/quant8/embedia.h"
#include "../embedia/quant8/quant8.h"
#include "../embedia/debug/embedia_debug.h"

depthwise_conv2d_layer_t init_depthwise_conv2d_10_data(void){

    static const quant8 weights[]={
        207, 255, 64, /* [ 0.23322988  0.42088026 -0.33705863] */
        0, 211, 102 /* [-0.59627402  0.2478739  -0.18835092] */
    };
    static const quant8 biases[]={
        0 /* -0.09022430330514908 */
    };

    depthwise_conv2d_layer_t layer = {weights, biases, 1, { 2, 3 }, 1, {1, 1},{ 0.003988840299494126, 149 },{ 1, 0 } };
        
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

data3d_t input = { 1, 6, 6, (float[]){ 2, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 0, 0, 0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2,
 1, 0, 2, 0, 1, 0, 0, 0, 2, 0, 0 } };

data3d_t real_output = { 1, 6, 6, (float[]){  1.247284  , -0.8163126 , -0.0902243 , -0.8039848 ,  0.11264332,
 -0.1329144 ,  0.7515362 ,  0.37623546, -0.2785752 , -0.8931695 ,
 -0.02310727,  0.02544793, -0.21905223, -0.5674524 , -1.500785  ,
 -1.4169633 , -0.08873598, -0.32056478, -0.03606442, -0.63402534,
 -0.42909715,  1.0396266 , -0.8370203 ,  1.4658699 ,  0.41447756,
  0.7971157 , -0.90781355,  1.247284  , -1.1533712 ,  0.33065596,
 -0.0902243 , -0.0902243 , -0.7643416 ,  0.7515362 ,  0.37623546,
 -0.0902243  } };

depthwise_conv2d_layer_t depthwise_conv2d_10_data;

data3d_t output;


#define ERROR_BOUND 0.05

measures_info_t info; 
  
int main(){

        depthwise_conv2d_10_data = init_depthwise_conv2d_10_data();

    
    //************************ LAYER  0 ***********************//
    depthwise_conv2d_layer(depthwise_conv2d_10_data, input, &output);

// Debug function for layer depthwise_conv2d_10
print_data3d_t("depthwise_conv2d_10", output);
    //************************ LAYER  1 ***********************//
    

// Debug function for layer depthwise_conv2d_101
print_data3d_t("depthwise_conv2d_101", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}