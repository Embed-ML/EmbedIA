#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/quant8/embedia.h"
#include "../embedia/quant8/quant8.h"
#include "../embedia/debug/embedia_debug.h"

depthwise_conv2d_layer_t init_depthwise_conv2d_2_data(void){

    static const quant8 weights[]={
        74, 255, /* [-0.30686545  0.79400057] */
        0, 62 /* [-0.75644231 -0.37838954] */
    };
    static const quant8 biases[]={
        0 /* 0.133894145488739 */
    };

    depthwise_conv2d_layer_t layer = {weights, biases, 1, { 2, 2 }, 1, {1, 1},{ 0.006080168135025922, 124 },{ 1, 0 } };
        
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

data3d_t input = { 1, 6, 6, (float[]){ 2, 2, 2, 1, 2, 1, 2, 2, 1, 0, 1, 1, 1, 1, 0, 1, 2, 1, 1, 0, 0, 1, 1, 0, 2,
 2, 0, 2, 2, 1, 1, 1, 1, 1, 2, 0 } };

data3d_t real_output = { 1, 6, 6, (float[]){ -1.1614993 , -0.7831098 , -0.4422785 ,  1.0366403 , -0.82066804,
 -0.9294136 , -0.02666748, -0.4422785 , -0.55136085, -0.5853267 ,
 -1.2702448 , -0.9294136 , -0.13541305, -0.17297131,  0.5495052 ,
  0.28019798, -0.4422785 , -0.17297131, -2.4426348 , -1.3789904 ,
  0.17111564, -1.6486344 , -2.0642455 , -0.62254816, -0.02666748,
 -1.6146686 ,  0.58706343, -0.405057  , -1.1987207 , -0.17297131,
  0.62102926,  0.62102926,  0.62102926,  1.4150298 , -0.47983676,
  0.13389415 } };

depthwise_conv2d_layer_t depthwise_conv2d_2_data;

data3d_t output;


#define ERROR_BOUND 0.05

measures_info_t info; 
  
int main(){

        depthwise_conv2d_2_data = init_depthwise_conv2d_2_data();

    
    //************************ LAYER  0 ***********************//
    depthwise_conv2d_layer(depthwise_conv2d_2_data, input, &output);

// Debug function for layer depthwise_conv2d_2
print_data3d_t("depthwise_conv2d_2", output);
    //************************ LAYER  1 ***********************//
    

// Debug function for layer depthwise_conv2d_21
print_data3d_t("depthwise_conv2d_21", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}