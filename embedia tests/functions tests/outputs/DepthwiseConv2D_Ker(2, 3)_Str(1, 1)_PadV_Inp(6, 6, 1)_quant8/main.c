#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/quant8/embedia.h"
#include "../embedia/quant8/quant8.h"
#include "../embedia/debug/embedia_debug.h"

depthwise_conv2d_layer_t init_depthwise_conv2d_8_data(void){

    static const quant8 weights[]={
        0, 89, 202, /* [-0.35934469 -0.05371177  0.34109253] */
        195, 255, 131 /* [0.31614906 0.52587622 0.09280097] */
    };
    static const quant8 biases[]={
        0 /* 0.07896237820386887 */
    };

    depthwise_conv2d_layer_t layer = {weights, biases, 1, { 2, 3 }, 0, {1, 1},{ 0.003471454568937713, 104 },{ 1, 0 } };
        
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

data3d_t input = { 1, 6, 6, (float[]){ 2, 0, 0, 0, 2, 2, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 2, 0, 2, 2, 2, 0, 1, 2,
 2, 2, 2, 2, 1, 0, 0, 1, 0, 0, 2 } };

data3d_t real_output = { 1, 4, 5, (float[]){  0.8209755 ,  1.3299376 ,  1.6959736 ,  1.5885501 ,  0.3622367 ,
  0.26633096,  0.84902376,  0.5087494 ,  1.2980646 ,  2.2359955 ,
  1.3499564 ,  1.1269017 ,  2.5233762 ,  1.804687  ,  1.1225019 ,
  1.478217  ,  0.02783548,  0.46091074,  0.25118357, -0.22045606 } };

depthwise_conv2d_layer_t depthwise_conv2d_8_data;

data3d_t output;


#define ERROR_BOUND 0.05

measures_info_t info; 
  
int main(){

        depthwise_conv2d_8_data = init_depthwise_conv2d_8_data();

    
    //************************ LAYER  0 ***********************//
    depthwise_conv2d_layer(depthwise_conv2d_8_data, input, &output);

// Debug function for layer depthwise_conv2d_8
print_data3d_t("depthwise_conv2d_8", output);
    //************************ LAYER  1 ***********************//
    

// Debug function for layer depthwise_conv2d_81
print_data3d_t("depthwise_conv2d_81", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}