#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/quant8/embedia.h"
#include "../embedia/quant8/quant8.h"
#include "../embedia/debug/embedia_debug.h"

depthwise_conv2d_layer_t init_depthwise_conv2_d_data(void){

    static const quant8 weights[]={
        127, 206, 57, /* [-0.26659754 -0.63811105 -0.41380483] */
        2, 205, 69 /* [-0.52954078 -0.10500264 -0.38955507] */
    };
    static const quant8 biases[]={
        0 /* 0.040246523916721344 */
    };

    depthwise_conv2d_layer_t layer = {weights, biases, 1, { 2, 3 }, 1, {1, 1},{ 0.002090621228311576, 255 },{ 1, 0 } };

    return layer;
}


typedef struct{
    float acc_error;
    int match;
    int total;
} measures_info_t;

void measure_error(data3d_t o_real, data3d_t o_pred, float bnd_error, measures_info_t* info){
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

data3d_t input = { 1, 6, 6, (float[]){ 1, 2, 0, 1, 0, 0, 1, 2, 1, 1, 2, 2, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 2, 2,
 0, 2, 0, 0, 2, 1, 2, 0, 0, 0, 1 } };

data3d_t real_output = { 1, 6, 6, (float[]){ -2.309587  , -2.6316743 , -2.4603927 , -2.0115182 , -1.7450073 ,
 -1.2288404 , -2.2045844 , -2.1263833 , -2.603946  , -2.081627  ,
 -2.4351854 , -2.2987115 , -0.7873631 , -1.6255307 , -0.5979512 ,
 -1.2926542 , -1.4819773 , -0.965897  , -0.16975877, -2.21175   ,
 -0.8078698 , -1.6992375 , -2.2045844 , -1.7125784 , -2.1200883 ,
 -2.0601044 , -2.2950573 , -0.49294856, -1.1769183 , -1.3409783 ,
 -1.4254742 , -1.5025731 , -0.49294856,  0.04024652, -0.3735583 ,
 -0.5978645  } };

depthwise_conv2d_layer_t depthwise_conv2_d_data;

data3d_t output;


#define ERROR_BOUND 0.05

measures_info_t info;

int main(){

        depthwise_conv2_d_data = init_depthwise_conv2_d_data();


    //************************ LAYER  0 ***********************//
    depthwise_conv2d_layer(depthwise_conv2_d_data, input, &output);

// Debug function for layer depthwise_conv2_d
print_data3d_t("depthwise_conv2_d", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}
