#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/fixed16/embedia.h"
#include "../embedia/fixed16/fixed.h"
#include "../embedia/debug/embedia_debug.h"


separable_conv2d_layer_t init_separable_conv2_d_data(void){

        
        static fixed depth_weights[]={
            -21, -48, /* [-0.08345693 -0.18764278] */
            -5, -63, /* [-0.02017152 -0.24641854] */
            85, -136, /* [ 0.33305115 -0.53294909] */
            -13, -112, /* [-0.05002594 -0.43710282] */
            -105, -62, /* [-0.40908843 -0.24280429] */
            148, 46 /* [0.58004111 0.17895162] */

        };
        // static filter_t depth_filter = {3, {2, 2}, depth_weights };
        static filter_t depth_filter = { depth_weights };

        static filter_t point_filters[2];
        
        static fixed point_weights0[]={187, -62, -17,  /* [-0.06701791] */
        };
        static filter_t point_filter0 = {point_weights0, -10 /* -0.04063640162348747 */};
        point_filters[0] = point_filter0;
        
        static fixed point_weights1[]={107, -240, -68,  /* [-0.26735491] */
        };
        static filter_t point_filter1 = {point_weights1, 7 /* 0.029214564710855484 */};
        point_filters[1] = point_filter1;
        
        separable_conv2d_layer_t layer = {2, point_filters, 3, {1, 1}, depth_filter, 3, {2, 2}, 1, {1, 1} };
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
               printf("%f   %f\n", FX2FL(o_real.data[pp]), FX2FL(o_pred.data[pr]));
               error = fabs(FX2FL(o_real.data[pr])-FX2FL(o_pred.data[pp]));
               info->acc_error += error;
               if (error <= bnd_error)
                    info->match++;
            }
        }
    }
}

data3d_t input = { 3, 5, 5, (fixed[]){   0, 512, 512, 512, 256, 512, 256, 256, 256, 256,   0, 512,   0, 512, 256,
   0,   0, 256, 256, 512,   0, 256, 512, 256,   0, 512,   0, 256, 256, 512,
 512,   0, 256, 256, 256, 512,   0, 256, 256,   0,   0, 512, 512, 256, 256,
   0, 256,   0, 256,   0, 512, 512, 256,   0, 512, 512,   0, 512, 512, 256,
 512, 256, 256,   0, 512, 256, 256, 256,   0, 512, 512, 512,   0, 256, 256 } };

data3d_t real_output = { 2, 5, 5, (fixed[]){  -68,  168,  -38,  283,  -20,  152,   40,  405,  -65, -194,  -15,  281,
  -29,  114,   -7,   70, -134,  -51,  -47,  -57,  -77,  120,  -84,   90,
  -58,  154,  -71,  332,  -67,  -82,  -21,  106,  -72,  205, -111,  268,
  -80,   95,  -83, -170,   15,  210,  -47,  -17,  -42,  168,  -32,   30,
  -24,  -44 } };

separable_conv2d_layer_t separable_conv2_d_data;

data3d_t output;


#define ERROR_BOUND 0.025

measures_info_t info; 
  
int main(){

        separable_conv2_d_data = init_separable_conv2_d_data();

    
    //<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//
    channel_adapt_layer(input, &output);

// Debug function for layer channels_adapter
print_data3d_t("channels_adapter", output);
    //************************ LAYER  0 ***********************//
    input = output;
    separable_conv2d_layer(separable_conv2_d_data, input, &output);

// Debug function for layer separable_conv2_d
print_data3d_t("separable_conv2_d", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}