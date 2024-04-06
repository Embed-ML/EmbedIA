#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/fixed8/embedia.h"
#include "../embedia/fixed8/fixed.h"
#include "../embedia/debug/embedia_debug.h"


separable_conv2d_layer_t init_separable_conv2d_63_data(void){

        
        static fixed depth_weights[]={
            -2, -1, -1, /* [-0.19189249 -0.16422336 -0.15076917] */
            2, 0, -1, /* [ 0.26319587 -0.02771586 -0.16521879] */
            3, 0, 2, /* [0.37281543 0.01555252 0.21563762] */
            -3, 1, -3, /* [-0.35651326  0.14771533 -0.35337648] */
            3, 2, 3, /* [0.39376152 0.22723222 0.31558859] */
            -1, -3, -3, /* [-0.06301403 -0.36207521 -0.37448391] */
            3, 3, 3, /* [0.37633514 0.33678406 0.33047259] */
            2, -3, -3, /* [ 0.22471273 -0.39294225 -0.3592748 ] */
            0, 3, -1 /* [ 0.02812913  0.35336566 -0.15194556] */

        };
        // static filter_t depth_filter = {3, {3, 3}, depth_weights };
        static filter_t depth_filter = { depth_weights };

        static filter_t point_filters[2];
        
        static fixed point_weights0[]={-6, -5, -8,  /* [-0.99134755] */
        };
        static filter_t point_filter0 = {point_weights0, 0 /* 0.04370945319533348 */};
        point_filters[0] = point_filter0;
        
        static fixed point_weights1[]={8, -2, 6,  /* [0.80948126] */
        };
        static filter_t point_filter1 = {point_weights1, 0 /* 0.004345504101365805 */};
        point_filters[1] = point_filter1;
        
        separable_conv2d_layer_t layer = {2, point_filters, 3, {1, 1}, depth_filter, 3, {3, 3}, 1, {2, 2} };
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

data3d_t input = { 3, 5, 5, (fixed[]){  8,  8, 16, 16,  0,  8,  8,  8, 16, 16,  0, 16,  8,  0,  8, 16,  0, 16,
 16,  8, 16,  8,  8,  8,  8,  0, 16,  8,  0, 16, 16, 16,  0, 16,  8,  8,
  0,  8,  0,  0,  8, 16, 16,  8,  0, 16, 16, 16, 16, 16,  8,  8, 16,  0,
 16, 16,  8,  0,  0, 16, 16,  8, 16,  8,  8,  8,  0,  8, 16, 16,  8, 16,
  0, 16,  0 } };

data3d_t real_output = { 2, 3, 3, (fixed[]){   6,  -4,   4,   2, -11,  13,  -3,   7, -10,  17, -25,  19,   6,  -8,  11,
 -11, -13,  11 } };

separable_conv2d_layer_t separable_conv2d_63_data;

data3d_t output;


#define ERROR_BOUND 0.5

measures_info_t info; 
  
int main(){

        separable_conv2d_63_data = init_separable_conv2d_63_data();

    
    //<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//
    channel_adapt_layer(input, &output);

// Debug function for layer channels_adapter
print_data3d_t("channels_adapter", output);
    //************************ LAYER  0 ***********************//
    input = output;
    separable_conv2d_layer(separable_conv2d_63_data, input, &output);

// Debug function for layer separable_conv2d_63
print_data3d_t("separable_conv2d_63", output);
    //************************ LAYER  1 ***********************//
    

// Debug function for layer separable_conv2d_631
print_data3d_t("separable_conv2d_631", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}