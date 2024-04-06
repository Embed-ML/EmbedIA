#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/fixed8/embedia.h"
#include "../embedia/fixed8/fixed.h"
#include "../embedia/debug/embedia_debug.h"


conv2d_layer_t init_conv2d_53_data(void){

        static filter_t filters[2];
        
        static const fixed weights0[]={
           -1,    0,    3, /* [-0.0786626   0.02918449  0.34734857] */
           -1,    0,    0, /* [-0.15743658  0.04219744 -0.02274534] */
           2,    -3,    -3, /* [ 0.2609098  -0.35124728 -0.35891005] */
           -3,    2,    1, /* [-0.32349077  0.29122418  0.12964919] */
           3,    1,    0, /* [ 0.35946167  0.12855476 -0.0019246 ] */
           0,    0,    1, /* [0.03862196 0.06180373 0.16825122] */
           1,    1,    -2, /* [ 0.0786258   0.0675675  -0.23258361] */
           0,    3,    -1, /* [ 0.03401509  0.32264763 -0.07541281] */
           -1,    -1,    -1 /* [-0.14216378 -0.13873214 -0.10687676] */
        };
        static filter_t filter0 = { weights0, 0};  //-0.028857922181487083
        filters[0]=filter0;
            
        static const fixed weights1[]={
           2,    0,    -2, /* [ 0.18925178 -0.01820847 -0.30360419] */
           1,    -1,    -3, /* [ 0.09244227 -0.12801214 -0.3565774 ] */
           0,    0,    1, /* [0.05945444 0.03699014 0.17234504] */
           2,    1,    -1, /* [ 0.29419005  0.16023773 -0.11731365] */
           1,    2,    -2, /* [ 0.09104812  0.27697855 -0.20327896] */
           -1,    -1,    1, /* [-0.07117954 -0.06303847  0.07334101] */
           0,    -2,    -1, /* [ 0.01356983 -0.23540142 -0.11461607] */
           -1,    -1,    0, /* [-0.07911339 -0.08682832  0.04152194] */
           -3,    3,    1 /* [-0.31938434  0.36436886  0.06969893] */
        };
        static filter_t filter1 = { weights1, 0};  //0.04897194728255272
        filters[1]=filter1;
            
        conv2d_layer_t layer = {2, filters, 3, { 3, 3 }, 0, {1, 2} };
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

data3d_t input = { 3, 5, 5, (fixed[]){  0, 16,  0,  8,  0,  8,  0,  0, 16,  8,  8,  8, 16,  8,  0, 16, 16,  0,
  8, 16,  8, 16,  8,  8,  0,  8,  8,  0,  8, 16,  0,  0, 16, 16,  8,  0,
  0,  8,  0,  8,  0,  8,  8, 16,  8,  8,  8, 16,  8,  8,  0,  0,  8,  8,
  8, 16, 16, 16, 16, 16, 16,  8, 16,  0,  0,  8, 16,  8,  0,  0, 16,  8,
  0,  0, 16 } };

data3d_t real_output = { 2, 2, 3, (fixed[]){ -7, -5,  9,  2,  4, -5, -8,  3,  4,  2, 13, -7 } };

conv2d_layer_t conv2d_53_data;

data3d_t output;


#define ERROR_BOUND 0.5

measures_info_t info; 
  
int main(){

        conv2d_53_data = init_conv2d_53_data();

    
    //<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//
    channel_adapt_layer(input, &output);

// Debug function for layer channels_adapter
print_data3d_t("channels_adapter", output);
    //************************ LAYER  0 ***********************//
    input = output;
    conv2d_strides_layer(conv2d_53_data, input, &output);

// Debug function for layer conv2d_53
print_data3d_t("conv2d_53", output);
    //************************ LAYER  1 ***********************//
    

// Debug function for layer conv2d_531
print_data3d_t("conv2d_531", output);

    measure_error(real_output, output, ERROR_BOUND, & info);
    printf("Test result: %7.3f %%\n", 100.0 * info.match / info.total);
    printf(" Acc. error: %7.3f \n", info.acc_error);
    printf(" Elem count: %3d \n", info.total);
    return 0;
}