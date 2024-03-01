#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/float/embedia.h"


conv2d_layer_t init_conv2d_data(void){

        static filter_t filters[2];
        
        static const float weights0[]={
           0.27236253023147583,    0.23517024517059326, 
           0.12666255235671997,    -0.001301586627960205, 
           -0.1098078191280365,    -0.3579414486885071, 
           -0.4982643723487854,    0.5111086368560791, 
           -0.45576173067092896,    -0.17883819341659546, 
           -0.3712605834007263,    -0.3974045217037201 
        };
        static filter_t filter0 = {3, 2, weights0, 0.0}; 
        filters[0]=filter0;
            
        static const float weights1[]={
           0.29962456226348877,    0.006930649280548096, 
           0.05254310369491577,    -0.19263720512390137, 
           0.4421335458755493,    -0.06279444694519043, 
           0.11824190616607666,    -0.4580167233943939, 
           0.3470233082771301,    -0.06291589140892029, 
           0.5235278606414795,    0.4925335645675659 
        };
        static filter_t filter1 = {3, 2, weights1, 0.0}; 
        filters[1]=filter1;
            
        conv2d_layer_t layer = {2, filters, 0, {2, 1} };
        return layer;
}
        

float measure_error(data3d_t o_real, data3d_t o_pred, float err){
    int x, y, c, pr, pp, match;
    for (match=0,c=0, pp=0; c<o_real.channels; c++){
        for (y=0; y<o_real.height; y++){
            for (x=0; x<o_real.width; x++, pp++){
               pr = (y*o_real.width+x)*o_real.channels + c;
               printf("%f   %f\n", o_real.data[pp], o_pred.data[pr]);
               if (fabs(o_real.data[pr]-o_pred.data[pp]) <= err)
                    match++;
            }
        }
    }
    return 100.0*match/(o_real.channels*o_real.height*o_real.width);
}
data3d_t input = { 3, 5, 5, (float[]){ 1, 3, 1, 3, 0, 1, 1, 2, 3, 1, 2, 1, 2, 1, 4, 2, 0, 4, 3, 3, 0, 4, 2, 4, 4,
 1, 4, 1, 2, 4, 1, 2, 2, 4, 2, 3, 3, 3, 3, 1, 2, 0, 1, 0, 3, 4, 1, 3, 3, 4,
 4, 0, 2, 0, 4, 1, 2, 0, 1, 4, 3, 0, 1, 0, 3, 2, 1, 4, 0, 1, 2, 4, 3, 0, 3 } };

data3d_t real_output = { 2, 4, 2, (float[]){  0.31155372,  2.1781604 , -2.343314  ,  1.7343981 , -5.032726  ,
  5.3257236 , -3.0513277 ,  4.5143075 , -1.8249775 ,  3.050478  ,
 -3.478174  ,  4.576171  , -2.6457694 ,  3.1406713 , -2.061242  ,
  3.8896623  } };

conv2d_layer_t conv2d_data;

data3d_t output;

# define ERROR_BOUND 1e-05

int main(){

    conv2d_data = init_conv2d_data();
    
    //<<<<<<<<<<<<<<<<<<<<< INTERNAL LAYER >>>>>>>>>>>>>>>>>>>>>//
    channel_adapt_layer(input, &output);

    //************************ LAYER  0 ***********************//
    input = output;
    conv2d_layer(conv2d_data, input, &output);

    printf("Test result: %6.3f %%\n", measure_error(real_output, output, ERROR_BOUND));

    return 0;
}