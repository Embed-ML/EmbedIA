#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/float/embedia.h"


            float measure_error(data2d_t o_real, data2d_t o_pred, float err){
                int i, match;
                for (match=0,i=0; i<o_real.width*o_real.height; i++){
                    printf("%f   %f\n", o_real.data[i], o_pred.data[i]);
                    if (fabs(o_real.data[i]-o_pred.data[i]) <= err){
                        match++;
                    }
                }
                return 100.0*match/(o_real.width*o_real.height);
            }

data2d_t input = { 1, 10, (float[]){ 0.54400782, 0.13718516, 0.6180683 , 0.52417516, 0.93476466, 0.58781235,
 0.21715174, 0.37104577, 0.69662287, 0.76240494 } };

data2d_t real_output = { 1, 10, (float[]){ 0.54400784, 0.13718516, 0.6180683 , 0.52417517, 0.9347647 , 0.58781236,
 0.21715173, 0.37104577, 0.69662285, 0.7624049  } };


data2d_t output;

# define ERROR_BOUND 1e-05

int main(){


    //************************ LAYER  0 ***********************//
    output = input;
    relu_activation(output.data, 10);

    //relu_activation(output.data, 10);
    printf("Test result: %6.3f %%\n", measure_error(real_output, output, ERROR_BOUND));

    return 0;
}
