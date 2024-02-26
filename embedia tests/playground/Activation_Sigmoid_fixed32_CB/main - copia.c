#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/fixed32/embedia.h"
#include "../embedia/fixed32/fixed.h"


            float measure_error(data2d_t o_real, data2d_t o_pred, float err){
                int i, match;
                for (match=0,i=0; i<o_real.width*o_real.height; i++){
                    printf("%f   %f\n", o_real.data[i], o_pred.data[i]);
                    if (fabs(FX2FL(o_real.data[i])-FX2FL(o_pred.data[i])) <= err){
                        match++;
                    }
                }
                return 100.0*match/(o_real.width*o_real.height);
            }

data2d_t input = { 1, 10, (fixed[]){ 26146, 35678, 45425, 29331, 95099, 99711, 83365, 54175, 63708, 92940 } };

data2d_t real_output = { 1, 10, (fixed[]){ 72051, 74401, 76780, 72838, 88320, 89327, 85702, 78890, 81157, 87844 } };


data2d_t output;

# define ERROR_BOUND 0.0005

int main(){

    
    //************************ LAYER  0 ***********************//
    output = input;
    sigmoid_activation(output.data, 10);

    printf("Test result: %6.3f %%\n", measure_error(real_output, output, ERROR_BOUND));

    return 0;
}