#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/fixed32/embedia.h"
#include "../embedia/fixed32/fixed.h"


            float measure_error(data2d_t o_real, data2d_t o_pred, float err){
                int i, match;
                for (match=0,i=0; i<o_real.width*o_real.height; i++){
                    printf("%f   %f\n", FX2FL(o_real.data[i]), FX2FL(o_pred.data[i]));
                    if (fabs(FX2FL(o_real.data[i])-FX2FL(o_pred.data[i])) <= err){
                        match++;
                    }
                }
                return 100.0*match/(o_real.width*o_real.height);
            }

data2d_t input = { 1, 10, (fixed[]){   7681,  67300,  76058, 119731,  85988,  97190,  94329,  55579,  45719,
 109982 } };

data2d_t real_output = { 1, 10, (fixed[]){ 67456, 82001, 84034, 93548, 86294, 88778, 88151, 79226, 76851, 91524 } };


data2d_t output;

# define ERROR_BOUND 0.0005

int main(){


    //************************ LAYER  0 ***********************//
    output = input;
    sigmoid_activation(output.data, 10);

    printf("Test result: %6.3f %%\n", measure_error(real_output, output, ERROR_BOUND));

    return 0;
}
