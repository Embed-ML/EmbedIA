#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../embedia/float/embedia.h"
#include "../embedia/debug/embedia_debug.h"


conv2d_layer_t init_conv2d_data(void){

        static filter_t filters[4];

        static const float weights0[]={
           -0.24145030975341797,    0.11338114738464355,    0.1882951259613037,
           -0.2120606005191803,    0.0667535662651062,    -0.09122979640960693,
           0.31222647428512573,    -0.20956943929195404,    0.21217995882034302
        };
        static filter_t filter0 = {1, 3, weights0, 0.0};
        filters[0]=filter0;

        static const float weights1[]={
           -0.1826169341802597,    -0.34202197194099426,    0.221848726272583,
           -0.10085070133209229,    -0.0090562105178833,    0.1751686930656433,
           -0.18389146029949188,    -0.14596666395664215,    -0.30948078632354736
        };
        static filter_t filter1 = {1, 3, weights1, 0.0};
        filters[1]=filter1;

        static const float weights2[]={
           -0.09333878755569458,    -0.2912323474884033,    -0.25017109513282776,
           -0.00824311375617981,    0.35675525665283203,    -0.08929911255836487,
           0.2523718476295471,    0.1605854034423828,    -0.13304349780082703
        };
        static filter_t filter2 = {1, 3, weights2, 0.0};
        filters[2]=filter2;

        static const float weights3[]={
           0.08463281393051147,    -0.008681774139404297,    -0.06654176115989685,
           0.15492606163024902,    0.20798403024673462,    0.26175594329833984,
           0.2971768379211426,    -0.08004996180534363,    -0.3034684360027313
        };
        static filter_t filter3 = {1, 3, weights3, 0.0};
        filters[3]=filter3;

        conv2d_layer_t layer = {4, filters };
        return layer;
}

float measure_error(data3d_t o_real, data3d_t o_pred, float err){
    int x, y, c, pr, pp, match;
    for (match=0,c=0, pp=0; c<o_real.channels; c++){
        for (y=0; y<o_real.height; y++){
            for (x=0; x<o_real.width; x++, pp++){
               pr = (y*o_real.width+x)*o_real.channels + c;
               printf("%f   %f\n", o_real.data[pr], o_pred.data[pp]);
               if (fabs(o_real.data[pr]-o_pred.data[pp]) <= err)
                    match++;
            }
        }
    }
    return 100.0*match/(o_real.channels*o_real.height*o_real.width);
}
conv2d_layer_t conv2d_data;
data3d_t input = { 1, 5, 5, (float[]){ 0.68116429, 0.25812691, 0.29089657, 0.11086272, 0.91856703, 0.70146573,
 0.42881027, 0.75326967, 0.64843872, 0.69085408, 0.43347126, 0.47114719,
 0.03197366, 0.7512518 , 0.87351031, 0.52853741, 0.11033391, 0.99697199,
 0.21276954, 0.34150734, 0.48992304, 0.47000246, 0.91665766, 0.67212629,
 0.89810136 } }
;
data3d_t real_output = { 4, 3, 3, (float[]){ -0.22588813, -0.24919829,  0.0492053 ,  0.51248425,  0.19152951,
 -0.38232306,  0.09483641,  0.31425148, -0.02630033, -0.23395343,
 -0.11331122,  0.07921636,  0.31113675, -0.5718743 , -0.1986825 ,
  0.024724  , -0.19168614, -0.339959  , -0.3215481 ,  0.15130335,
  0.32455826, -0.3831733 , -0.00190157,  0.57760763,  0.00807278,
 -0.55525815, -0.1625322 ,  0.22605656,  0.15232219, -0.34143746,
  0.270931  ,  0.13207516,  0.34947127, -0.65626585, -0.1833223 ,
  0.17222041 } }
;
data3d_t output;

# define ERROR_BOUND 1e-05
int main(){
    conv2d_data = init_conv2d_data();

    conv2d_layer(conv2d_data, input, &output);

    print_data3d_t("conv2d", output);

    printf("Test result: %6.3f %%\n", measure_error(real_output, output, ERROR_BOUND));
return 0;
}
