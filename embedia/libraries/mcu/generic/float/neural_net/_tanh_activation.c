/**
 * Function: tanh_activation
 * Lines: 784-789
 */

void tanh_activation(float *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = 2/(1+exp(-2*data[i])) - 1;
    }
}
