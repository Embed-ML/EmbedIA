/**
 * Function: sigmoid_activation
 * Lines: 796-801
 */

void sigmoid_activation(float *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = 1 / (1 + exp(-data[i]));
    }
}
