/**
 * Function: leakyrelu_activation
 * Lines: 772-777
 */

void leakyrelu_activation(float *data, uint32_t length, float alpha){
    uint32_t i;
    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? alpha*data[i] : data[i];
    }
}
