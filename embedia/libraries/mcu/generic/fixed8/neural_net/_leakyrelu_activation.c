/**
 * Function: leakyrelu_activation
 * Lines: 511-516
 */

void leakyrelu_activation(fixed *data, uint32_t length, fixed alpha){
    uint32_t i;
    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? FIXED_MUL(alpha, data[i]) : data[i];
    }
}
