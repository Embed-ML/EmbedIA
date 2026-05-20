/**
 * Function: sigmoid_activation
 * Lines: 535-540
 */

void sigmoid_activation(fixed *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = FIXED_DIV(FIX_ONE, FIX_ONE + fixed_exp(-data[i]));
    }
}
