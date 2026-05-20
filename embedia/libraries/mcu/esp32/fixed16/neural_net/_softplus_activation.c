/**
 * Function: softplus_activation
 * Lines: 560-566
 */

void softplus_activation(fixed *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = fixed_log( fixed_exp(data[i])+1 );
    }
}
