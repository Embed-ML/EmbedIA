/**
 * Function: softplus_activation
 * Lines: 821-827
 */

void softplus_activation(float *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = log( exp(data[i])+1 );
    }
}
