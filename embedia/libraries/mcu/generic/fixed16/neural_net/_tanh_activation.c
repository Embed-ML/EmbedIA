/**
 * Function: tanh_activation
 * Lines: 523-528
 */

void tanh_activation(fixed *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = fixed_tanh(data[i]);
    }
}
