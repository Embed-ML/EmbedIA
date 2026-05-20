/**
 * Function: softsign_activation
 * Lines: 808-813
 */

void softsign_activation(float *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = data[i] / (fabs(data[i])+1);
    }
}
