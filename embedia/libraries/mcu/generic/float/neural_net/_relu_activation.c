/**
 * Function: relu_activation
 * Lines: 743-749
 */

void relu_activation(float *data, uint32_t length){
    uint32_t i;

    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? 0 : data[i];
    }
}
