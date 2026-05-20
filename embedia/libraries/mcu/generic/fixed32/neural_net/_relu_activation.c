/**
 * Function: relu_activation
 * Lines: 735-741
 */

void relu_activation(fixed *data, uint32_t length){
    uint32_t i;

    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? 0 : data[i];
    }
}
