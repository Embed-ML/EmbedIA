/**
 * Function: softsign_activation
 * Lines: 547-552
 */

void softsign_activation(fixed *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = FIXED_DIV(data[i],(fixed_abs(data[i])+FIX_ONE));
    }
}
