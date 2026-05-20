/**
 * Function: normalization2
 * Lines: 594-602
 */

void normalization2(normalization_layer_t n, data1d_t input, data1d_t * output) {
    uint32_t i;
    output->length = input.length;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = FIXED_MUL(input.data[i],n.inv_div_val[i]);
    }
}
