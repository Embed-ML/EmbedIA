/**
 * Function: normalization2
 * Lines: 855-863
 */

void normalization2(normalization_layer_t n, data1d_t input, data1d_t * output) {
    uint32_t i;
    output->length = input.length;
    output->data = (float*)swap_alloc(sizeof(float)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = input.data[i]*n.inv_div_val[i];
    }
}
