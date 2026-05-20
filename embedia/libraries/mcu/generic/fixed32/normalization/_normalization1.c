/**
 * Function: normalization1
 * Lines: 831-840
 */

void normalization1(normalization_layer_t n, data1d_t input, data1d_t * output){
    uint32_t i;

    output->length = input.length;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] =  FIXED_MUL((input.data[i]-n.sub_val[i]),n.inv_div_val[i]);
    }
}
