/**
 * Function: global_max_pooling1d_layer
 * Lines: 657-674
 */

void global_max_pooling1d_layer(data2d_t input, data1d_t* output) {
    uint32_t c, i;
    fixed max_val, val;

    output->length = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    for (c = 0; c < input.channels; c++) {
        max_val = FIX_MIN;
        for (i = 0; i < input.width; i++) {
            val = input.data[c * input.width + i];
            if (val > max_val) {
                max_val = val;
            }
        }
        output->data[c] = max_val;
    }
}
