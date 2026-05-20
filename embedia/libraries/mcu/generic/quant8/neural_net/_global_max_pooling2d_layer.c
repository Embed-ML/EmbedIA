/**
 * Function: global_max_pooling2d_layer
 * Lines: 605-624
 */

void global_max_pooling2d_layer(data3d_t input, data1d_t* output) {
    uint32_t c, i, j;
    fixed max_val;

    output->length = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    for (c = 0; c < input.channels; c++) {
        max_val = FIX_MIN;
        for (i = 0; i < input.height; i++) {
            for (j = 0; j < input.width; j++) {
                fixed val = input.data[c * input.height * input.width + i * input.width + j];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        output->data[c] = max_val;
    }
}
