/**
 * Function: global_max_pooling1d_layer
 * Lines: 664-681
 */

void global_max_pooling1d_layer(data2d_t input, data1d_t* output) {
    uint32_t c, i;
    float max_val;

    output->length = input.channels;
    output->data = (float*)swap_alloc(sizeof(float) * output->length);

    for (c = 0; c < input.channels; c++) {
        max_val = -INFINITY;
        for (i = 0; i < input.width; i++) {
            float val = input.data[c * input.width + i];
            if (val > max_val) {
                max_val = val;
            }
        }
        output->data[c] = max_val;
    }
}
