/**
 * Function: conv1d_layer
 * Lines: 281-304
 */

void conv1d_layer(conv1d_layer_t layer, data2d_t input, data2d_t * output) {
    int32_t delta, i, k, f_pos, i_pos;
    int16_t f, c;
    float value;

    // calculate output size and allocate memory
    calc_alloc_conv1d_output(layer.n_filters, layer.kernel_size, layer.stride,
                            layer.padding, input, output);

    for (f = 0; f < layer.n_filters; f++) {
        delta = f * output->width;
        for (i = 0; i < output->width; i++) {
            value = 0;
            for (c = 0; c < layer.channels; c++) {
                for (k = 0; k < layer.kernel_size; k++) {
                    f_pos = (c * layer.kernel_size) + k;
                    i_pos = (c * input.width) + (i + k);
                    value += layer.filters[f].weights[f_pos] * input.data[i_pos];
                }
            }
            output->data[delta + i] = value + layer.filters[f].bias;
        }
    }
}
