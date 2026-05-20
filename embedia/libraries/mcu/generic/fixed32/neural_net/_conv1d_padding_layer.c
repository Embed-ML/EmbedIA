/**
 * Function: conv1d_padding_layer
 * Lines: 219-248
 */

void conv1d_padding_layer(conv1d_layer_t layer, data2d_t input, data2d_t * output) {
    int32_t delta, i, k, f_pos, i_pos;
    int16_t f, c, i_pad, pad;
    dfixed value;

    // calculate output size and allocate memory
    calc_alloc_conv1d_output(layer.n_filters, layer.kernel_size, layer.stride,
                            layer.padding, input, output);

    pad = compute_padding_1d(layer.stride, input.width, layer.kernel_size, output->width);

    for (f = 0; f < layer.n_filters; f++) {
        delta = f * output->width;
        for (i = 0; i < output->width; i++) {
            value = 0;
            for (c = 0; c < layer.channels; c++) {
                for (k = 0; k < layer.kernel_size; k++) {
                    i_pad = i * layer.stride + k - pad;
                    // Check for valid input access within padded bounds
                    if (i_pad >= 0 && i_pad < input.width) {
                        f_pos = (c * layer.kernel_size) + k;
                        i_pos = (c * input.width) + i_pad;
                        value += FIXED_MUL(layer.filters[f].weights[f_pos], input.data[i_pos]);
                    }
                }
            }
            output->data[delta + i] = value + layer.filters[f].bias;
        }
    }
}
