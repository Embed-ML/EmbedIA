/**
 * Function: conv1d_layer
 * Lines: 281-304
 */

void conv1d_layer(conv1d_layer_t layer, data2d_t input, data2d_t * output) {
    int32_t delta, i, k, f_pos, i_pos;
    int16_t f, c;
    int32_t acc;

    // calculate output size and allocate memory
    calc_alloc_conv1d_output(layer.n_filters, layer.kernel_size, layer.stride, layer.padding, input, output);

    const uint16_t scale_q = layer.qparam.scale_q;
    const int8_t zero_point = layer.qparam.zero_point;

    for (f = 0; f < layer.n_filters; f++) {
        delta = f * output->width;
        for (i = 0; i < output->width; i++) {
            acc = 0;
            for (c = 0; c < layer.channels; c++) {
                for (k = 0; k < layer.kernel_size; k++) {
                    f_pos = (c * layer.kernel_size) + k;
                    i_pos = (c * input.width) + (i + k);
                    ///value += FIXED_MUL(layer.filters[f].weights[f_pos], input.data[i_pos]);
                    int8_t weight_q = layer.filters[f].weights[f_pos];
                    // fixed16 × int8 → int32 (acumulación exacta)
                    acc += (int32_t)input.data[i_pos] * (int16_t)(weight_q - zero_point);
                }
            }
            // Dequantize once, add bias
            dfixed result = (dfixed)((acc * (int32_t)scale_q + QUANT_SCALE_HALF) >> SCALE_TO_FX_SHIFT);
            result += FX2DFX(layer.filters[f].bias);

            output->data[delta + i] = DFX2FX_SAT(result);
        }
    }
}
