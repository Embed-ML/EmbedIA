static void pointwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t *output, uint32_t delta) {
    uint32_t i, j, c, i_pos;
    dfixed sum;

    const uint16_t scale_q = layer.qparam.scale_q;
    const int8_t zero_point = layer.qparam.zero_point;

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            sum = 0;
            for (c = 0; c < layer.point_channels; c++) {
                i_pos = (c * input.height * input.width) + (i * 1) * input.width + (j * 1);

                int8_t weight_q = filter.weights[c];
                // fixed16 × int8 → int32 (acumulación exacta)
                sum += (int32_t)input.data[i_pos] * (int16_t)(weight_q - zero_point);
            }

            // Descuantizar una vez + bias
            dfixed result = (dfixed)((sum * (int32_t)scale_q + QUANT_SCALE_HALF) >> SCALE_TO_FX_SHIFT);
            result += FX2DFX(layer.point_filters[c].bias);

			output->data[delta + i*output->width + j] = DFX2FX_SAT(result);
        }
    }
}

