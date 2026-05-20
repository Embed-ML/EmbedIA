static void depthwise(separable_conv2d_layer_t layer, data3d_t input, data3d_t *output) {
    uint32_t i, j, k, l, c;
    uint32_t f_pos, i_pos;
    int32_t pad_h, pad_w;
    int32_t i_pad, j_pad;
    dfixed sum;


    pad_h = compute_padding(layer.strides.h, input.height, layer.depth_kernel_sz.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.depth_kernel_sz.w, output->width);

    const uint16_t scale_q = layer.qparam.scale_q;
    const int8_t zero_point = layer.qparam.zero_point;

    for (c = 0; c < layer.depth_channels; c++) {
        for (i = 0; i < output->height; i++) {
            for (j = 0; j < output->width; j++) {

                sum = 0;

                for (k = 0; k < layer.depth_kernel_sz.h; k++) {
                    for (l = 0; l < layer.depth_kernel_sz.w; l++) {

                        i_pad = (int32_t)(i * layer.strides.h) + (int32_t)k - pad_h;
                        j_pad = (int32_t)(j * layer.strides.w) + (int32_t)l - pad_w;

                        if (i_pad >= 0 && i_pad < (int32_t)input.height &&
                            j_pad >= 0 && j_pad < (int32_t)input.width) {

                            f_pos = (c * layer.depth_kernel_sz.h * layer.depth_kernel_sz.w) +
                                    k * layer.depth_kernel_sz.w + l;

                            i_pos = (c * input.height * input.width) +
                                    (uint32_t)i_pad * input.width + (uint32_t)j_pad;

                            int8_t weight_q = layer.depth_weights[f_pos];
                            // fixed16 × int8 → int32 (acumulación exacta)
                            sum += (int32_t)input.data[i_pos] * (int16_t)(weight_q - zero_point);

                        }
                    }
                }
                // Descuantizar una vez + bias
                dfixed result = (dfixed)((sum * (int32_t)scale_q + QUANT_SCALE_HALF) >> SCALE_TO_FX_SHIFT);
                result += FX2DFX(layer.point_filters[c].bias);

                uint32_t out_idx = c * output->height * output->width + i * output->width + j;

                output->data[out_idx] = DFX2FX_SAT(result);
            }
        }
    }
}
