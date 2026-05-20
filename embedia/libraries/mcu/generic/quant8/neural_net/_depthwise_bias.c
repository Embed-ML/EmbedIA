static void depthwise_bias(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output){
    uint32_t i, j, k, l, c, f_pos, i_pos, pad_h, pad_w, j_pad, i_pad;
    dfixed sum;

    pad_h = compute_padding(layer.strides.h, input.height, layer.kernel_sz.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.kernel_sz.w, output->width);

    const uint16_t scale_q = layer.qparam.scale_q;
    const int8_t zero_point = layer.qparam.zero_point;

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            for (c = 0; c < layer.channels; c++) {
                sum = 0;
                for (k = 0; k < layer.kernel_sz.h; k++) {
                    for (l = 0; l < layer.kernel_sz.w; l++) {

                        i_pad = i * layer.strides.h + k - pad_h;
                        j_pad = j * layer.strides.w + l - pad_w;
                        // Check for valid input access within padded bounds
                        if (i_pad >= 0 && i_pad < input.height && j_pad >= 0 && j_pad < input.width) {
                            f_pos = (c * layer.kernel_sz.h * layer.kernel_sz.w) + k * layer.kernel_sz.w + l;
                            i_pos = (c * input.height * input.width) + i_pad * input.width + j_pad;

                            int8_t weight_q = layer.weights[f_pos];
                            // fixed16 × int8 → int32 (acumulación exacta)
                            sum += (int32_t)input.data[i_pos] * (int16_t)(weight_q - zero_point);


                        }
                    }
                }

                // Descuantizar una vez + bias
                dfixed result = (dfixed)((sum * (int32_t)scale_q + QUANT_SCALE_HALF) >> SCALE_TO_FX_SHIFT);
                result += FX2DFX(layer.bias[c]);

                uint32_t out_idx = c * output->height * output->width + i * output->width + j;

                output->data[out_idx] = DFX2FX_SAT(result);
            }
        }
    }
}