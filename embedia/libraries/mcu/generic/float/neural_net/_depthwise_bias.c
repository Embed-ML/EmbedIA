static void depthwise_bias(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output)
{
    uint32_t i, j, k, l, c;
    uint32_t f_pos, i_pos;
    uint32_t pad_h, pad_w;
    int32_t i_pad, j_pad;   // puede ser negativo → mejor usar signed
    float sum;

    pad_h = compute_padding(layer.strides.h, input.height, layer.kernel_sz.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.kernel_sz.w, output->width);

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            for (c = 0; c < layer.channels; c++) {

                sum = 0;

                for (k = 0; k < layer.kernel_sz.h; k++) {
                    for (l = 0; l < layer.kernel_sz.w; l++) {

                        i_pad = (int32_t)(i * layer.strides.h + k) - (int32_t)pad_h;
                        j_pad = (int32_t)(j * layer.strides.w + l) - (int32_t)pad_w;

                        if (i_pad >= 0 && i_pad < (int32_t)input.height &&
                            j_pad >= 0 && j_pad < (int32_t)input.width) {

                            f_pos = (c * layer.kernel_sz.h * layer.kernel_sz.w) +
                                    (k * layer.kernel_sz.w) + l;

                            i_pos = (c * input.height * input.width) +
                                    ((uint32_t)i_pad * input.width) + (uint32_t)j_pad;

                            sum += layer.weights[f_pos] * input.data[i_pos];
                        }
                    }
                }

                output->data[c * output->height * output->width + i * output->width + j] = sum + layer.bias[c];
            }
        }
    }
}