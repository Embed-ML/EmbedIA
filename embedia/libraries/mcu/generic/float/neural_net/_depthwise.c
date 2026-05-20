#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

static void depthwise(separable_conv2d_layer_t layer, data3d_t input, data3d_t *output) {
    uint32_t i, j, k, l, c, f_pos, i_pos;
    int32_t pad_h, pad_w, i_pad, j_pad;
    float sum;

    // Calcular padding
    if (layer.padding == PAD_SAME) {
        int32_t pad_total_h = (output->height - 1) * layer.strides.h + layer.depth_kernel_sz.h - input.height;
        int32_t pad_total_w = (output->width - 1) * layer.strides.w + layer.depth_kernel_sz.w - input.width;
        pad_h = (pad_total_h > 0) ? (pad_total_h / 2) : 0;
        pad_w = (pad_total_w > 0) ? (pad_total_w / 2) : 0;
    } else {  // PAD_VALID
        pad_h = 0;
        pad_w = 0;
    }

    for (c = 0; c < layer.depth_channels; c++) {
        for (i = 0; i < output->height; i++) {
            for (j = 0; j < output->width; j++) {
                sum = 0;
                for (k = 0; k < layer.depth_kernel_sz.h; k++) {
                    for (l = 0; l < layer.depth_kernel_sz.w; l++) {
                        i_pad = (int32_t)(i * layer.strides.h) + (int32_t)k - (int32_t)pad_h;
                        j_pad = (int32_t)(j * layer.strides.w) + (int32_t)l - (int32_t)pad_w;

                        if (i_pad >= 0 && i_pad < (int32_t)input.height &&
                            j_pad >= 0 && j_pad < (int32_t)input.width) {
                            f_pos = (c * layer.depth_kernel_sz.h * layer.depth_kernel_sz.w) +
                                   k * layer.depth_kernel_sz.w + l;
                            i_pos = (c * input.height * input.width) + i_pad * input.width + j_pad;
                            sum += layer.depth_weights[f_pos] * input.data[i_pos];
                        }
                    }
                }
                output->data[c * output->height * output->width + i * output->width + j] =
                    sum + layer.depth_bias[c];
            }
        }
    }
}