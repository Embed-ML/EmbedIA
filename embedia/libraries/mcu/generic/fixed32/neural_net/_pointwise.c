/**
 * Function: pointwise
 * Lines: 348-362
 */

static void pointwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t *output, uint32_t delta) {
    uint32_t i, j, c, i_pos;
    fixed sum;

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            sum = 0;
            for (c = 0; c < layer.point_channels; c++) {
                i_pos = (c * input.height * input.width) + i * input.width + j;
                sum += FIXED_MUL(filter.weights[c], input.data[i_pos]);
            }
            output->data[delta + i * output->width + j] = sum + filter.bias;
        }
    }
}
