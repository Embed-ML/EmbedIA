static void pointwise(separable_conv2d_layer_t layer, filter_t filter,
                     data3d_t input, data3d_t *output, uint32_t delta) {
    uint32_t i, j, c;
    uint32_t spatial_size = input.height * input.width;

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            float sum = 0.0f;

            for (c = 0; c < layer.point_channels; c++) {
                uint32_t input_idx = c * spatial_size + i * input.width + j;
                sum += filter.weights[c] * input.data[input_idx];
            }

            uint32_t output_idx = delta + i * output->width + j;
            output->data[output_idx] = sum + filter.bias;
        }
    }
}