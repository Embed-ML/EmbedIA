static void pointwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t *output, uint32_t delta) {
    uint32_t i, j, c, i_pos;
    dfixed sum;

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            sum = 0;
            for (c = 0; c < layer.point_channels; c++) {
                i_pos = (c * input.height * input.width) + (i * 1) * input.width + (j * 1);
                sum += DFIXED_MUL(filter.weights[c], input.data[i_pos]);
            }
			sum = sum + FIXED_TO_DFIXED(filter.bias);
			if (sum > DFIX_MAX)
				sum = FIX_MAX;
			else if (sum < DFIX_MIN)
				sum = FIX_MIN;
			else sum = DFIXED_TO_FIXED(sum);

			output->data[delta + i*output->width + j] = sum;
        }
    }
}
