void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t *output) {
    uint32_t delta, i;
    data3d_t depth_output;

    // calcular dimensiones de salida para depthwise
    calc_conv2d_output_size(layer.depth_channels, layer.depth_kernel_sz,
                            layer.strides, layer.padding, input, &depth_output);

    // dimensiones de la salida final
    output->channels = layer.n_filters;
    output->height   = depth_output.height;
    output->width    = depth_output.width;

    // alocar depth_output (temporal) y output final en el mismo slot
    uint32_t depth_sz  = sizeof(fixed) * depth_output.channels
                                       * depth_output.height
                                       * depth_output.width;
    uint32_t output_sz = sizeof(fixed) * output->channels
                                       * output->height
                                       * output->width;

    // alocar depth_output y output en el mismo slot de memoria
    swap_alloc_slice(depth_sz, output_sz, &depth_output.data, &output->data);

    // depthwise sobre depth_output
    depthwise(layer, input, &depth_output);

    // pointwise sobre depth_output → output
    for (i = 0; i < layer.n_filters; i++) {
        delta = i * output->height * output->width;
        pointwise(layer, layer.point_filters[i], depth_output, output, delta);
    }
}