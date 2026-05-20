void depthwise_conv2d_layer(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output){
    calc_alloc_conv2d_output(layer.channels, layer.kernel_sz, layer.strides, layer.padding, input, output);
    depthwise_bias(layer, input, output);
}
