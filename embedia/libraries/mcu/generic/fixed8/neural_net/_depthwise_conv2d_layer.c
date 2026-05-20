/* @embedia-note
 * DEPTHWISE CONVOLUTION IMPLEMENTATION:
 * - Applies one filter per input channel (depth multiplier = 1)
 * - Delegates actual computation to depthwise_bias helper function
 * - Output dimensions calculated by calc_alloc_conv2d_output (handles padding/strides)
 * - Memory allocated via swap_alloc for efficient buffer reuse
 */
void depthwise_conv2d_layer(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output) {
    calc_alloc_conv2d_output(layer.channels, layer.kernel_sz, layer.strides, layer.padding, input, output);
    depthwise_bias(layer, input, output);
}
