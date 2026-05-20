/**
 * Function: dense_layer
 * Lines: 448-457
 */

void dense_layer(dense_layer_t *layer, data1d_t *input, data1d_t *output) {
    output->length = layer->output_size;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    for (uint32_t i = 0; i < layer->output_size; i++) {
        // Get the weights for the i-th neuron: strides across input_size
        const fixed *neuron_weights = &layer->weights[i * layer->input_size];
        output->data[i] =  RX2R_SAT(dot_product_bias(neuron_weights, input->data, input->length, layer->biases[i]));
    }
}
