/**
 * Function: dense_layer
 * Lines: 349-369
 */

/* @embedia-note
 * FIXED8 DENSE LAYER IMPLEMENTATION:
 * - Uses dot_product_bias helper for optimized vector operations
 * - Explicit saturation checks (DFIX_MAX/DFIX_MIN) prevent overflow/underflow
 * - Weight matrix stored in row-major order: weights[neuron_idx * input_size + input_idx]
 * - Each neuron processes entire input vector before moving to next neuron
 * - Critical for memory efficiency: output allocated via swap_alloc
 */
void dense_layer(dense_layer_t *layer, data1d_t *input, data1d_t *output) {
    output->length = layer->output_size;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    for (uint32_t i = 0; i < layer->output_size; i++) {
        // Get the weights for the i-th neuron: strides across input_size
        const fixed *neuron_weights = &layer->weights[i * layer->input_size];
        const dfixed res = dot_product_bias(
            neuron_weights,           // Peso del i-ésimo neurón
            input->data,              // Datos de entrada
            input->length,            // Tamaño del vector de entrada
            layer->biases[i]          // Bias del i-ésimo neurón
        );
        if (res> DFIX_MAX)
            output->data[i] = FIX_MAX;
        else if (res < DFIX_MIN)
                 output->data[i] = FIX_MIN;
             else
                 output->data[i] = DFIXED_TO_FIXED(res);
    }
}
