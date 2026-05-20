/**
 * Function: dense_layer
 * Lines: 399-443
 */

void dense_layer(dense_layer_t * dense_layer, data1d_t * input, data1d_t * output){

    uint32_t i, j;
    int32_t acc;
    dfixed result;

    output->length = dense_layer->output_size;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * dense_layer->output_size);

    const int input_size = input->length;
    const uint16_t scale_q = dense_layer->qparam.scale_q;
    const int8_t zero_point = dense_layer->qparam.zero_point;

    const int8_t *weights_ptr = dense_layer->weights;
    const fixed *input_ptr = input->data;

    for (i = 0; i < dense_layer->output_size; i++)
    {
        acc = 0;

        // Loop unrolling: procesar 4 elementos a la vez
        j = 0;
        for (; j + 3 < input_size; j += 4)
        {
            acc += (int32_t)input_ptr[j+0] * (int16_t)(weights_ptr[j+0] - zero_point);
            acc += (int32_t)input_ptr[j+1] * (int16_t)(weights_ptr[j+1] - zero_point);
            acc += (int32_t)input_ptr[j+2] * (int16_t)(weights_ptr[j+2] - zero_point);
            acc += (int32_t)input_ptr[j+3] * (int16_t)(weights_ptr[j+3] - zero_point);
        }

        // Elementos restantes
        for (; j < input_size; j++)
        {
            acc += (int32_t)input_ptr[j] * (int16_t)(weights_ptr[j] - zero_point);
        }

        // Descuantización + bias
        result = (dfixed)((acc * (int32_t)scale_q + QUANT_SCALE_HALF) >> SCALE_TO_FX_SHIFT);
        result += dense_layer->biases[i];

        output->data[i] = DFX2FX_SAT(result);

        weights_ptr += input_size;
    }
}
