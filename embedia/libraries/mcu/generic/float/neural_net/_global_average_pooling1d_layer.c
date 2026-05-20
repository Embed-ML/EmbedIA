void global_average_pooling1d_layer(data2d_t input, data1d_t* output){
    uint32_t c, i;

    output->length = input.channels;
    output->data   = (float*)swap_alloc(sizeof(float) * output->length);

    const float    recip_width = 1.0f / (float)input.width;

    for(c = 0; c < input.channels; c++){
        float sum = 0.0f;
        const uint32_t ch_in = c * input.width;

        for(i = 0; i < input.width; i++){
            sum += input.data[ch_in + i];
        }

        output->data[c] = sum * recip_width;
    }
}