void global_average_pooling1d_layer(data2d_t input, data1d_t* output){
    uint32_t c, i;

    output->length = input.channels;
    output->data   = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    const uint32_t recip_width = FIXED_AVG_RECIP(input.width);

    for(c = 0; c < input.channels; c++){
        int32_t sum = 0;
        const uint32_t ch_in = c * input.width;

        for(i = 0; i < input.width; i++){
            sum += input.data[ch_in + i];
        }

        output->data[c] = FIXED_AVG_APPLY(sum, recip_width);
    }
}