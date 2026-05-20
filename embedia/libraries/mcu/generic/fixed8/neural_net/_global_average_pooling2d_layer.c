void global_average_pooling2d_layer(data3d_t input, data1d_t* output){
    uint32_t c, i, j;

    output->length = input.channels;
    output->data   = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    const uint32_t spatial_size  = input.height * input.width;
    const uint32_t recip_spatial = FIXED_AVG_RECIP(spatial_size);

    for(c = 0; c < input.channels; c++){
        int32_t sum = 0;
        const uint32_t ch_in = c * spatial_size;

        for(i = 0; i < input.height; i++){
            const uint32_t row_in = i * input.width;

            for(j = 0; j < input.width; j++){
                sum += input.data[ch_in + row_in + j];
            }
        }

        output->data[c] = FIXED_AVG_APPLY(sum, recip_spatial);
    }
}