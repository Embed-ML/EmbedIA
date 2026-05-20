void global_average_pooling2d_layer(data3d_t input, data1d_t* output){
    uint32_t c, i, j;

    output->length = input.channels;
    output->data   = (float*)swap_alloc(sizeof(float) * output->length);

    // Precompute reciprocal and spatial size once
    uint32_t spatial_size = input.height * input.width;
    float recip_spatial   = 1.0f / (float)spatial_size;

    for(c = 0; c < input.channels; c++){
        float sum = 0.0f;
        uint32_t ch_in = c * spatial_size;

        for(i = 0; i < input.height; i++){
            uint32_t row_in = i * input.width;

            for(j = 0; j < input.width; j++){
                sum += input.data[ch_in + row_in + j];
            }
        }

        output->data[c] = sum * recip_spatial;
    }
}