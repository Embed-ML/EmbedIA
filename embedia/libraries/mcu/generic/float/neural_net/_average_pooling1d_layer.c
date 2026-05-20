void average_pooling1d_layer(pooling1d_layer_t pool, data2d_t input, data2d_t* output){
    uint32_t c, t, k;

    output->channels = input.channels;
    output->width    = (input.width - pool.size) / pool.strides + 1;
    output->data     = (real_t*)swap_alloc(sizeof(real_t) * output->channels * output->width);

    // Precompute reciprocal once — replaces division with multiplication inside loop
    float recip_pool = 1.0f / (float)pool.size;

    for(c = 0; c < input.channels; c++){
        uint32_t ch_in  = c * input.width;
        uint32_t ch_out = c * output->width;

        for(t = 0; t < output->width; t++){
            real_t acc = 0;
            uint32_t base = ch_in + t * pool.strides;

            for(k = 0; k < pool.size; k++){
                acc += input.data[base + k];
            }

            output->data[ch_out + t] = acc * recip_pool;
        }
    }
}