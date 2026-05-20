void average_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
    uint32_t c, i, j, aux1, aux2;

    output->height   = ((uint32_t)((input.height - pool.size) / pool.strides)) + 1;
    output->width    = ((uint32_t)((input.width  - pool.size) / pool.strides)) + 1;
    output->channels = input.channels;
    output->data     = (float*)swap_alloc(sizeof(float) * output->channels * output->height * output->width);

    // Precompute reciprocal as float — replaces division with multiplication inside loop
    float recip_pool = 1.0f / (float)(pool.size * pool.size);

    for(c = 0; c < output->channels; c++){
        uint32_t ch_in  = c * input.height  * input.width;
        uint32_t ch_out = c * output->height * output->width;

        for(i = 0; i < output->height; i++){
            uint32_t row_out = i * output->width;
            uint32_t row_in  = i * pool.strides * input.width;

            for(j = 0; j < output->width; j++){
                float sum = 0.0f;
                uint32_t col_in = j * pool.strides;

                for(aux1 = 0; aux1 < pool.size; aux1++){
                    uint32_t row_pool = aux1 * input.width;

                    for(aux2 = 0; aux2 < pool.size; aux2++){
                        sum += input.data[ch_in + row_in + row_pool + col_in + aux2];
                    }
                }

                output->data[ch_out + row_out + j] = sum * recip_pool;
            }
        }
    }
}