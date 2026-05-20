void max_pooling1d_layer(pooling1d_layer_t pool, data2d_t input, data2d_t* output){
    uint32_t c, i, aux;

    output->width    = ((uint32_t)((input.width - pool.size) / pool.strides)) + 1;
    output->channels = input.channels;
    output->data     = (compute_t*)swap_alloc(sizeof(compute_t) * output->channels * output->width);

    for(c = 0; c < output->channels; c++){
        uint32_t channel_offset = c * input.width;      // evita c * input.width en cada i
        uint32_t out_offset     = c * output->width;    // evita c * output->width en cada i

        for(i = 0; i < output->width; i++){
            uint32_t start_idx = i * pool.strides;      // evita recalcular en cada aux
            compute_t max = input.data[channel_offset + start_idx];  // inicializar con primer elemento
                                                                   // evita FIX_MIN + comparación extra
            for(aux = 1; aux < pool.size; aux++){        // empieza en 1, ya leímos el 0
                compute_t num = input.data[channel_offset + start_idx + aux];
                if(num > max) max = num;
            }
            output->data[out_offset + i] = max;
        }
    }
}