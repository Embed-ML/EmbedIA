/**
 * Function: average_pooling1d_layer
 * Lines: 570-593
 */

void average_pooling1d_layer(pooling1d_layer_t pool, data2d_t input, data2d_t* output){
    uint32_t c, i, aux;
    uint32_t count = pool.size;
    fixed avg, num;

    // Calculate output dimensions
    output->width = ((uint32_t)((input.width - pool.size) / pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->channels * output->width);

    // Process each channel
    for(c = 0; c < output->channels; c++){
        // Process each output position
        for(i = 0; i < output->width; i++){
            avg = 0;
            // Sum values in pooling window
            for(aux = 0; aux < pool.size; aux++){
                num = input.data[c * input.width + (i * pool.strides + aux)];
                avg += num;
            }
            output->data[c * output->width + i] = FIXED_DIV(avg, INT_TO_FIXED(count));
        }
    }
}
