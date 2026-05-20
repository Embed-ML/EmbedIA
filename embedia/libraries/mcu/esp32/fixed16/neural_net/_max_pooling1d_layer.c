/**
 * Function: max_pooling1d_layer
 * Lines: 538-562
 */

void max_pooling1d_layer(pooling1d_layer_t pool, data2d_t input, data2d_t* output){
    uint32_t c, i, aux;
    fixed max, num;

    // Calculate output dimensions
    output->width = ((uint16_t) ((input.width - pool.size)/pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->channels * output->width);

    // Process each channel
    for(c = 0; c < output->channels; c++){
        // Process each output position
        for(i = 0; i < output->width; i++){
            max = FIX_MIN;
            // Find maximum in pooling window
            for(aux = 0; aux < pool.size; aux++){
                num = input.data[c * input.width + (i * pool.strides + aux)];
                if(num > max){
                    max = num;
                }
            }
            output->data[c * output->width + i] = max;
        }
    }
}
