/**
 * Function: max_pooling2d_layer
 * Lines: 378-408
 */

/* @embedia-note
 * MAX POOLING 2D IMPLEMENTATION:
 * - Downsamples spatial dimensions by selecting maximum value in each pool window
 * - Output size: ((input_size - pool_size) / strides) + 1
 * - Initialized with -FIX_MAX to handle negative values correctly
 * - No arithmetic operations on values (just comparison), so no overflow risk
 * - Memory allocated via swap_alloc for output buffer
 * - Assumes square pooling windows (pool.size x pool.size)
 */
void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
    uint32_t c, i , j, aux1, aux2;
    fixed max = -FIX_MAX;
    fixed num;

    output->height = ((uint16_t)((input.height - pool.size) / pool.strides)) + 1;
    output->width  = ((uint16_t)((input.width  - pool.size) / pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->channels * output->height * output->width);

    for(c=0; c<output->channels; c++){
        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){

                max = -FIX_MAX;

                for(aux1=0; aux1<pool.size; aux1++){
                        for(aux2=0; aux2<pool.size; aux2++){

                        num = input.data[c*input.width*input.height + (i*pool.strides + aux1)*input.width + j*pool.strides + aux2];

                        if(num>max){
                            max = num;
                        }
                    }
                }
                output->data[c*output->width*output->height + i*output->width + j] = max;
            }
        }
    }
}
