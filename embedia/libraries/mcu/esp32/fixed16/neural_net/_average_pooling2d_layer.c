/**
 * Function: average_pooling2d_layer
 * Lines: 417-444
 */

void average_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
    uint32_t c, i, j, aux1, aux2;
    fixed cant = INT_TO_FIXED(pool.size*pool.size);
    dfixed avg = 0;
    fixed num;

    output->height = ((uint32_t)((input.height - pool.size) / pool.strides)) + 1;
    output->width  = ((uint32_t)((input.width  - pool.size) / pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->channels * output->height * output->width);

    for(c=0; c<output->channels; c++){
        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){

                avg = 0;

                for(aux1=0; aux1<pool.size; aux1++){
                    for(aux2=0; aux2<pool.size; aux2++){
                        num = input.data[c*input.width*input.height + (i*pool.strides + aux1)*input.width + j*pool.strides + aux2];
                        avg += num;
                    }
                }
                output->data[c*output->width*output->height + i*output->width + j] = FIXED_DIV(avg,cant);
            }
        }
    }
}
