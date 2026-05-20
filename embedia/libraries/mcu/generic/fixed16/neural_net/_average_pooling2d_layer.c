/* @embedia-note
 * AVERAGE POOLING 2D OPTIMIZATIONS:
 * - Channel and row offsets precomputed outside inner loops (avoids repeated multiplications)
 * - Reciprocal precomputed once before loops — only one division total
 * - FIXED_AVG_APPLY uses only multiply + shift inside loop (no division)
 * - sum accumulated in int32_t: safe for pool up to 16x16 with values in [-8, 8)
 * - Precondition: input values in [-8, 8) and pool_size <= 16
 *   guarantees no overflow during accumulation
 */
void average_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
    uint32_t c, i, j, aux1, aux2;
    uint32_t pool_cells = pool.size * pool.size;

    output->height   = ((uint32_t)((input.height - pool.size) / pool.strides)) + 1;
    output->width    = ((uint32_t)((input.width  - pool.size) / pool.strides)) + 1;
    output->channels = input.channels;
    output->data     = (fixed*)swap_alloc(sizeof(fixed) * output->channels * output->height * output->width);

    // Precompute reciprocal once — avoids division inside the loop
    // uint32_t safely holds values up to (1<<FIX_AVG_PREC) for all fixed variants
    uint32_t recip_pool = FIXED_AVG_RECIP(pool_cells);

    for(c = 0; c < output->channels; c++){
        uint32_t ch_in  = c * input.height  * input.width;
        uint32_t ch_out = c * output->height * output->width;

        for(i = 0; i < output->height; i++){
            uint32_t row_out = i * output->width;
            uint32_t row_in  = i * pool.strides * input.width;

            for(j = 0; j < output->width; j++){
                int32_t sum = 0;
                uint32_t col_in = j * pool.strides;

                for(aux1 = 0; aux1 < pool.size; aux1++){
                    uint32_t row_pool = aux1 * input.width;

                    for(aux2 = 0; aux2 < pool.size; aux2++){
                        sum += input.data[ch_in + row_in + row_pool + col_in + aux2];
                    }
                }

                // Multiply by precomputed reciprocal — no division, no 64-bit arithmetic
                output->data[ch_out + row_out + j] = FIXED_AVG_APPLY(sum, recip_pool);
            }
        }
    }
}