/* @embedia-note
 * AVERAGE POOLING 1D FIXED8/FIXED16 OPTIMIZATIONS:
 * - Channel offsets precomputed outside inner loops (avoids repeated multiplications)
 * - FIXED_AVG_RECIP precomputes the reciprocal of pool.size once before the loop
 * - FIXED_AVG_APPLY uses only multiply + shift inside loop (no division)
 * - sum accumulated in int32_t: dfixed risks overflow for large pool sizes
 * - recip_pool in uint32_t: safely holds values up to (1<<FIX_AVG_PREC)
 * - Precondition: input values in [-8, 8) and pool_size <= 181
 *   guarantees no overflow during accumulation
 */
void average_pooling1d_layer(pooling1d_layer_t pool, data2d_t input, data2d_t* output){
    uint32_t c, i, aux;

    output->width    = ((uint32_t)((input.width - pool.size) / pool.strides)) + 1;
    output->channels = input.channels;
    output->data     = (fixed*)swap_alloc(sizeof(fixed) * output->channels * output->width);

    // Precompute reciprocal once — avoids division inside the loop
    // uint32_t safely holds values up to (1<<FIX_AVG_PREC) for all fixed variants
    uint32_t recip_pool = FIXED_AVG_RECIP(pool.size);

    for(c = 0; c < output->channels; c++){
        uint32_t ch_in  = c * input.width;
        uint32_t ch_out = c * output->width;

        for(i = 0; i < output->width; i++){
            int32_t sum = 0;
            uint32_t start_idx = i * pool.strides;

            for(aux = 0; aux < pool.size; aux++){
                sum += input.data[ch_in + start_idx + aux];
            }

            output->data[ch_out + i] = FIXED_AVG_APPLY(sum, recip_pool);
        }
    }
}