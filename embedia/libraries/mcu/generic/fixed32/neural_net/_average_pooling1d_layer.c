/* @embedia-note
 * AVERAGE POOLING 1D FIXED32 OPTIMIZATIONS:
 * - Accumulates directly in fixed (int32 Q16) without promotion to dfixed
 * - Channel offsets precomputed outside inner loops (avoids repeated multiplications)
 * - FIXED_AVG_RECIP precomputes the reciprocal of pool_cells once before the loop
 * - FIXED_AVG_APPLY uses only multiply + shift inside loop (no division)
 * - Precision: FIX_AVG_PREC = FIX_FRC_SZ-4 keeps fixed × fixed within int32
 * - Precondition: input values in [-8, 8) and pool_size <= 181
 *   guarantees no overflow during fixed accumulation
 */
void average_pooling1d_layer(pooling1d_layer_t pool, data2d_t input, data2d_t* output){
    uint32_t c, i, aux;

    output->width    = ((uint32_t)((input.width - pool.size) / pool.strides)) + 1;
    output->channels = input.channels;
    output->data     = (fixed*)swap_alloc(sizeof(fixed) * output->channels * output->width);

    // Precompute reciprocal once — avoids division inside the loop
    fixed recip_pool = FIXED_AVG_RECIP(pool.size);

    for(c = 0; c < output->channels; c++){
        uint32_t ch_in  = c * input.width;
        uint32_t ch_out = c * output->width;

        for(i = 0; i < output->width; i++){
            fixed sum = 0;
            uint32_t start_idx = i * pool.strides;

            for(aux = 0; aux < pool.size; aux++){
                sum += input.data[ch_in + start_idx + aux];
            }

            output->data[ch_out + i] = FIXED_AVG_APPLY(sum, recip_pool);
        }
    }
}