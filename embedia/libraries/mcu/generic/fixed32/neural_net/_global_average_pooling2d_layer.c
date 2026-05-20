/* @embedia-note
 * GLOBAL AVERAGE POOLING FIXED32:
 * - Accumulates in fixed (int32 Q16) — sufficient for typical MCU spatial sizes
 * - Uses direct division instead of reciprocal — one division per channel,
 *   negligible cost compared to spatial accumulation
 * - Avoids int64 arithmetic — reciprocal optimization deferred for Cortex-M4
 * - Precondition: spatial_size <= 181 guarantees no overflow during accumulation
 */
void global_average_pooling2d_layer(data3d_t input, data1d_t* output){
    uint32_t c, i, j;

    output->length = input.channels;
    output->data   = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    uint32_t spatial_size = input.height * input.width;

    for(c = 0; c < input.channels; c++){
        fixed sum = 0;
        uint32_t ch_in = c * spatial_size;

        for(i = 0; i < input.height; i++){
            uint32_t row_in = i * input.width;

            for(j = 0; j < input.width; j++){
                sum += input.data[ch_in + row_in + j];
            }
        }

        output->data[c] = (fixed)FIXED_DIV_INT(sum, (int32_t)spatial_size);
    }
}