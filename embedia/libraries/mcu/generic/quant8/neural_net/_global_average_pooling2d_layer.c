void global_average_pooling2d_layer(data3d_t input, data1d_t* output) {
    uint32_t c, i, j;
    dfixed sum;
    uint32_t spatial_size = input.height * input.width;
    dfixed inv_spatial_size = DFIXED_DDIV_INT(DFIX_ONE, spatial_size); //Q16.16

    output->length = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    for (c = 0; c < input.channels; c++) {
        sum = 0;
        for (i = 0; i < input.height; i++) {
            for (j = 0; j < input.width; j++) {
                // Accumulate Q24.8
                sum = DFIXED_ADD(sum, input.data[c * input.height * input.width + i * input.width + j]);
            }
        }
        dfixed avg = sum * inv_spatial_size >> FIX_FRC_SZ; //Q24.8 *Q16.16 = Q40.24 >> Q48.16 (dfixed)

        output->data[c] = RX2R_SAT(avg);
    }
}

