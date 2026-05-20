void global_average_pooling1d_layer(data2d_t input, data1d_t* output) {
    uint32_t c, i;
    realx_t sum;
    realx_t inv_width;

    output->length = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    // Calcular recíproco con máxima precisión disponible
    // En lugar de DFIXED_DIV, usar la división directa de 64-bit
    // que tiene mejor precisión
    inv_width = DFIXED_DDIV_INT(INT2DFX(1), input.width);  // 1 / width en dfixed

    for (c = 0; c < input.channels; c++) {
        sum = REALX_ZERO;

        for (i = 0; i < input.width; i++) {
            sum = DFIXED_ADD(sum, FX2DFX(input.data[c * input.width + i]));
        }

        // Multiplicar por recíproco con DFIXED_MUL corregida
        realx_t avg = DFIXED_DDIV_INT(sum, input.width);
        output->data[c] = RX2R_SAT(avg);
    }
}