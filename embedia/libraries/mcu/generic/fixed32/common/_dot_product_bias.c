EMBEDIA_INLINE realx_t dot_product_bias(
    const real_t *vec_a,
    const real_t *vec_b,
    uint32_t length,
    real_t bias
){
    realx_t sum = REALX_ZERO;
    const real_t *a = vec_a;
    const real_t *b = vec_b;
    uint32_t remaining = length;

    // Procesar en bloques de 4 con operaciones separadas
    for (; remaining >= 4; remaining -= 4) {
        sum = DFIXED_ADD(sum, DFIXED_MUL(*a++, *b++));
        sum = DFIXED_ADD(sum, DFIXED_MUL(*a++, *b++));
        sum = DFIXED_ADD(sum, DFIXED_MUL(*a++, *b++));
        sum = DFIXED_ADD(sum, DFIXED_MUL(*a++, *b++));
    }

    // Procesar elementos restantes
    for (; remaining > 0; remaining--) {
        sum = DFIXED_ADD(sum, DFIXED_MUL(*a++, *b++));
    }

    return DFIXED_ADD(sum, FX2DFX(bias));
}