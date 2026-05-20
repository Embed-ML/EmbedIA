EMBEDIA_INLINE realx_t dot_product_bias(
    const real_t * vec_a,
    const real_t * vec_b,
    uint32_t length,
    real_t bias
){
    realx_t acc = REALX_ZERO;
    const real_t *a = vec_a;
    const real_t *b = vec_b;
    uint32_t remaining = length;

    for (; remaining >= 4u; remaining -= 4u) {
        acc += (*a++) * (*b++);
        acc += (*a++) * (*b++);
        acc += (*a++) * (*b++);
        acc += (*a++) * (*b++);
    }

    while (remaining--) {
        acc += (*a++) * (*b++);
    }

    return acc + bias;
}