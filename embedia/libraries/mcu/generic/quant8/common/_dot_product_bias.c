EMBEDIA_INLINE realx_t dot_product_bias(
    const compute_t *vec_a,
    const storage_t *vec_b,
    uint32_t length,
    real_t bias,
    qparam_t qp_vec_b
) {
    computex_t acc = CO2CX(bias);

    for (uint32_t i = 0; i < length; i++) {
        compute_t b = ST2CO(vec_b[i], qp_vec_b);
        DFIXED_MAC(acc, vec_a[i], b);
    }

    return acc;
}