/**
 * Function: kernel_rbf
 * Lines: 46-63
 */

static inline compute_t kernel_rbf(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const storage_t              *sv)
{
    computex_t sum_sq_diff = FX2DFX(FIX_ZERO);
    compute_t gamma = model->kernel.gamma;

    /* Optimización: reducir variables temporales y operaciones */
    for (uint16_t i = 0; i < model->n_features; i++) {
        compute_t sv_deq = DEQUANTIZE_FIXED(sv[i], model->qp_vectors);
        compute_t d = FIXED_SUB(x[i], sv_deq);
        sum_sq_diff += FX2DFX(FIXED_MUL(d, d));
    }

    compute_t exponent = FIXED_MUL(gamma, DFX2FX(sum_sq_diff));
    return fixed_exp(FIXED_NEG(exponent));
}
