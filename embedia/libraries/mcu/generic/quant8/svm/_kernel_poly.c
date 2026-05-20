/**
 * Function: kernel_poly
 * Lines: 68-93
 */

static inline compute_t kernel_poly(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const storage_t              *sv)
{
    compute_t dot = FIX_ZERO;
    compute_t sv_deq;
    compute_t gamma = model->kernel.gamma;
    compute_t intercept = model->kernel.intercept;

    /* Producto punto optimizado */
    for (uint16_t i = 0; i < model->n_features; i++) {
        sv_deq = DEQUANTIZE_FIXED(sv[i], model->qp_vectors);
        dot = FIXED_ADD(dot, FIXED_MUL(x[i], sv_deq));
    }

    compute_t term = FIXED_ADD(FIXED_MUL(gamma, dot), intercept);

    /* Fast path for common degrees - avoid expensive fixed_pow */
    switch (model->kernel.degree) {
        case 2: return FIXED_MUL(term, term);
        case 3: return FIXED_MUL(FIXED_MUL(term, term), term);
        case 4: return fixed_pow(term, INT_TO_FIXED(model->kernel.degree));
        default: return fixed_pow(term, INT_TO_FIXED(model->kernel.degree));
    }
}
