/**
 * Function: kernel_sigmoid
 * Lines: 95-111
 */

static inline compute_t kernel_sigmoid(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const storage_t              *sv)
{
    compute_t dot = FIX_ZERO;
    compute_t sv_deq;
    compute_t gamma = model->kernel.gamma;
    compute_t intercept = model->kernel.intercept;

    for (uint16_t i = 0; i < model->n_features; i++) {
        sv_deq = DEQUANTIZE_FIXED(sv[i], model->qp_vectors);
        dot = FIXED_ADD(dot, FIXED_MUL(x[i], sv_deq));
    }
    compute_t arg = FIXED_ADD(FIXED_MUL(gamma, dot), intercept);
    return fixed_tanh(arg);
}
