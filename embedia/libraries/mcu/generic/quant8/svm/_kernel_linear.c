/**
 * Function: kernel_linear
 * Lines: 26-41
 */

static inline compute_t kernel_linear(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const storage_t              *sv)
{
    computex_t acc = 0;
    int32_t sv_int;

    /* Acumulamos en dfixed (int32_t) sin dequantizar */
    for (uint16_t i = 0; i < model->n_features; i++) {
        sv_int = (int32_t)sv[i];  /* storage_t es int8_t */
        acc += (computex_t)x[i] * sv_int;
    }
    /* Dequantizamos el resultado final */
    return FIXED_MUL(DFX2FX(acc), FL2FX(1.0f / model->qp_vectors.scale_q));
}
