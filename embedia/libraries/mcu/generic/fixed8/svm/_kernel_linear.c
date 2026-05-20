/**
 * @brief Computes the linear kernel between an input vector and a support vector (8-bit fixed-point).
 *
 * Applies the linear kernel formula: dot(x, sv)
 * which is simply the dot product of the input and support vector using 8-bit fixed-point arithmetic.
 * Uses int32 accumulation with careful rounding and saturation to handle precision constraints.
 *
 * @param[in] model Pointer to the SVM classifier model (kernel parameters are not used).
 * @param[in] x Pointer to the input feature vector of size n_features (8-bit fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (8-bit fixed-point).
 *
 * @return The computed linear kernel value (dot product) between the input and support vector (8-bit fixed-point).
 *
 * @note Dot product is accumulated in int32 at scale 2^(2*FRC), then rounded and saturated
 *       to prevent overflow before returning.
 */

static inline compute_t kernel_linear(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    int32_t acc = 0;  /* scale: 2^(2*FIX_FRC_SZ) */

    for (uint16_t i = 0; i < model->n_features; i++) {
        /* Each term: (int32_t)(x[i]) * (int32_t)(sv[i])
         * Both operands are fixed (int8), product fits in int32. */
        acc += (int32_t)x[i] * (int32_t)sv[i];
    }

    /* Shift right by FRC bits: scale 2^(2*FRC) → 2^FRC (fixed scale).
     * Round by adding 0.5 at the shifted scale before shifting. */
    acc += (int32_t)1 << (FIX_FRC_SZ - 1);
    acc >>= FIX_FRC_SZ;

    /* Saturate to fixed range [-128, 127] */
    if (acc < (int32_t)FIX_MIN) return FIX_MIN;
    if (acc > (int32_t)FIX_MAX) return FIX_MAX;
    return (compute_t)acc;
}
