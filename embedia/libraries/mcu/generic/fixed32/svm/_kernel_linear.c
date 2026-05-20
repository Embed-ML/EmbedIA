/**
 * @brief Computes the linear kernel between an input vector and a support vector (fixed-point).
 *
 * Applies the linear kernel formula: dot(x, sv)
 * which is simply the dot product of the input and support vector using fixed-point arithmetic.
 * Uses high-precision fixed-point accumulation (dfixed) with rounding and saturation.
 *
 * @param[in] model Pointer to the SVM classifier model (kernel parameters are not used).
 * @param[in] x Pointer to the input feature vector of size n_features (fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (fixed-point).
 *
 * @return The computed linear kernel value (dot product) between the input and support vector (fixed-point).
 *
 * @note Dot product is accumulated in dfixed for high precision, then rounded and saturated
 *       to prevent overflow before returning.
 */

static inline compute_t kernel_linear(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    /* Mejora: acumulador en dfixed */
    dfixed acc = 0;

    for (uint16_t i = 0; i < model->n_features; i++) {
        DFIXED_MAC(acc, x[i], sv[i]);
    }

    return DFX2FX_RND_SAT(acc);
}
