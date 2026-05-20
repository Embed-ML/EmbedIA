/**
 * @brief Computes the linear kernel between an input vector and a support vector (16-bit fixed-point).
 *
 * Applies the linear kernel formula: dot(x, sv)
 * which is simply the dot product of the input and support vector using 16-bit fixed-point arithmetic.
 *
 * @param[in] model Pointer to the SVM classifier model (kernel parameters are not used).
 * @param[in] x Pointer to the input feature vector of size n_features (16-bit fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (16-bit fixed-point).
 *
 * @return The computed linear kernel value (dot product) between the input and support vector (16-bit fixed-point).
 */

static inline compute_t kernel_linear(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    compute_t acc = FIX_ZERO;
    for (uint16_t i = 0; i < model->n_features; i++) {
        acc = FIXED_ADD(acc, FIXED_MUL(x[i], sv[i]));
    }
    return acc;
}
