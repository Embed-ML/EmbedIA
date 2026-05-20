/**
 * @brief Computes the RBF (Radial Basis Function) kernel between an input vector and a support vector (16-bit fixed-point).
 *
 * Applies the RBF kernel formula: exp(-gamma * ||x - sv||^2)
 * which computes the squared Euclidean distance between the vectors using 16-bit fixed-point arithmetic.
 *
 * @param[in] model Pointer to the SVM classifier model containing the gamma kernel parameter.
 * @param[in] x Pointer to the input feature vector of size n_features (16-bit fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (16-bit fixed-point).
 *
 * @return The computed RBF kernel value between the input and support vector (16-bit fixed-point).
 */

static inline compute_t kernel_rbf(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    compute_t sum = FIX_ZERO;
    compute_t d;

    for (uint16_t i = 0; i < model->n_features; i++) {
        d = FIXED_SUB(x[i], sv[i]);
        sum = FIXED_ADD(sum, FIXED_MUL(d, d));
    }
    /* exp(-gamma * sum) */
    compute_t exponent = FIXED_MUL(model->kernel.gamma, sum);
    return fixed_exp(FIXED_NEG(exponent));
}
