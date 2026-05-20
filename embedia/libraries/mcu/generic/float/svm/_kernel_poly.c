/**
 * @brief Computes the polynomial kernel between an input vector and a support vector.
 *
 * Applies the polynomial kernel formula: (gamma * dot(x, sv) + intercept)^degree
 * with optimizations for common degree values (2 and 3) to avoid expensive powf() calls.
 *
 * @param[in] model Pointer to the SVM classifier model containing kernel parameters
 *                  (gamma, intercept, and degree).
 * @param[in] x Pointer to the input feature vector of size n_features.
 * @param[in] sv Pointer to the support vector of size n_features.
 *
 * @return The computed polynomial kernel value between the input and support vector.
 *
 * @note For degree values of 2 or 3, direct multiplication is used instead of powf()
 *       for improved performance, especially on embedded systems without FPU.
 */
static inline compute_t kernel_poly(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    compute_t dot = 0;
    compute_t term;

    for (uint16_t i = 0; i < model->n_features; i++) {
        dot += x[i] * sv[i];
    }

    term = model->kernel.gamma * dot + model->kernel.intercept;

    /* Avoid powf() for the common cases — measurably faster on targets * without an FPU. */
    switch (model->kernel.degree) {
        case 2:  return term * term;
        case 3:  return term * term * term;
        default: return powf(term, model->kernel.degree);
    }
}
