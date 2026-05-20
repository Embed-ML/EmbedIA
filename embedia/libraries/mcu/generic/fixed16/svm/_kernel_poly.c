/**
 * @brief Computes the polynomial kernel between an input vector and a support vector (16-bit fixed-point).
 *
 * Applies the polynomial kernel formula: (gamma * dot(x, sv) + intercept)^degree
 * with optimizations for common degree values (2 and 3) to avoid expensive fixed_pow() calls.
 * Uses 16-bit fixed-point arithmetic for embedded systems.
 *
 * @param[in] model Pointer to the SVM classifier model containing kernel parameters
 *                  (gamma, intercept, and degree).
 * @param[in] x Pointer to the input feature vector of size n_features (16-bit fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (16-bit fixed-point).
 *
 * @return The computed polynomial kernel value between the input and support vector (16-bit fixed-point).
 *
 * @note For degree values of 2 or 3, direct multiplication is used instead of fixed_pow()
 *       for improved performance on embedded systems.
 */

static inline compute_t kernel_poly(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    compute_t dot = FIX_ZERO;
    compute_t term;

    for (uint16_t i = 0; i < model->n_features; i++) {
        dot = FIXED_ADD(dot, FIXED_MUL(x[i], sv[i]));
    }

     term = FIXED_ADD(FIXED_MUL(model->kernel.gamma, dot), model->kernel.intercept);

    /* Casos especiales rápidos (evitamos fixed_pow cuando es posible) */
    switch (model->kernel.degree) {
        case 2:  return FIXED_MUL(term, term);
        case 3:  return FIXED_MUL(FIXED_MUL(term, term), term);
        default: return fixed_pow(term, INT_TO_FIXED(model->kernel.degree));
    }
}
