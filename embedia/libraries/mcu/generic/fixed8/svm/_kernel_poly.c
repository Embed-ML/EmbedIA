/**
 * @brief Computes the polynomial kernel between an input vector and a support vector (8-bit fixed-point).
 *
 * Applies the polynomial kernel formula: (gamma * dot(x, sv) + intercept)^degree
 * with optimizations for common degree values (2 and 3) to avoid expensive fixed_pow() calls.
 * Uses 32-bit intermediate accumulation with careful rounding and saturation for
 * 8-bit fixed-point precision constraints.
 *
 * @param[in] model Pointer to the SVM classifier model containing kernel parameters
 *                  (gamma, intercept, and degree).
 * @param[in] x Pointer to the input feature vector of size n_features (8-bit fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (8-bit fixed-point).
 *
 * @return The computed polynomial kernel value between the input and support vector (8-bit fixed-point).
 *
 * @note For degree values of 2 or 3, direct multiplication is used instead of fixed_pow()
 *       for improved performance. Dot product is accumulated in int32 at scale 2^(2*FRC).
 */

static inline compute_t kernel_poly(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    int32_t acc = 0;  /* scale: 2^(2*FIX_FRC_SZ) */

    for (uint16_t i = 0; i < model->n_features; i++) {
        acc += (int32_t)x[i] * (int32_t)sv[i];
    }

    /* Bring dot product to fixed scale: >> FRC with rounding + saturate. */
    acc += (int32_t)1 << (FIX_FRC_SZ - 1);
    acc >>= FIX_FRC_SZ;

    compute_t dot;
    if      (acc < (int32_t)FIX_MIN) dot = FIX_MIN;
    else if (acc > (int32_t)FIX_MAX) dot = FIX_MAX;
    else                              dot = (compute_t)acc;

    /* term = gamma * dot + intercept  (all fixed scale) */
    compute_t term = FIXED_ADD(FIXED_MUL(model->kernel.gamma, dot),
                               model->kernel.intercept);

    /* Fast special cases avoid fixed_pow() */
    switch (model->kernel.degree) {
        case 2:  return FIXED_MUL(term, term);
        case 3:  return FIXED_MUL(FIXED_MUL(term, term), term);
        default: return fixed_pow(term, INT_TO_FIXED(model->kernel.degree));
    }
}
