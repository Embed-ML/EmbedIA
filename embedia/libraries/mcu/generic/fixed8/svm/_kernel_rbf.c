/**
 * @brief Computes the RBF (Radial Basis Function) kernel between an input vector and a support vector (8-bit fixed-point).
 *
 * Applies the RBF kernel formula: exp(-gamma * ||x - sv||^2)
 * which computes the squared Euclidean distance between the vectors using 8-bit fixed-point arithmetic.
 * Uses int16 for difference computation to prevent int8 wrap-around, and int32 for accumulation
 * with careful scale management.
 *
 * @param[in] model Pointer to the SVM classifier model containing the gamma kernel parameter.
 * @param[in] x Pointer to the input feature vector of size n_features (8-bit fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (8-bit fixed-point).
 *
 * @return The computed RBF kernel value between the input and support vector (8-bit fixed-point).
 *
 * @note Distance accumulation uses int32 at scale 2^(2*FRC). Exponent computation involves
 *       careful scale management and is clamped to FIX_EXP_MAX to prevent overflow.
 */

static inline compute_t kernel_rbf(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    int32_t sum = 0;  /* accumulates d*d terms, scale: 2^(2*FIX_FRC_SZ) */

    for (uint16_t i = 0; i < model->n_features; i++) {
        /* Compute difference in int16 to avoid int8 wrap-around.
         * Both x[i] and sv[i] are at fixed scale 2^FRC. */
        int16_t d = (int16_t)x[i] - (int16_t)sv[i];

        /* d*d at scale 2^(2*FRC). int32 prevents overflow. */
        sum += (int32_t)d * (int32_t)d;
    }

    /* Multiply gamma (scale 2^FRC) by sum (scale 2^(2*FRC)).
     * Result exponent32 is at scale 2^(3*FRC).
     * Using int32 here preserves the small fractional products that
     * would be lost if sum were first reduced to fixed scale. */
    int32_t exponent32 = (int32_t)model->kernel.gamma * (sum >> FIX_FRC_SZ);
    /* exponent32 is now at scale 2^(2*FRC), shift to fixed scale 2^FRC. */
    exponent32 += (int32_t)1 << (FIX_FRC_SZ - 1);  /* rounding */
    exponent32 >>= FIX_FRC_SZ;

    /* Saturate to fixed range. */
    compute_t exponent;
    if      (exponent32 < (int32_t)FIX_MIN) exponent = FIX_MIN;
    else if (exponent32 > (int32_t)FIX_MAX) exponent = FIX_MAX;
    else                                     exponent = (compute_t)exponent32;

    /* Clamp exponent to safe range for fixed_exp. */
    exponent = FIXED_MIN(exponent, FIX_EXP_MAX);

    /* result = exp(-exponent) */
    return fixed_exp(FIXED_NEG(exponent));
}
