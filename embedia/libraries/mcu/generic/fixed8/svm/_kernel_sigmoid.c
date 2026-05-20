/**
 * @brief Computes the sigmoid kernel function between an input vector and a support vector (8-bit fixed-point).
 *
 * Applies the sigmoid kernel formula: tanh(gamma * dot(x, sv) + intercept)
 * where dot(x, sv) is the dot product of the input and support vector.
 * Uses 32-bit intermediate accumulation with careful rounding and saturation to handle
 * 8-bit fixed-point precision constraints.
 *
 * @param[in] model Pointer to the SVM classifier model containing kernel parameters
 *                  (gamma and intercept).
 * @param[in] x Pointer to the input feature vector of size n_features (8-bit fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (8-bit fixed-point).
 *
 * @return The computed sigmoid kernel value between the input and support vector (8-bit fixed-point).
 *
 * @note Dot product is accumulated in int32 at scale 2^(2*FRC), then carefully
 *       rounded and saturated before conversion back to fixed-point scale.
 */

static inline compute_t kernel_sigmoid(
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

    /* arg = gamma * dot + intercept  (all fixed scale) */
    compute_t arg = FIXED_ADD(FIXED_MUL(model->kernel.gamma, dot),
                              model->kernel.intercept);

    return fixed_tanh(arg);
}
