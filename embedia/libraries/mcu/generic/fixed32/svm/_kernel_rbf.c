/**
 * @brief Computes the RBF (Radial Basis Function) kernel between an input vector and a support vector (fixed-point).
 *
 * Applies the RBF kernel formula: exp(-gamma * ||x - sv||^2)
 * which computes the squared Euclidean distance between the vectors using fixed-point arithmetic.
 * Includes saturation protection on the exponent argument to prevent overflow.
 *
 * @param[in] model Pointer to the SVM classifier model containing the gamma kernel parameter.
 * @param[in] x Pointer to the input feature vector of size n_features (fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (fixed-point).
 *
 * @return The computed RBF kernel value between the input and support vector (fixed-point).
 *
 * @note Distance accumulation uses dfixed for high precision. Exponent is clamped to FIX_EXP_MAX
 *       to prevent overflow in the exponential computation.
 */

static inline compute_t kernel_rbf(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{

    dfixed sum = 0;
    compute_t d;

    for (uint16_t i = 0; i < model->n_features; i++) {
        d = FIXED_SUB(x[i], sv[i]);
        DFIXED_MAC(sum, d, d);
    }

    /* Convertimos con control de saturación */
    compute_t sum_fx = DFX2FX_RND_SAT(sum);

    /* exp(-gamma * sum) */
    compute_t exponent = FIXED_MUL(model->kernel.gamma, sum_fx);

    /* Evitar saturación dura en exp */
    exponent = FIXED_MIN(exponent, FIX_EXP_MAX);

    return fixed_exp(FIXED_NEG(exponent));
}
