/**
 * @brief Computes the sigmoid kernel function between an input vector and a support vector (fixed-point).
 *
 * Applies the sigmoid kernel formula: tanh(gamma * dot(x, sv) + intercept)
 * where dot(x, sv) is the dot product of the input and support vector.
 * Uses high-precision fixed-point accumulation (dfixed) for dot product computation.
 *
 * @param[in] model Pointer to the SVM classifier model containing kernel parameters
 *                  (gamma and intercept).
 * @param[in] x Pointer to the input feature vector of size n_features (fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (fixed-point).
 *
 * @return The computed sigmoid kernel value between the input and support vector (fixed-point).
 *
 * @note Dot product is accumulated in dfixed for high precision, then rounded and saturated
 *       before final computation.
 */

static inline compute_t kernel_sigmoid(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    /* Mejora: dot en dfixed */
    dfixed dot_acc = 0;

    for (uint16_t i = 0; i < model->n_features; i++) {
        DFIXED_MAC(dot_acc, x[i], sv[i]);
    }

    compute_t dot = DFX2FX_RND_SAT(dot_acc);

    compute_t arg = FIXED_ADD(FIXED_MUL(model->kernel.gamma, dot), model->kernel.intercept);

    return fixed_tanh(arg);
}
