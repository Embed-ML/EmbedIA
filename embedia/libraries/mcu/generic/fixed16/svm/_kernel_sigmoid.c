/**
 * @brief Computes the sigmoid kernel function between an input vector and a support vector (16-bit fixed-point).
 *
 * Applies the sigmoid kernel formula: tanh(gamma * dot(x, sv) + intercept)
 * where dot(x, sv) is the dot product of the input and support vector.
 * Uses 16-bit fixed-point arithmetic for embedded systems.
 *
 * @param[in] model Pointer to the SVM classifier model containing kernel parameters
 *                  (gamma and intercept).
 * @param[in] x Pointer to the input feature vector of size n_features (16-bit fixed-point).
 * @param[in] sv Pointer to the support vector of size n_features (16-bit fixed-point).
 *
 * @return The computed sigmoid kernel value between the input and support vector (16-bit fixed-point).
 */

static inline compute_t kernel_sigmoid(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    compute_t dot = FIX_ZERO;
    compute_t arg;
    for (uint16_t i = 0; i < model->n_features; i++) {
        dot = FIXED_ADD(dot, FIXED_MUL(x[i], sv[i]));
    }
    arg = FIXED_ADD(FIXED_MUL(model->kernel.gamma, dot), model->kernel.intercept);
    return fixed_tanh(arg);
}
