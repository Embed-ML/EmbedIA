/**
 * @brief Computes the sigmoid kernel function between an input vector and a support vector.
 *
 * Applies the sigmoid kernel formula: tanh(gamma * dot(x, sv) + intercept)
 * where dot(x, sv) is the dot product of the input and support vector.
 *
 * @param[in] model Pointer to the SVM classifier model containing kernel parameters
 *                  (gamma and intercept).
 * @param[in] x Pointer to the input feature vector of size n_features.
 * @param[in] sv Pointer to the support vector of size n_features.
 *
 * @return The computed sigmoid kernel value between the input and support vector.
 */
static inline compute_t kernel_sigmoid(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const compute_t              *sv)
{
    compute_t dot = 0;
    for (uint16_t i = 0; i < model->n_features; i++) {
        dot += x[i] * sv[i];
    }
    return tanhf(model->kernel.gamma * dot + model->kernel.intercept);
}
