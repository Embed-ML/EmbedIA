/**
 * @brief Computes the RBF (Radial Basis Function) kernel between an input vector and a support vector.
 *
 * Applies the RBF kernel formula: exp(-gamma * ||x - sv||^2)
 * which computes the squared Euclidean distance between the vectors.
 *
 * @param[in] model Pointer to the SVM classifier model containing the gamma kernel parameter.
 * @param[in] x Pointer to the input feature vector of size n_features.
 * @param[in] sv Pointer to the support vector of size n_features.
 *
 * @return The computed RBF kernel value between the input and support vector.
 */
static inline compute_t kernel_rbf(
    const svm_classifier_layer_t *model,
    const compute_t                 *x,
    const compute_t                 *sv)
{
    compute_t sum = 0;
    compute_t d;

    for (uint16_t i = 0; i < model->n_features; i++) {
        d = x[i] - sv[i];
        sum += d * d;
    }
    return expf(-model->kernel.gamma * sum);
}
