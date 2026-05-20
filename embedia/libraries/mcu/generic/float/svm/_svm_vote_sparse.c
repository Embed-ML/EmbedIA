/**
 * @brief Computes class votes from sparse SVM support vectors using a kernel function.
 *
 * This function performs one-vs-one classification by computing decision values
 * for all class pairs and accumulating votes for each class based on the kernel
 * function result.
 *
 * @param[in] model Pointer to the SVM classifier model containing kernel parameters,
 *                  support vectors, coefficients, and intercepts.
 * @param[in] input Pointer to the input feature vector of size n_features.
 * @param[out] votes Pointer to output votes array of size n_classes. The function
 *                   accumulates votes for each class.
 * @param[in] kernel_fn Function pointer to the kernel function to use for computing
 *                       decision values (e.g., kernel_linear, kernel_rbf, kernel_poly).
 */

static inline void svm_vote_sparse(
    const svm_classifier_layer_t *model,
    const compute_t              *input,
    compute_t                    *votes,
    compute_t (*kernel_fn)(const svm_classifier_layer_t*, const compute_t*, const compute_t*))
{
    const uint16_t            n_classes  = model->n_classes;
    const uint16_t            n_features = model->n_features;
    const svm_pair_sparse_t  *pairs      = model->pairs;

    uint16_t pair_idx = 0;

    for (uint16_t i = 0; i < n_classes; i++) {
        for (uint16_t j = i + 1; j < n_classes; j++) {

            storage_t decision = model->icepts[pair_idx];

            const svm_pair_sparse_t *pair      = &pairs[pair_idx];
            const svm_coef_sparse_t *coef_list = pair->data;

            for (uint16_t k = 0; k < pair->count; k++) {
                const real_t *sv = model->vectors + (coef_list[k].idx * n_features);
                decision += coef_list[k].coef * kernel_fn(model, input, sv);
            }

            if (decision > 0)
                votes[i] += 1;
            else
                votes[j] += 1;

            pair_idx++;
        }
    }
}
