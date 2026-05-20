/**
 * @brief Computes class votes from sparse SVM support vectors using a kernel function (fixed16-point implementation).
 *
 * This function performs one-vs-one classification by computing decision values
 * for all class pairs and accumulating votes for each class based on the kernel
 * function result. Uses 16-bit fixed-point arithmetic for embedded systems.
 *
 * @param[in] model Pointer to the SVM classifier model containing kernel parameters,
 *                  support vectors, coefficients, and intercepts.
 * @param[in] input Pointer to the input feature vector of size n_features (16-bit fixed-point).
 * @param[out] votes Pointer to output votes array of size n_classes (16-bit fixed-point).
 *                   The function accumulates votes for each class.
 * @param[in] kernel_fn Function pointer to the kernel function to use for computing
 *                       decision values (e.g., kernel_linear, kernel_rbf, kernel_poly).
 */

static inline void svm_vote_sparse(
    const svm_classifier_layer_t *model,
    const compute_t              *input,
    compute_t                    *votes,
    compute_t (*kernel_fn)(const svm_classifier_layer_t*, const compute_t*, const compute_t*))
{
    const uint16_t n_classes = model->n_classes;
    uint16_t pair_idx = 0;

    for (uint16_t i = 0; i < n_classes; i++) {
        for (uint16_t j = i + 1; j < n_classes; j++) {

            storage_t decision = model->icepts[pair_idx];

            const svm_pair_sparse_t *pair = &model->pairs[pair_idx];

            for (uint16_t k = 0; k < pair->count; k++) {
                const compute_t *sv = model->vectors + (pair->data[k].idx * model->n_features);
                compute_t k_val = kernel_fn(model, input, sv);
                decision = FIXED_ADD(decision, FIXED_MUL(pair->data[k].coef, k_val));
            }

            if (decision > FIX_ZERO)
                votes[i] = FIXED_ADD(votes[i], FIX_ONE);
            else
                votes[j] = FIXED_ADD(votes[j], FIX_ONE);

            pair_idx++;
        }
    }
}
