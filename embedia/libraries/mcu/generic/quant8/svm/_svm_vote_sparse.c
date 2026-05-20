/**
 * Function: svm_vote_sparse
 * Lines: 116-146
 */

static inline void svm_vote_sparse(
    const svm_classifier_layer_t *model,
    const compute_t              *input,
    compute_t                    *votes,
    compute_t (*kernel_fn)(const svm_classifier_layer_t*, const compute_t*, const storage_t*))
{
    const uint16_t n_classes = model->n_classes;
    uint16_t pair_idx = 0;

    for (uint16_t i = 0; i < n_classes; i++) {
        for (uint16_t j = i + 1; j < n_classes; j++) {

            compute_t decision = model->icepts[pair_idx];
            const svm_pair_sparse_t *pair = &model->pairs[pair_idx];

            for (uint16_t k = 0; k < pair->count; k++) {
                const storage_t *sv = model->vectors + (pair->data[k].idx * model->n_features);
                compute_t k_val = kernel_fn(model, input, sv);
                compute_t coef_deq = DEQUANTIZE_FIXED(pair->data[k].coef, model->qp_coefs);
                decision = FIXED_ADD(decision, FIXED_MUL(coef_deq, k_val));
            }

            if (decision > FIX_ZERO)
                votes[i] = FIXED_ADD(votes[i], FIX_ONE);
            else
                votes[j] = FIXED_ADD(votes[j], FIX_ONE);

            pair_idx++;
        }
    }
}
