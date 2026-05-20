static inline void svm_vote_sparse(
    const svm_classifier_layer_t *model,
    const compute_t              *input,
    compute_t                    *votes,
    compute_t (*kernel_fn)(const svm_classifier_layer_t*,
                           const compute_t*, const compute_t*))
{
    const uint16_t n_classes  = model->n_classes;
    const uint16_t n_features = model->n_features;
    uint16_t pair_idx = 0;

    for (uint16_t i = 0; i < n_classes; i++) {
        for (uint16_t j = i + 1; j < n_classes; j++) {

            /* Promote intercept from fixed scale (2^FRC) to product scale
             * (2^(2*FRC)) so it can be added to coef*kval terms directly. */
            int32_t decision = (int32_t)model->icepts[pair_idx] << FIX_FRC_SZ;

            const svm_pair_sparse_t *pair = &model->pairs[pair_idx];

            for (uint16_t k = 0; k < pair->count; k++) {
                const compute_t *sv =
                    model->vectors + (pair->data[k].idx * n_features);

                /* kernel result is at fixed scale (int8) */
                compute_t kval = kernel_fn(model, input, sv);

                /* coef (fixed) * kval (fixed) → scale 2^(2*FRC), int32 */
                decision += (int32_t)pair->data[k].coef * (int32_t)kval;
            }

            /* Bring decision back to fixed scale: >> FRC with rounding. */
            decision += (int32_t)1 << (FIX_FRC_SZ - 1);
            decision >>= FIX_FRC_SZ;

            /* Only the sign matters for voting — no need to saturate. */
            if (decision > 0)
                votes[i] = FIXED_ADD(votes[i], FIX_ONE);
            else
                votes[j] = FIXED_ADD(votes[j], FIX_ONE);

            pair_idx++;
        }
    }
}