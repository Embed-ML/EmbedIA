/**
 * Function: svm_direct_classifier_layer
 * Lines: 204-223
 */

void svm_direct_classifier_layer(const svm_direct_classifier_layer_t *model,
                                 const data1d_t                      *input,
                                 data1d_t                            *output)
{
    compute_t *scores = alloc_output(output, model->n_classes);

    for (uint16_t i = 0; i < model->n_classes; i++) {
        const storage_t *w = model->coefs + (i * model->n_features);


        scores[i] = CX2CO_SAT(
            dot_product_bias(input->data, w, model->n_features, model->icepts[i], model->qp_coefs)
        );

    }
}