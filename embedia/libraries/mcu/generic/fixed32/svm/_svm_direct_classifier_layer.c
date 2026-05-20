/**
 * @brief Performs direct linear SVM classification without sparse representation.
 *
 * This function applies a direct linear classifier to the input data, computing
 * classification scores directly from the coefficients and intercepts without
 * using support vectors or kernel functions. Each class score is computed as
 * a dot product of weights and input features plus a bias term.
 *
 * @param[in] model Pointer to the direct SVM classifier model containing dense
 *                  coefficients and intercepts for each class.
 * @param[in] input Pointer to the input data structure containing feature vector
 *                  and its length.
 * @param[out] output Pointer to the output data structure where classification scores
 *                    will be stored (one score per class).
 *
 * @note The output array is allocated internally based on the number of classes.
 */
void svm_direct_classifier_layer(const svm_direct_classifier_layer_t *model,
                                 const data1d_t                      *input,
                                 data1d_t                            *output)
{
    real_t *scores = alloc_output(output, model->n_classes);

    for (uint16_t i = 0; i < model->n_classes; i++) {
        const real_t *w = model->coefs + (i * model->n_features);
        const computex_t temp = dot_product_bias(w, input->data,
                                     model->n_features,
                                     model->icepts[i]);
        scores[i] = CX2CO_SAT(temp);

    }
}
