/**
 * @brief Performs SVM classification using linear kernel.
 *
 * This function applies the linear kernel SVM classifier to the input data and
 * generates classification votes for each class. The classifier uses sparse
 * support vectors and one-vs-one voting strategy.
 *
 * @param[in] model Pointer to the SVM classifier model configured for linear kernel.
 * @param[in] input Pointer to the input data structure containing feature vector
 *                  and its length.
 * @param[out] output Pointer to the output data structure where classification votes
 *                    will be stored (one value per class).
 *
 * @note The output array is allocated internally based on the number of classes.
 */

void svm_linear_classifier_layer(const svm_classifier_layer_t *model,
                                 const data1d_t               *input,
                                 data1d_t                     *output)
{
    compute_t *votes = alloc_output(output, model->n_classes);
    svm_vote_sparse(model, (const compute_t*)input->data, votes, kernel_linear);
}
