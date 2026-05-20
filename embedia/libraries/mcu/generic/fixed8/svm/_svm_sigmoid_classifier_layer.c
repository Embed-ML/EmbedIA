/**
 * @brief Performs SVM classification using sigmoid kernel with sparse representation (8-bit fixed-point).
 *
 * This function applies the sigmoid kernel SVM classifier to the input data and
 * produces classification votes for each class. The output is generated through
 * one-vs-one voting using sparse support vectors with 8-bit fixed-point arithmetic.
 *
 * @param[in] model Pointer to the SVM classifier model configured for sigmoid kernel.
 * @param[in] input Pointer to the input data structure containing feature vector (8-bit fixed-point)
 *                  and its length.
 * @param[out] output Pointer to the output data structure where classification votes
 *                    will be stored (one value per class, in 8-bit fixed-point format).
 *
 * @note The output array is allocated internally based on the number of classes.
 */

void svm_sigmoid_classifier_layer(const svm_classifier_layer_t *model,
                                  const data1d_t               *input,
                                  data1d_t                     *output)
{
    compute_t *votes = alloc_output(output, model->n_classes);
    svm_vote_sparse(model, (const compute_t *)input->data, votes, kernel_sigmoid);
}