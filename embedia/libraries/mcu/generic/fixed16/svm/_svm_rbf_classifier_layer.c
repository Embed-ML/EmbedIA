/**
 * @brief Performs SVM classification using RBF (Radial Basis Function) kernel (16-bit fixed-point).
 *
 * This function applies the RBF kernel SVM classifier to the input data and
 * generates classification votes for each class. The classifier uses sparse
 * support vectors and one-vs-one voting strategy with 16-bit fixed-point arithmetic.
 *
 * @param[in] model Pointer to the SVM classifier model configured for RBF kernel.
 * @param[in] input Pointer to the input data structure containing feature vector (16-bit fixed-point)
 *                  and its length.
 * @param[out] output Pointer to the output data structure where classification votes
 *                    will be stored (one value per class, in 16-bit fixed-point format).
 *
 * @note The output array is allocated internally based on the number of classes.
 */

void svm_rbf_classifier_layer(const svm_classifier_layer_t *model,
                              const data1d_t               *input,
                              data1d_t                     *output)
{
    compute_t *votes = alloc_output(output, model->n_classes);
    svm_vote_sparse(model, (const compute_t*)input->data, votes, kernel_rbf);
}
