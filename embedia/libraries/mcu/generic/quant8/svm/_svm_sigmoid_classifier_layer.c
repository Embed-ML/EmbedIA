/**
 * Function: svm_sigmoid_classifier_layer
 * Lines: 188-194
 */

void svm_sigmoid_classifier_layer(const svm_classifier_layer_t *model,
                                  const data1d_t               *input,
                                  data1d_t                     *output)
{
    compute_t *votes = alloc_output(output, model->n_classes);
    svm_vote_sparse(model, (const compute_t*)input->data, votes, kernel_sigmoid);
}
