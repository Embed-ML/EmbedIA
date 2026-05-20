/**
 * Function: svm_linear_classifier_layer
 * Lines: 164-170
 */

void svm_linear_classifier_layer(const svm_classifier_layer_t *model,
                                 const data1d_t               *input,
                                 data1d_t                     *output)
{
    compute_t *votes = alloc_output(output, model->n_classes);
    svm_vote_sparse(model, (const compute_t*)input->data, votes, kernel_linear);
}
