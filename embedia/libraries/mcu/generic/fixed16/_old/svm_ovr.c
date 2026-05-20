/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 * GitHub: https://github.com/Embed-ML/EmbedIA
 *
 * SVM classifier — One-vs-Rest (OvR) strategy
 *
 */

#include <stdlib.h>

#include "common.h"
#include "svm_ovr.h"

/* =========================================================================
 * Internal helpers — kernel functions
 *
 * All kernels share the same signature so they can be passed as function
 * pointers to svm_score_ovr(), avoiding code duplication across the four
 * public entry points.
 *
 * The model pointer is included in the signature so that gamma, degree and
 * intercept can be accessed without extra parameters. The compiler inlines
 * these field accesses when optimisation is enabled.
 * ====================================================================== */

static inline compute_t kernel_linear(
    const svm_classifier_layer_t *model,
    const compute_t                     *x,
    const compute_t                     *sv)
{
    compute_t acc = COMPUTE_ZERO;
    for (uint16_t i = 0; i < model->n_features; i++)
        acc = FIXED_ADD(acc, FIXED_MUL(x[i], sv[i]));
    return acc;
}

static inline compute_t kernel_rbf(
    const svm_classifier_layer_t *model,
    const compute_t                     *x,
    const compute_t                     *sv)
{
    compute_t sum = COMPUTE_ZERO;
    for (uint16_t i = 0; i < model->n_features; i++) {
        compute_t d = FIXED_SUB(x[i], sv[i]);
        sum = FIXED_ADD(sum, FIXED_MUL(d, d));
    }
    /* exp(-gamma * sum) usando fixed_exp */
    compute_t arg = FIXED_MUL(model->kernel.gamma, sum);
    return fixed_exp(FIXED_NEG(arg));
}

static inline compute_t kernel_poly(
    const svm_classifier_layer_t *model,
    const compute_t                     *x,
    const compute_t                     *sv)
{
    compute_t dot = COMPUTE_ZERO;
    for (uint16_t i = 0; i < model->n_features; i++)
        dot = FIXED_ADD(dot, FIXED_MUL(x[i], sv[i]));

    compute_t term = FIXED_ADD(FIXED_MUL(model->kernel.gamma, dot),
                               model->kernel.intercept);

    /* Avoid fixed_pow() for the common cases — measurably faster on targets
     * without an FPU.                                                       */
    switch (model->kernel.degree) {
        case 2:  return FIXED_MUL(term, term);
        case 3:  return FIXED_MUL(FIXED_MUL(term, term), term);
        default: return fixed_pow(term, INT_TO_FIXED(model->kernel.degree));
    }
}

static inline compute_t kernel_sigmoid(
    const svm_classifier_layer_t *model,
    const compute_t                     *x,
    const compute_t                     *sv)
{
    compute_t dot = COMPUTE_ZERO;
    for (uint16_t i = 0; i < model->n_features; i++)
        dot = FIXED_ADD(dot, FIXED_MUL(x[i], sv[i]));

    compute_t arg = FIXED_ADD(FIXED_MUL(model->kernel.gamma, dot),
                              model->kernel.intercept);
    /* tanh(...) usando fixed_tanh */
    return fixed_tanh(arg);
}

/* =========================================================================
 * svm_score_ovr — OvR scoring engine
 *
 * For each class i, computes:
 *
 *   scores[i] = icepts[i] + sum_j( coefs[i*n_SV + j] * K(x, vectors[j]) )
 *
 * The kernel function is passed as a pointer so the four public entry
 * points can share this implementation without branching on kernel type
 * at runtime inside the inner loop.
 *
 * @param model     Trained OvR model.
 * @param input     Raw feature pointer (input->data from the caller).
 * @param scores    Output array of length n_classes, zero-init'd by caller.
 * @param kernel_fn One of the four kernel_* functions above.
 * ====================================================================== */

static inline void svm_score_ovr(
    const svm_classifier_layer_t *model,
    const compute_t                     *input,
    compute_t                           *scores,
    compute_t (*kernel_fn)(const svm_classifier_layer_t*, const compute_t*, const compute_t*)
)
{
    const uint16_t n_classes  = model->n_classes;
    const uint16_t n_features = model->n_features;
    const uint16_t n_SV       = model->n_SV;

    for (uint16_t i = 0; i < n_classes; i++) {
        compute_t dec = model->icepts[i];

        const compute_t *row = model->coefs + (i * n_SV);

        for (uint16_t j = 0; j < n_SV; j++) {
            const compute_t *sv = model->vectors + (j * n_features);
            compute_t k_val = kernel_fn(model, input, sv);
            dec = FIXED_ADD(dec, FIXED_MUL(row[j], k_val));
        }

        scores[i] = dec;
    }
}

/* =========================================================================
 * Internal helper — allocate and zero-init the output buffer
 * ====================================================================== */

static inline compute_t *alloc_output(data1d_t *output, uint16_t n_classes)
{
    compute_t *buf = (compute_t *)swap_alloc(n_classes * sizeof(compute_t));
    output->data   = buf;
    output->length = n_classes;
    for (uint16_t i = 0; i < n_classes; i++)
        buf[i] = COMPUTE_ZERO;
    return buf;
}

/* =========================================================================
 * Public API — kernel SVM, OvR
 * ====================================================================== */

void svm_linear_classifier_layer(const svm_classifier_layer_t *model,
                                 const data1d_t                      *input,
                                 data1d_t                            *output)
{
    compute_t *scores = alloc_output(output, model->n_classes);
    svm_score_ovr(model, input->data, scores, kernel_linear);
}

void svm_rbf_classifier_layer(const svm_classifier_layer_t *model,
                               const data1d_t               *input,
                               data1d_t                     *output)
{
    compute_t *scores = alloc_output(output, model->n_classes);
    svm_score_ovr(model, input->data, scores, kernel_rbf);
}

void svm_poly_classifier_layer(const svm_classifier_layer_t *model,
                               const data1d_t               *input,
                               data1d_t                     *output)
{
    compute_t *scores = alloc_output(output, model->n_classes);
    svm_score_ovr(model, input->data, scores, kernel_poly);
}

void svm_sigmoid_classifier_layer(const svm_classifier_layer_t *model,
                                  const data1d_t               *input,
                                  data1d_t                     *output)
{
    compute_t *scores = alloc_output(output, model->n_classes);
    svm_score_ovr(model, input->data, scores, kernel_sigmoid);
}
