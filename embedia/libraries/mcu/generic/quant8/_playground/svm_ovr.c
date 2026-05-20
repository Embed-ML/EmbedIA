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
 * QUANT8 IMPLEMENTATION NOTES:
 *   - Support vectors and coefficients are stored as int8 (storage_t)
 *   - Dequantization to fixed16 happens on-the-fly using qparam_t
 *   - Intercepts are stored directly as fixed16 for better precision
 *   - No additional RAM buffers are allocated
 */

#include <stdlib.h>

#include "common.h"
#include "svm_ovr.h"

/* =========================================================================
 * Internal helpers — dequantization and kernel functions
 *
 * All kernels dequantize support vectors on-the-fly to avoid RAM overhead.
 * The input feature vector is already in fixed16 format.
 * ====================================================================== */

/**
 * @brief Dequantize a single support vector element
 * @param qval Quantized int8 value
 * @param qp Quantization parameters
 * @return Dequantized fixed16 value
 */
static inline compute_t dequantize(storage_t qval, qparam_t qp) {
    return DEQUANTIZE_FIXED(qval, qp);
}

/**
 * @brief Linear kernel with on-the-fly dequantization
 */
static inline compute_t kernel_linear(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const storage_t              *sv_quant)
{
    compute_t acc = COMPUTE_ZERO;
    for (uint16_t i = 0; i < model->n_features; i++) {
        compute_t sv = dequantize(sv_quant[i], model->qp_vectors);
        acc = FIXED_ADD(acc, FIXED_MUL(x[i], sv));
    }
    return acc;
}

/**
 * @brief RBF kernel with on-the-fly dequantization
 *
 * K(x, sv) = exp(-gamma * ||x - sv||²)
 */
static inline compute_t kernel_rbf(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const storage_t              *sv_quant)
{
    compute_t sum = COMPUTE_ZERO;
    for (uint16_t i = 0; i < model->n_features; i++) {
        compute_t sv = dequantize(sv_quant[i], model->qp_vectors);
        compute_t d = FIXED_SUB(x[i], sv);
        sum = FIXED_ADD(sum, FIXED_MUL(d, d));
    }
    /* exp(-gamma * sum) using fixed_exp */
    compute_t arg = FIXED_MUL(model->kernel.gamma, sum);
    return fixed_exp(FIXED_NEG(arg));
}

/**
 * @brief Polynomial kernel with on-the-fly dequantization
 *
 * K(x, sv) = (gamma * <x,sv> + intercept)^degree
 */
static inline compute_t kernel_poly(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const storage_t              *sv_quant)
{
    compute_t dot = COMPUTE_ZERO;
    for (uint16_t i = 0; i < model->n_features; i++) {
        compute_t sv = dequantize(sv_quant[i], model->qp_vectors);
        dot = FIXED_ADD(dot, FIXED_MUL(x[i], sv));
    }

    compute_t term = FIXED_ADD(FIXED_MUL(model->kernel.gamma, dot),
                               model->kernel.intercept);

    /* Avoid fixed_pow() for common cases — measurably faster on targets
     * without an FPU. */
    switch (model->kernel.degree) {
        case 2:  return FIXED_MUL(term, term);
        case 3:  return FIXED_MUL(FIXED_MUL(term, term), term);
        default: return fixed_pow(term, INT_TO_FIXED(model->kernel.degree));
    }
}

/**
 * @brief Sigmoid kernel with on-the-fly dequantization
 *
 * K(x, sv) = tanh(gamma * <x,sv> + intercept)
 */
static inline compute_t kernel_sigmoid(
    const svm_classifier_layer_t *model,
    const compute_t              *x,
    const storage_t              *sv_quant)
{
    compute_t dot = COMPUTE_ZERO;
    for (uint16_t i = 0; i < model->n_features; i++) {
        compute_t sv = dequantize(sv_quant[i], model->qp_vectors);
        dot = FIXED_ADD(dot, FIXED_MUL(x[i], sv));
    }

    compute_t arg = FIXED_ADD(FIXED_MUL(model->kernel.gamma, dot),
                              model->kernel.intercept);
    /* tanh(...) using fixed_tanh */
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
 * Both coefficients and support vectors are dequantized on-the-fly.
 * Intercepts are already in fixed16 format.
 *
 * @param model     Trained OvR model (quantized storage)
 * @param input     Raw feature pointer (input->data from the caller)
 * @param scores    Output array of length n_classes, zero-init'd by caller
 * @param kernel_fn One of the four kernel_* functions above
 * ====================================================================== */

static inline void svm_score_ovr(
    const svm_classifier_layer_t *model,
    const compute_t              *input,
    compute_t                    *scores,
    compute_t (*kernel_fn)(const svm_classifier_layer_t*, const compute_t*, const storage_t*)
)
{
    const uint16_t n_classes  = model->n_classes;
    const uint16_t n_SV       = model->n_SV;
    const qparam_t qp_coefs   = model->qp_coefs;

    for (uint16_t i = 0; i < n_classes; i++) {
        /* Intercepts are already in fixed16 format */
        compute_t dec = model->icepts[i];

        /* Pointer to coefficient row for class i */
        const storage_t *coef_row = model->coefs + (i * n_SV);

        for (uint16_t j = 0; j < n_SV; j++) {
            /* Get support vector pointer (quantized) */
            const storage_t *sv = model->vectors + (j * model->n_features);

            /* Compute kernel value (dequantizes SV internally) */
            compute_t k_val = kernel_fn(model, input, sv);

            /* Dequantize coefficient and accumulate */
            compute_t coef = dequantize(coef_row[j], qp_coefs);
            dec = FIXED_ADD(dec, FIXED_MUL(coef, k_val));
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
                                 const data1d_t               *input,
                                 data1d_t                     *output)
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

/* =========================================================================
 * Public API — direct linear classifier (LinearSVC equivalent)
 *
 * No kernel evaluation, no support vectors. Each class has a weight vector
 * of length n_features stored densely in coefs[i * n_features ...].
 * Weights are dequantized on-the-fly; intercepts are fixed16.
 * ====================================================================== */

void svm_direct_classifier_layer(const svm_direct_classifier_layer_t *model,
                                 const data1d_t                      *input,
                                 data1d_t                            *output)
{
    compute_t *scores = alloc_output(output, model->n_classes);

    for (uint16_t i = 0; i < model->n_classes; i++) {
        const storage_t *w_quant = model->coefs + (i * model->n_features);
        compute_t bias = model->icepts[i];  /* Already fixed16 */

        /* Compute dot product with on-the-fly dequantization */
        computex_t acc = COMPUTEX_ZERO;
        for (uint16_t f = 0; f < model->n_features; f++) {
            compute_t w = dequantize(w_quant[f], model->qp_coefs);
            DFIXED_MAC(acc, w, input->data[f]);
        }

        scores[i] = FIXED_ADD(CX2CO(acc), bias);
    }
}