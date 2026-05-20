#ifndef _SVM_H
#define _SVM_H
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
 * SVM classifier — One-vs-One (OvO) strategy
 *
 * This file implements multiclass SVM inference using the OvO strategy:
 * k*(k-1)/2 binary classifiers are trained, one per class pair. Prediction
 * is determined by majority voting across all pairwise decisions.
 *
 * Coefficients are stored in sparse format: for each pair (i,j) only the
 * support vectors with non-zero contribution are kept, saving memory when
 * the model is sparse (common in practice).
 *
 * For the OvR strategy, see svm_ovr.h / svm_ovr.c.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* Kernel type constants */
/* ------------------------------------------------------------------------- */

typedef uint8_t svm_kernel_type_t;
#define SVM_KERNEL_LINEAR   0
#define SVM_KERNEL_POLY     1
#define SVM_KERNEL_RBF      2
#define SVM_KERNEL_SIGMOID  3

/* ------------------------------------------------------------------------- */
/* Sparse structures */
/* ------------------------------------------------------------------------- */

typedef uint16_t vector_id_t;

typedef struct {
    vector_id_t idx;      /* índice en model->vectors */
    storage_t   coef;     /* coeficiente dual α (quant8) */
} svm_coef_sparse_t;

typedef struct {
    uint16_t                       count;
    const svm_coef_sparse_t *data;
} svm_pair_sparse_t;

/* ------------------------------------------------------------------------- */
/* Kernel configuration (en fixed) */
/* ------------------------------------------------------------------------- */

typedef struct {
    svm_kernel_type_t type;
    compute_t         gamma;      /* gamma fixed */
    compute_t         intercept;  /* intercept fixed */
    uint8_t           degree;
} svm_kernel_config_t;

/* ------------------------------------------------------------------------- */
/* Main SVM OvO model structure for fixed-point */
/* ------------------------------------------------------------------------- */

typedef struct {
    uint16_t                 n_classes;
    uint16_t                 n_features;
    uint16_t                 n_SV;
    uint16_t                 n_pairs;

    svm_kernel_config_t      kernel;

    const storage_t          *vectors;   /* support vectors [n_SV × n_features]  quant8 */
    const svm_pair_sparse_t  *pairs;     /* sparse coefficients */
    const compute_t          *icepts;    /* intercepts [n_pairs] en fixed */
#if DATA_TYPE_IMPL == DT_QUANT8
    qparam_t                 qp_vectors; /* qparam for support vectors */
    qparam_t                 qp_coefs;   /* qparam for support vectors */
#endif
} svm_classifier_layer_t;

/* -------------------------------------------------------------------------
 * Direct linear classifier layer — OvR  (equivalent to sklearn LinearSVC)
 *
 * Does not use support vectors. Stores the learned weight matrix directly:
 *
 *   score_i = icepts[i] + dot(coefs[i * n_features .. (i+1)*n_features], x)
 *
 * This representation is more memory-efficient than the kernel variant for
 * linear models because it avoids storing support vectors altogether.
 *
 * Coefficient layout:
 *   coefs[i * n_features + f] = weight of feature f for class i
 *
 * QUANT8 STORAGE:
 *   coefs are quantized int8 values; icepts are stored as compute_t (fixed16)
 * ---------------------------------------------------------------------- */

typedef struct {
    uint16_t        n_classes;   /* number of output classes                   */
    uint16_t        n_features;  /* input feature dimension                    */
    const storage_t *coefs;      /* weight matrix [n_classes × n_features]     */
    const compute_t *icepts;     /* bias vector   [n_classes] (fixed16)        */
#if DATA_TYPE_IMPL == DT_QUANT8
    qparam_t        qp_coefs;    /* quantization params for weights            */
#endif
} svm_direct_classifier_layer_t;


typedef svm_direct_classifier_layer_t svm_linear_classifier_layer_t;

/* ------------------------------------------------------------------------- */
/* Public API - Fixed point versions */
/* ------------------------------------------------------------------------- */

void svm_linear_classifier_layer(const svm_classifier_layer_t *model,
                                       const data1d_t                     *input,
                                       data1d_t                           *output);

void svm_rbf_classifier_layer(const svm_classifier_layer_t *model,
                                    const data1d_t                     *input,
                                    data1d_t                           *output);

void svm_poly_classifier_layer(const svm_classifier_layer_t *model,
                                     const data1d_t                     *input,
                                     data1d_t                           *output);

void svm_sigmoid_classifier_layer(const svm_classifier_layer_t *model,
                                        const data1d_t                     *input,
                                        data1d_t                           *output);


/**
 * @brief Linear SVM inference — OvR, direct weight-vector dot product.
 *
 * Computes score_i = dot(coefs[i], x) + icepts[i] for each class i.
 * No kernel evaluation, no support vectors.
 *
 * Equivalent to sklearn's LinearSVC.predict() and to any linear model
 * that stores learned weights directly (logistic regression, perceptron).
 *
 * @param[in]  model  Trained model with weight matrix and bias vector.
 * @param[in]  input  Feature vector; length must equal model->n_features.
 * @param[out] output Allocated by swap_alloc; length == model->n_classes.
 *                    Values are raw decision scores; use argmax to classify.
 */
void svm_direct_classifier_layer(const svm_direct_classifier_layer_t *model,
                                 const data1d_t                      *input,
                                 data1d_t                            *output);

#ifdef __cplusplus
}
#endif

#endif /* _SVM_H */