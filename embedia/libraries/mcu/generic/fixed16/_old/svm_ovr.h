#ifndef _SVM_OVR_H
#define _SVM_OVR_H
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
 * This file implements multiclass SVM inference using the OvR strategy:
 * k binary classifiers are trained, one per class, each separating that
 * class from all others. Prediction is the argmax of the k decision scores.
 *
 * Two variants are provided:
 *   svm_classifier_layer_t      — kernel SVM (RBF, poly, sigmoid, linear)
 *                                 uses explicit support vectors
 *   svm_direct_classifier_layer_t — linear SVM without support vectors
 *                                   (equivalent to sklearn's LinearSVC)
 *
 * For the OvO strategy, see svm_ovo.h / svm_ovo.c.
 */

#include "common.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Kernel type constants
 * ---------------------------------------------------------------------- */

typedef uint8_t svm_kernel_type_t;
#define SVM_KERNEL_LINEAR   0
#define SVM_KERNEL_POLY     1
#define SVM_KERNEL_RBF      2
#define SVM_KERNEL_SIGMOID  3

/* -------------------------------------------------------------------------
 * Kernel configuration
 * ---------------------------------------------------------------------- */

typedef struct {
    svm_kernel_type_t type;      /* kernel variant                          */
    storage_t         gamma;     /* scale factor (poly, RBF, sigmoid)       */
    storage_t         intercept; /* bias term    (poly, sigmoid)            */
    uint8_t           degree;    /* exponent     (poly only)                */
} svm_kernel_config_t;

/* -------------------------------------------------------------------------
 * Kernel SVM classifier layer — OvR
 *
 * Stores k sets of dual coefficients (one per class) over the shared
 * support vector matrix. For each class i, the decision score is:
 *
 *   score_i = icepts[i] + sum_j( coefs[i*n_SV + j] * K(x, vectors[j]) )
 *
 * Coefficient layout:
 *   coefs[i * n_SV + j] = dual coefficient of support vector j
 *                         in the binary classifier for class i
 *                         (zero if sv j does not participate in class i)
 *
 * Prediction: argmax over scores[0..n_classes-1]
 * ---------------------------------------------------------------------- */

typedef struct {
    uint16_t             n_classes;  /* number of output classes            */
    uint16_t             n_features; /* input feature dimension             */
    uint16_t             n_SV;       /* total number of support vectors     */

    svm_kernel_config_t  kernel;

    const storage_t     *vectors;    /* support vectors [n_SV × n_features] */
    const storage_t     *coefs;      /* dual coefs      [n_classes × n_SV]  */
    const storage_t     *icepts;     /* intercepts      [n_classes]         */

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
 * Prediction: argmax over scores[0..n_classes-1]
 * ---------------------------------------------------------------------- */

typedef struct {
    uint16_t        n_classes;  /* number of output classes                   */
    uint16_t        n_features; /* input feature dimension                    */
    const storage_t *coefs;      /* weight matrix [n_classes × n_features]     */
    const storage_t *icepts;     /* bias vector   [n_classes]                  */
} svm_direct_classifier_layer_t;

/* -------------------------------------------------------------------------
 * Public API — kernel SVM, OvR
 * ---------------------------------------------------------------------- */

/**
 * @brief Multiclass SVM inference — linear kernel, OvR.
 *
 * Computes score_i = icepts[i] + sum_j( coefs[i,j] * dot(x, sv_j) )
 * for each class i and returns raw decision scores in output->data.
 *
 * @param[in]  model  Trained OvR model with support vectors and dual coefs.
 * @param[in]  input  Feature vector; length must equal model->n_features.
 * @param[out] output Allocated by swap_alloc; length == model->n_classes.
 *                    Values are raw decision scores; use argmax to classify.
 */
void svm_linear_classifier_layer(const svm_classifier_layer_t *model,
                                 const data1d_t               *input,
                                 data1d_t                     *output);

/**
 * @brief Multiclass SVM inference — RBF kernel, OvR.
 *
 * Uses exp(-gamma * ||x - sv||²) as the kernel function.
 * See svm_linear_classifier_layer() for parameter details.
 */
void svm_rbf_classifier_layer(const svm_classifier_layer_t *model,
                               const data1d_t               *input,
                               data1d_t                     *output);

/**
 * @brief Multiclass SVM inference — polynomial kernel, OvR.
 *
 * Uses (gamma * <x,sv> + intercept)^degree as the kernel function.
 * Degrees 2 and 3 are computed without powf() for speed.
 * See svm_linear_classifier_layer() for parameter details.
 */
void svm_poly_classifier_layer(const svm_classifier_layer_t *model,
                               const data1d_t               *input,
                               data1d_t                     *output);

/**
 * @brief Multiclass SVM inference — sigmoid kernel, OvR.
 *
 * Uses tanh(gamma * <x,sv> + intercept) as the kernel function.
 * See svm_linear_classifier_layer() for parameter details.
 */
void svm_sigmoid_classifier_layer(const svm_classifier_layer_t *model,
                                  const data1d_t               *input,
                                  data1d_t                     *output);



#ifdef __cplusplus
}
#endif

#endif /* _SVM_OVR_H */