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
 * SVM classifier - One-vs-One (OvO) strategy - OPTIMIZED VERSION
 */

#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "svm_ovo.h"
#include "quant8.h"

/* =========================================================================
 * Internal helpers - optimized kernel functions
 * ========================================================================= */

// @embedia-include svm/kernel_linear.c

/* =========================================================================
 * OPTIMIZED RBF Kernel - reduced dequantization overhead
 * ========================================================================= */
// @embedia-include svm/kernel_rbf.c

/* =========================================================================
 * OPTIMIZED Poly Kernel - minimal RAM usage, fast power for degrees 2-3
 * ========================================================================= */
// @embedia-include svm/kernel_poly.c

// @embedia-include svm/kernel_sigmoid.c

/* =========================================================================
 * svm_vote_sparse - OvO voting engine (unchanged - already optimal)
 * ========================================================================= */
// @embedia-include svm/svm_vote_sparse.c

/* =========================================================================
 * Internal helper - allocate output buffer (unchanged)
 * ========================================================================= */
// @embedia-include svm/alloc_output.c

/* =========================================================================
 * Public API - OvO classifier layers (unchanged)
 * ========================================================================= */
// @embedia-include svm/svm_linear_classifier_layer.c

// @embedia-include svm/svm_rbf_classifier_layer.c

// @embedia-include svm/svm_poly_classifier_layer.c

// @embedia-include svm/svm_sigmoid_classifier_layer.c

/* =========================================================================
 * Public API — direct linear classifier (LinearSVC equivalent)
 *
 * No kernel evaluation, no support vectors. Each class has a weight vector
 * of length n_features stored densely in coefs[i * n_features ...].
 * Weights are dequantized on-the-fly; intercepts are fixed16.
 * ====================================================================== */

// @embedia-include svm/svm_direct_classifier_layer.c
