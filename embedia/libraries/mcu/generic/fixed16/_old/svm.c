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
 */

#include <stdlib.h>
#include <math.h>

#include "common.h"
#include "svm_ovo.h"

/* =========================================================================
 * Internal helpers — kernel functions
 *
 * All kernels share the same signature so they can be passed as function
 * pointers to svm_vote_sparse(), avoiding code duplication across the four
 * public entry points.
 *
 * The model pointer is included in the signature so that gamma, degree and
 * intercept can be accessed without extra parameters. The compiler inlines
 * these field accesses when optimisation is enabled.
 * ====================================================================== */

#include <stdlib.h>
#include "common.h"
#include "svm_ovo.h"           /* versión fixed */

/* =========================================================================
 * Kernel functions en punto fijo
 * ========================================================================= */

// @embedia-include svm/kernel_linear.c

// @embedia-include svm/kernel_rbf.c

// @embedia-include svm/kernel_poly.c

// @embedia-include svm/kernel_sigmoid.c

/* =========================================================================
 * svm_vote_sparse — OvO voting engine
 *
 * Iterates over all n_pairs = n_classes*(n_classes-1)/2 binary classifiers.
 * For each pair (i,j) only the support vectors with non-zero coefficients
 * are evaluated (sparse layout), so the inner loop is shorter than a dense
 * evaluation over all n_SV vectors.
 *
 * Each binary decision increments the vote counter of the winning class
 * by 1. The output is raw vote counts — use argmax to get the predicted
 * class. No normalisation is applied.
 *
 * @param model      Trained OvO model.
 * @param input      Raw feature pointer (input->data from the caller).
 * @param votes      Accumulator array of length n_classes, zero-init'd
 *                   by the caller.
 * @param kernel_fn  One of the four kernel_* functions above.
 * ====================================================================== */

// @embedia-include svm/svm_vote_sparse.c

/* =========================================================================
 * Internal helper — allocate and zero-init the output buffer
 * ====================================================================== */

// @embedia-include svm/alloc_output.c

/* =========================================================================
 * Public API — OvO classifier layers
 * ====================================================================== */

// @embedia-include svm/svm_linear_classifier_layer.c

// @embedia-include svm/svm_rbf_classifier_layer.c

// @embedia-include svm/svm_poly_classifier_layer.c

// @embedia-include svm/svm_sigmoid_classifier_layer.c
