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
 * ========================================================================= */

#include <stdlib.h>
#include "common.h"
#include "svm_ovo.h"

/* =========================================================================
 * Kernel functions en punto fijo
 * ========================================================================= */

// @embedia-include svm/kernel_linear.c

// @embedia-include svm/kernel_rbf.c

// @embedia-include svm/kernel_poly.c

// @embedia-include svm/kernel_sigmoid.c

/* =========================================================================
 * svm_vote_sparse — OvO voting engine
 * ========================================================================= */

// @embedia-include svm/svm_vote_sparse.c

/* =========================================================================
 * Internal helper — allocate and zero-init the output buffer
 * ========================================================================= */

// @embedia-include svm/alloc_output.c

/* =========================================================================
 * Public API — OvO classifier layers
 * ========================================================================= */

// @embedia-include svm/svm_linear_classifier_layer.c

// @embedia-include svm/svm_rbf_classifier_layer.c

// @embedia-include svm/svm_poly_classifier_layer.c

// @embedia-include svm/svm_sigmoid_classifier_layer.c
