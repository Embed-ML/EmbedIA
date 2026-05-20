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
 * Fixed-point 8-bit implementation.
 */

#include <stdlib.h>

#include "common.h"
#include "fixed.h"
#include "svm_ovo.h"

/* =========================================================================
 * Precision note
 *
 * fixed  = int8_t,  Q(INT).FRC  where FRC = FIX_FRC_SZ  (e.g. Q4.4)
 * dfixed = int16_t, scale 2^(2*FRC)                      (e.g. Q8.8)
 *
 * For kernel_linear the dot product x·sv accumulates n_features terms of
 * (fixed * fixed).  With int8 operands the raw product fits in int16 but
 * with many features the int16 accumulator can overflow.  We therefore use
 * an int32_t accumulator throughout (same approach as DDFIXED_MAC).
 *
 * For kernel_rbf and kernel_sigmoid the intermediate values (squared
 * distances, dot products) are even more likely to overflow int16, so
 * int32_t accumulators are mandatory.
 *
 * Scaling:
 *   raw product of two Q(n).FRC values  →  scale 2^(2*FRC)
 *   after >> FRC                         →  scale 2^FRC  (= fixed scale)
 * ====================================================================== */

/* =========================================================================
 * kernel_linear
 *
 *   result = x · sv   (dot product)
 *
 * Accumulation in int32_t to avoid overflow with many features.
 * Scale after accumulation: 2^(2*FRC).
 * Shift right by FRC to bring back to fixed scale 2^FRC.
 * ====================================================================== */
// @embedia-include svm/kernel_linear.c

/* =========================================================================
 * kernel_rbf
 *
 *   result = exp( -gamma * ||x - sv||² )
 *
 * Step 1: compute squared distance in int32_t.
 *         d = x[i] - sv[i]  (computed in int16 to avoid int8 wrap-around)
 *         sum += d * d       (scale 2^(2*FRC), accumulated in int32)
 *
 * Step 2: compute exponent = gamma * sum WITHOUT reducing sum to fixed
 *         scale first.  Reducing sum to Q4.4 before multiplying by gamma
 *         loses precision when gamma is small (e.g. gamma=0.0625 and
 *         sum_fx=0.625 gives 0.039, which rounds to 0 in Q4.4).
 *
 *         Instead, keep sum at scale 2^(2*FRC) and multiply by gamma
 *         (scale 2^FRC) directly:
 *           exponent32 = gamma * sum   →  scale 2^(3*FRC)
 *         Then shift right by 2*FRC to bring back to fixed scale 2^FRC.
 *
 * Step 3: result = fixed_exp( -exponent )
 * ====================================================================== */
// @embedia-include svm/kernel_rbf.c

/* =========================================================================
 * kernel_poly
 *
 *   result = (gamma * <x,sv> + intercept) ^ degree
 *
 * Dot product accumulated in int32_t (same reasoning as kernel_linear).
 * ====================================================================== */
// @embedia-include svm/kernel_poly.c

/* =========================================================================
 * kernel_sigmoid
 *
 *   result = tanh( gamma * <x,sv> + intercept )
 *
 * Dot product accumulated in int32_t (same reasoning as kernel_linear).
 * ====================================================================== */
// @embedia-include svm/kernel_sigmoid.c

/* =========================================================================
 * svm_vote_sparse — OvO voting engine
 *
 * decision = icept + sum_k( coef_k * K(x, sv_k) )
 *
 * Both icept and coef_k are fixed scale (int8).
 * K() returns fixed scale (int8).
 * coef_k * K() is therefore at scale 2^(2*FRC) — accumulated in int32_t
 * to avoid overflow across many support vectors.
 * icept is promoted to the same scale before accumulation (<<FRC).
 * Final decision is brought back to fixed scale (>>FRC + saturate).
 * ====================================================================== */
static inline void svm_vote_sparse(
    const svm_classifier_layer_t *model,
    const compute_t              *input,
    compute_t                    *votes,
    compute_t (*kernel_fn)(const svm_classifier_layer_t*,
                           const compute_t*, const compute_t*))
{
    const uint16_t n_classes  = model->n_classes;
    const uint16_t n_features = model->n_features;
    uint16_t pair_idx = 0;

    for (uint16_t i = 0; i < n_classes; i++) {
        for (uint16_t j = i + 1; j < n_classes; j++) {

            /* Promote intercept from fixed scale (2^FRC) to product scale
             * (2^(2*FRC)) so it can be added to coef*kval terms directly. */
            int32_t decision = (int32_t)model->icepts[pair_idx] << FIX_FRC_SZ;

            const svm_pair_sparse_t *pair = &model->pairs[pair_idx];

            for (uint16_t k = 0; k < pair->count; k++) {
                const compute_t *sv =
                    model->vectors + (pair->data[k].idx * n_features);

                /* kernel result is at fixed scale (int8) */
                compute_t kval = kernel_fn(model, input, sv);

                /* coef (fixed) * kval (fixed) → scale 2^(2*FRC), int32 */
                decision += (int32_t)pair->data[k].coef * (int32_t)kval;
            }

            /* Bring decision back to fixed scale: >> FRC with rounding. */
            decision += (int32_t)1 << (FIX_FRC_SZ - 1);
            decision >>= FIX_FRC_SZ;

            /* Only the sign matters for voting — no need to saturate. */
            if (decision > 0)
                votes[i] = FIXED_ADD(votes[i], FIX_ONE);
            else
                votes[j] = FIXED_ADD(votes[j], FIX_ONE);

            pair_idx++;
        }
    }
}

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
