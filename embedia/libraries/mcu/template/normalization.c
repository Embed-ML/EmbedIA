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
 */

#include "normalization.h"

// ========================================================
// Normalization Functions Implementation
// ========================================================

/*
 * Normalization type 1 (mean and variance)
 * - Subtracts mean and divides by standard deviation
 * - Input preprocessing for better training stability
 * - Based on scikit-learn normalization techniques
 */

// Function: normalization1
// @embedia-include normalization/_normalization1.c


/*
 * Normalization type 2 (variance only)
 * - Scales by standard deviation without mean subtraction
 * - Used in certain network architectures
 * - Based on scikit-learn MaxAbsScaler
 */

// Function: normalization2
// @embedia-include normalization/_normalization2.c
