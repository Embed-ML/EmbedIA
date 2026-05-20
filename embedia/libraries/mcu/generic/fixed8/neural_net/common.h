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

/**
 * @file common.h
 * @brief Memory management and optimized math operations for embedded ML and neural networks
 *
 * This module provides:
 * - Deterministic memory allocation using a double-buffer system
 * - Optimized mathematical functions for microcontrollers (when available on MCU)
 * - Core utilities for embedded signal processing and neural network inference
 *
 * @note Dependencies:
 * - stdint.h: required for standard integer types
 * - math.h: required only if FPU-based operations are enabled
 * - common.h: project-specific core definitions (required)
 */

#ifndef _COMMON_H
#define _COMMON_H

// Detects compiler and define EMBEDIA_INLINE
#if defined(__GNUC__) || defined(__clang__) || defined(__ARMCC_VERSION) || defined(__IAR_SYSTEMS_ICC__)
    #if defined(__IAR_SYSTEMS_ICC__)
        #define EMBEDIA_INLINE _Pragma("inline=forced") static inline
    #elif defined(__ARMCC_VERSION) && (__ARMCC_VERSION < 6000000)
        #define EMBEDIA_INLINE __inline __attribute__((always_inline)) static
    #else
        #define EMBEDIA_INLINE __attribute__((always_inline)) static inline
    #endif
#else
    #define EMBEDIA_INLINE static inline
#endif

// Generic warning Message
#if defined(__GNUC__) || defined(__clang__)
    #define WARN_MSG(txt) _Pragma("GCC warning \"" #txt "\"")
#else
    #define WARN_MSG(txt) _Pragma("message(\"WARNING: \" #txt)")
#endif

#include <stdlib.h>
#include <stdint.h>
#include "fixed.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct data3d_t
 * @brief 3D data container for volumetric or multi-channel 2D data
 * @details Typically used for RGB images, feature maps, or multi-channel signals
 *
 * - channels: Number of channels/features (e.g., 3 for RGB, 64 for feature maps)
 * - width: Horizontal dimension (e.g., image width, time steps)
 * - height: Vertical dimension (e.g., image height, frequency bins)
 * - data: Flattened array in [channels, height, width] order
 */
typedef struct {
    uint16_t channels;  /**< Number of channels/features */
    uint16_t width;     /**< Horizontal dimension (width/time steps) */
    uint16_t height;    /**< Vertical dimension (height/frequency bins) */
    fixed * data;       /**< Flattened data array in CHW order */
} data3d_t;

/**
 * @struct data2d_t
 * @brief 2D data structure with multiple semantic interpretations
 * @details Dual-purpose container for different data types:
 *
 * - For images: use width and height
 * - For 1D signals: use width (time steps) and channels (features)
 *
 * @note height and channels share the same memory location
 */
typedef struct {
    uint16_t width;     /**< Primary dimension (width/time steps/length) */
    union {
        uint16_t height;    /**< For images: vertical dimension */
        uint16_t channels;  /**< For signals: number of channels/features */
    };
    fixed * data;       /**< Flattened data array */
} data2d_t;

/**
 * @struct data1d_t
 * @brief 1D data container for vectors and sequential data
 * @details Used for flattened arrays, feature vectors, or time series
 *
 * - length: Total number of elements in the vector
 * - data: Contiguous array of fixed-point values
 */
typedef struct {
    uint32_t length;    /**< Total number of elements in the vector */
    fixed * data;       /**< Contiguous array of values */
} data1d_t;

/**
 * @struct size2d_t
 * @brief 2D size specification for kernels, strides, and pooling operations
 * @details Commonly used for convolutional layer parameters
 *
 * - h: Vertical dimension (height/kernel rows)
 * - w: Horizontal dimension (width/kernel columns)
 */
typedef struct {
    uint16_t h;         /**< Vertical dimension (height/rows) */
    uint16_t w;         /**< Horizontal dimension (width/columns) */
} size2d_t;


/**
 * @brief Initializes the double buffer system to a known state
 *
 * Resets the buffer allocation system before starting a new processing sequence.
 * Must be called before the first use of swap_alloc() or when restarting a pipeline.
 *
 * @note
 * - Resets buffer index to MAX_BUFFER-1 (next allocation uses buffer 0)
 * - Does not clear memory contents (data persists until overwritten)
 * - No dynamic allocation — only state reset
 * - Safe to call multiple times
 *
 * @see swap_alloc()
 */
void prepare_buffers();

/**
 * @brief Allocates memory using double buffer system
 * @param size Number of bytes to allocate
 * @return Pointer to memory, or NULL if failed
 * @see prepare_buffers
 */
void * swap_alloc(uint32_t s);


/**
 * @brief Finds the index of the maximum value in a 1D data vector
 *
 * This function is commonly used in machine learning inference to determine
 * the predicted class by finding the position of the highest probability
 * in the output layer (e.g., after softmax).
 *
 * @param data Input 1D data vector of type `data1d_t`
 * @return     Index (position) of the maximum value in the vector
 *
 * @note
 * - If multiple elements have the same maximum value, returns the **first occurrence**.
 * - The input vector must have `length > 0`. For empty vectors, behavior is undefined.
 *
 * @example
 * data1d_t output = { .length = 3, .data = (float[]){0.1f, 0.7f, 0.2f} };
 * uint32_t predicted_class = argmax(output); // Returns 1
 */
uint32_t argmax(data1d_t data);


/**
 * @brief Computes the dot product of two fixed-point arrays with bias addition
 *
 * Computes: sum(a[i] * b[i]) + bias using fixed-point arithmetic.
 * Designed for neural network layers where bias is added after weighted sum.
 *
 * @param weights  First array (e.g., neuron weights)
 * @param input    Second array (e.g., input values)
 * @param length   Number of elements in both arrays
 * @param bias     Bias value to add (in fixed-point format)
 * @return         Sum of products plus bias, in doble fixed-point format
 *
 * @note Uses dfixed accumulator to prevent overflow during accumulation.
 */
#include <stdio.h>
EMBEDIA_INLINE dfixed dot_product_bias(
    const fixed* weights,
    const fixed* input,
    uint32_t length,
    fixed bias
) {
    dfixed result = FIXED_TO_DFIXED(bias);
    for (uint32_t i = 0; i < length; i++) {
        result += DFIXED_MUL(weights[i], input[i]);
    }
    return result;
}


/**
 * @brief Computes the dot product of two float arrays
 *
 * @param a   First array
 * @param b   Second array
 * @param len Number of elements
 * @return Dot product (sum of a[i]*b[i])
 */
#define dot_product(a, b, len) dot_product_bias(a, b, len, 0)


#ifdef __cplusplus
}
#endif

#endif