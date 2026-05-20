#ifndef _COMMON_H
#define _COMMON_H
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

/* EmbedIA implementation Data Types */
#define DT_FLOAT 1
#define DT_FIXED32 2
#define DT_FIXED16 3
#define DT_FIXED8 4
#define DT_QUANT8 5

/**
 * @brief Storage qualifier for model weights and parameters.
 *
 * Defines whether model weights are stored in FLASH (const) or RAM.
 *
 * Options:
 *   const    → weights in FLASH (default, recommended for RAM-constrained MCUs)
 *   (empty)  → weights in RAM   (faster on MCUs with D-Cache, e.g. Cortex-M7)
 *
 */
#ifndef EMBEDIA_MODEL_STORAGE
#define EMBEDIA_MODEL_STORAGE const
#endif

#include <stdlib.h>
#include <stdint.h>
#include "realtype.h"

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
    real_t * data;      /**< Flattened data array in CHW order */
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
    union {
        uint16_t height;    /**< For images: vertical dimension */
        uint16_t channels;  /**< For signals: number of channels/features */
    };
    uint16_t width;     /**< Primary dimension (width/time steps/length) */
    real_t * data;       /**< Flattened data array */
} data2d_t;

/**
 * @struct data1d_t
 * @brief 1D data container for vectors and sequential data
 * @details Used for flattened arrays, feature vectors, or time series
 *
 * - length: Total number of elements in the vector
 * - data: Contiguous array of real_t (floating-point) values
 */
typedef struct {
    uint32_t length;    /**< Total number of elements in the vector */
    real_t * data;      /**< Contiguous array of values */
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


/**@group Generic layer functions */

/**@brief Copies data between 1D, 2D, and 3D structures
 *
 * These functions perform deep copies of data structures, including:
 * - Memory allocation with swap_alloc()
 * - Data copying from source to destination
 * - Proper initialization of metadata (length, width, height, channels)
 *
 * @note
 * - Input structure must be properly initialized
 * - Memory is allocated allways for the output->data pointer, even if it was previously allocated
 * - All dimensions (length/width/height/channels) are copied
 *
 * @param input Source structure (will be read from)
 * @param output Destination structure (will be modified)
 *
 * @see swap_alloc()
 */
void copy_data_1d(const data1d_t *input, data1d_t *output);
void copy_data_2d(const data2d_t *input, data2d_t *output);
void copy_data_3d(const data3d_t *input, data3d_t *output);


/**@endgroup Generic layer functions */

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
 * @brief Allocates a block and slices it into 2 independently aligned regions.
 *
 * Both regions live in the same swap buffer slot, so the total cost to
 * ALLOC_BUFFER_SZ is (size1_aligned + size2_aligned), not two separate slots.
 * This allows internal temporaries and output to share one slot, minimizing
 * peak RAM usage without any extra static buffers.
 *
 * The Python estimator must use:
 *   slot_size = align4(size1) + align4(size2)
 * when computing ALLOC_BUFFER_SZ for layers that call swap_alloc_slice.
 *
 * @param size1  Size in bytes of the first region (e.g. temporaries)
 * @param size2  Size in bytes of the second region (e.g. output)
 * @param ptr1   Receives pointer to first region
 * @param ptr2   Receives pointer to second region (guaranteed 4-byte aligned)
 */
void swap_alloc_slice(uint32_t size1, uint32_t size2,
                      void **ptr1, void **ptr2);

/**
 * @brief Allocates a block and slices it into 3 independently aligned regions.
 *
 * Extension of swap_alloc_slice for layers that need two internal temporaries
 * plus an output buffer (e.g. STFT needs data_re, data_im, and output->data).
 *
 * The Python estimator must use:
 *   slot_size = align4(size1) + align4(size2) + align4(size3)
 *
 * @param size1  Size in bytes of the first region
 * @param size2  Size in bytes of the second region
 * @param size3  Size in bytes of the third region
 * @param ptr1   Receives pointer to first region
 * @param ptr2   Receives pointer to second region
 * @param ptr3   Receives pointer to third region
 */
void swap_alloc_slice3(uint32_t size1, uint32_t size2, uint32_t size3,
                       void **ptr1, void **ptr2, void **ptr3);



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
 *
 */
// @embedia-include common/_dot_product_bias.c



/**
 * @brief Computes the dot product of two real_t arrays
 *
 * @param a   First array
 * @param b   Second array
 * @param len Number of elements
 * @return Dot product (sum of a[i]*b[i])
 */
#define dot_product(a, b, len, qp) dot_product_bias(a, b, len, REAL_ZERO, qp)


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
 * data1d_t output = { .length = 3, .data = (real_t[]){0.1f, 0.7f, 0.2f} };
 * uint32_t predicted_class = argmax(output); // Returns 1
 */

EMBEDIA_INLINE int32_t argmax(const real_t *data, uint32_t length)
{
    if (length == 0u) return -1;

    const real_t *ptr = data;
    real_t max_val = *ptr;
    const real_t *max_ptr = ptr;

    uint32_t remaining = length;
    ptr++;                        // ya leímos el primero
    remaining--;

    // Procesar en bloques de 4
    for (; remaining >= 4u; remaining -= 4u)  {
        if (*ptr > max_val) { max_val = *ptr; max_ptr = ptr; }
        ptr++;

        if (*ptr > max_val) { max_val = *ptr; max_ptr = ptr; }
        ptr++;

        if (*ptr > max_val) { max_val = *ptr; max_ptr = ptr; }
        ptr++;

        if (*ptr > max_val) { max_val = *ptr; max_ptr = ptr; }
        ptr++;
    }

    // Procesar elementos restantes
    for (; remaining > 0u; remaining--)
    {
        if (*ptr > max_val) {
            max_val = *ptr;
            max_ptr = ptr;
        }
        ptr++;
    }

    return (uint32_t)(max_ptr - data);
}

/**
 * @brief Finds the index of the minimum value in a 1D data vector
 *
 * Useful for finding the least confident class or detecting anomalies.
 *
 * @param data     Input 1D data vector
 * @param length   Number of elements
 * @return         Index of the minimum value (first occurrence if ties)
 *
 * @note If length == 0, behavior is undefined (consider adding check if needed)
 */
EMBEDIA_INLINE int32_t argmin(const real_t *data, uint32_t length)
{
    if (length == 0u) return -1;

    const real_t *ptr = data;
    real_t min_val = *ptr;
    const real_t *min_ptr = ptr;

    uint32_t remaining = length;
    ptr++;
    remaining--;

    for (; remaining >= 4u; remaining -= 4u)
    {
        if (*ptr < min_val) { min_val = *ptr; min_ptr = ptr; }
        ptr++;

        if (*ptr < min_val) { min_val = *ptr; min_ptr = ptr; }
        ptr++;

        if (*ptr < min_val) { min_val = *ptr; min_ptr = ptr; }
        ptr++;

        if (*ptr < min_val) { min_val = *ptr; min_ptr = ptr; }
        ptr++;
    }

    for (; remaining > 0u; remaining--)
    {
        if (*ptr < min_val)
        {
            min_val = *ptr;
            min_ptr = ptr;
        }
        ptr++;
    }

    return (uint32_t)(min_ptr - data);
}

/**
 * @brief Finds the maximum value in a 1D data vector
 *
 * @param data     Input vector
 * @param length   Number of elements
 * @return         The maximum value found
 *
 * @note Returns the first occurrence value if there are ties
 */
EMBEDIA_INLINE real_t max_val(const real_t *data, uint32_t length)
{
    if (length == 0u) return REAL_MIN;

    const real_t *ptr = data;
    real_t max_val = *ptr;

    uint32_t remaining = length - 1u;
    ptr++;

    for (; remaining >= 4u; remaining -= 4u)
    {
        if (*ptr > max_val) max_val = *ptr;
        ptr++;

        if (*ptr > max_val) max_val = *ptr;
        ptr++;

        if (*ptr > max_val) max_val = *ptr;
        ptr++;

        if (*ptr > max_val) max_val = *ptr;
        ptr++;
    }

    for (; remaining > 0u; remaining--)
    {
        if (*ptr > max_val) max_val = *ptr;
        ptr++;
    }

    return max_val;
}

/**
 * @brief Finds the minimum value in a 1D data vector
 */
EMBEDIA_INLINE real_t min_val(const real_t *data, uint32_t length)
{
    if (length == 0u) return REAL_MAX;

    const real_t *ptr = data;
    real_t min_val = *ptr;

    uint32_t remaining = length - 1;
    ptr++;

    for (; remaining >= 4; remaining -= 4)
    {
        if (*ptr < min_val) min_val = *ptr;
        ptr++;
        if (*ptr < min_val) min_val = *ptr;
        ptr++;
        if (*ptr < min_val) min_val = *ptr;
        ptr++;
        if (*ptr < min_val) min_val = *ptr;
        ptr++;
    }

    for (; remaining > 0u; remaining--)
    {
        if (*ptr < min_val) min_val = *ptr;
        ptr++;
    }

    return min_val;
}

/**
 * @brief Computes the sum of all elements in a 1D data vector
 *
 * @param data     Input 1D data vector
 * @param length   Number of elements
 * @return         Sum of all elements
 *
 * @note If length == 0, returns 0.0f
 */
// @embedia-include common/_sum_val.c



/**
 * @brief Computes the mean (average) of all elements in a 1D data vector
 *
 * @param data     Input 1D data vector
 * @param length   Number of elements
 * @return         Mean value (sum / length), or 0.0f if length == 0
 *
 * @note Uses simple division; for better numerical stability in very large arrays
 *       consider Kahan summation or other compensated algorithms if needed.
 */
// Function: mean_val
// @embedia-include common/_mean_val.c


#ifdef __cplusplus
}
#endif

#endif