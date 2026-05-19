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

#include <stdlib.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


/**
 * Structure that stores an array of float data (float * data) in vector form.
 * Specifies the number of channels, the width and the height of the array.
 */


/**
 * @brief Multi-dimensional data containers for different processing stages
 */
typedef struct{
    uint16_t channels;
    uint16_t width;
    uint16_t height;
    float * data;
}data3d_t;

typedef struct{
    uint16_t width;
    uint16_t height;
    float * data;
}data2d_t;

typedef struct{
    uint32_t length;
    float * data;
}data1d_t;

typedef struct{
    uint16_t h;
    uint16_t w;
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
 * @brief Computes the dot product of two float arrays
 *
 * @param a   First array
 * @param b   Second array
 * @param len Number of elements
 * @return Dot product (sum of a[i]*b[i])
 */
EMBEDIA_INLINE float dot_product(const float* a, const float* b, uint32_t len) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}


#ifdef __cplusplus
}
#endif

#endif