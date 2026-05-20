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
 * @file common.c
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

#include "common.h"
#include <strings.h>
#include <stdio.h>


typedef struct{
    uint32_t size;
    void  * data;
} raw_buffer;


void copy_data_1d(const data1d_t *input, data1d_t *output){

    output->length = input->length;
    const uint32_t size = sizeof(real_t) * input->length;

    output->data = (real_t*) swap_alloc(size);
    memcpy(output->data, input->data, size);
}

void copy_data_2d(const data2d_t *input, data2d_t *output){

    output->width  = input->width;
    output->height = input->height;
    const uint32_t size = sizeof(real_t) * input->width * input->height;

    output->data = (real_t*) swap_alloc(size);
    memcpy(output->data, input->data, size);
}

void copy_data_3d(const data3d_t *input, data3d_t *output){

    output->width    = input->width;
    output->height   = input->height;
    output->channels = input->channels;
    const uint32_t size = sizeof(real_t) * input->width * input->height * input->channels;

    output->data = (real_t*) swap_alloc(size);
    memcpy(output->data, input->data, size);
}

#define MAX_BUFFER 2

// Size of the static memory pool (in bytes). Redefined when exporting
#define ALLOC_BUFFER_SZ 10000

/* -----------------------------------------------------------------------
 * Internal state
 * ----------------------------------------------------------------------- */

// last allocated buffer index
static unsigned char id = MAX_BUFFER - 1;

// size and pointer info for each buffer
static raw_buffer buffer[MAX_BUFFER];

// static memory pool — size defined at export time by Python estimator
static unsigned char pool_buffer[ALLOC_BUFFER_SZ];


/* -----------------------------------------------------------------------
 * prepare_buffers()
 *  Resets the double buffer system to a known state.
 *  Must be called before the first inference and between inferences
 *  if the pipeline is restarted.
 * ----------------------------------------------------------------------- */
void prepare_buffers() {
    id = MAX_BUFFER - 1;
    buffer[0].size = 0;
    buffer[1].size = 0;
}


/* -----------------------------------------------------------------------
 * swap_alloc()
 *  Core double-buffer allocator. Alternates between two zones of
 *  pool_buffer: buffer 0 grows from the left, buffer 1 from the right.
 *  Each allocation automatically invalidates the previous one on the
 *  same side, so no fragmentation is possible.
 *
 *  Alignment: every allocation is rounded up to 4 bytes, which is
 *  required for Cortex-M0 and optimal for M3/M4.
 * ----------------------------------------------------------------------- */
void *swap_alloc(uint32_t s) {
    if (s != 0)
        s = (s + 3) & ~3;  // 4-byte alignment

    if (++id == MAX_BUFFER)
        id = 0;

    // collision check: new block must not overlap the other buffer
    if ((buffer[1 - id].size + s) > ALLOC_BUFFER_SZ) {
        printf("\n\n   <<< Insufficient buffer size. "
               "Required %d bytes, available %d bytes >>>\n\n\n",
               s, ALLOC_BUFFER_SZ - buffer[1 - id].size);
        return NULL;
    }

    buffer[id].size = s;

    // buffer 0 occupies the left side, buffer 1 the right side
    if (id == 0) {
        buffer[id].data = pool_buffer;
    } else {
        buffer[id].data = pool_buffer + ALLOC_BUFFER_SZ - s;
    }

#if DEBUG_MODE && DEBUG_ALLOC_BUFFER
    printf("*******************************\n"
           " memory allocated: %d bytes\n"
           "*******************************\n",
           buffer[0].size + buffer[1].size);
#endif

    return buffer[id].data;
}


/* -----------------------------------------------------------------------
 * swap_alloc_slice()
 *  Allocates a single block via swap_alloc and slices it into 2 regions.
 *  Each region is individually 4-byte aligned within the block.
 *
 *  Typical use: one region for internal temporaries, one for output.
 *
 *  Example — KNN:
 *    fixed        *dq_data;
 *    DistanceIndex *heap;
 *    swap_alloc_slice(
 *        n_features  * sizeof(fixed),
 *        n_neighbors * sizeof(DistanceIndex),
 *        (void**)&dq_data,
 *        (void**)&heap
 *    );
 *
 *  The Python estimator must account for size1 + size2 (aligned) when
 *  computing ALLOC_BUFFER_SZ for layers that use swap_alloc_slice.
 * ----------------------------------------------------------------------- */
void swap_alloc_slice(uint32_t size1, uint32_t size2,
                      void **ptr1, void **ptr2) {
    // align each slice independently so ptr2 is always aligned
    size1 = (size1 + 3) & ~3;
    size2 = (size2 + 3) & ~3;

    uint8_t *block = (uint8_t *)swap_alloc(size1 + size2);

    *ptr1 = block;
    *ptr2 = block + size1;
}


/* -----------------------------------------------------------------------
 * swap_alloc_slice3()
 *  Same as swap_alloc_slice but divides the block into 3 regions.
 *
 *  Typical use: two internal temporary buffers plus output, as in STFT
 *  where data_re, data_im and output->data all live in the same slot.
 *
 *  Example — STFT:
 *    fixed *data_re, *data_im;
 *    fixed *out;
 *    swap_alloc_slice3(
 *        frame_length * sizeof(fixed),               // data_re
 *        frame_length * sizeof(fixed),               // data_im
 *        n_frames * n_fft * ch * sizeof(fixed),      // output
 *        (void**)&data_re,
 *        (void**)&data_im,
 *        (void**)&out
 *    );
 *    output->data = out;
 * ----------------------------------------------------------------------- */
void swap_alloc_slice3(uint32_t size1, uint32_t size2, uint32_t size3,
                       void **ptr1, void **ptr2, void **ptr3) {
    size1 = (size1 + 3) & ~3;
    size2 = (size2 + 3) & ~3;
    size3 = (size3 + 3) & ~3;

    uint8_t *block = (uint8_t *)swap_alloc(size1 + size2 + size3);

    *ptr1 = block;
    *ptr2 = block + size1;
    *ptr3 = block + size1 + size2;
}



