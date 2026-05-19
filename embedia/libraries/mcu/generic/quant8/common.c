/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
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

#include <stdio.h>

// Size of the static memory pool (in bytes). Redefined when exporting
#define ALLOC_BUFFER_SZ 10000

typedef struct{
    uint32_t size;
    void  * data;
} raw_buffer;


#define MAX_BUFFER 2

// last allocated buffer
static unsigned char id = MAX_BUFFER-1;

// info for size and pointer of buffer (only 2 buffers available)
static raw_buffer buffer[MAX_BUFFER] = {0};

// Buffer for memory management
static unsigned char pool_buffer[ALLOC_BUFFER_SZ];


/**
 * Initializes the double buffer system to a known state. Resets the buffer allocation
 * system before starting a new processing sequence. Must be called before first use
 * of swap_alloc() or when restarting a pipeline workflow.
 *
 * Notes:
 * - Resets buffer index to MAX_BUFFER-1, so next swap_alloc() uses buffer 0
 * - No memory allocation/deallocation - just state reset
 * - Safe to call multiple times
 */
void prepare_buffers(){
    id = MAX_BUFFER-1;
    buffer[0].size =0;
    buffer[1].size =0;
}

/**
 * swap_alloc(): Double-buffer allocator for pipeline processing within a static pool.
 * Buffers alternate to reuse memory without fragmentation.
 * Designed for deterministic behavior on MCUs.
 * Each new allocation automatically invalidates the previous one.
 */
void * swap_alloc(uint32_t s){

    if (s!=0) // Apply 4-byte alignment (critical for MCU like Cortex-M0, optimal for M3/M4)
        s = (s + 3) & ~3;

    // Check for buffer collision
    if ((buffer[1-id].size+s) > ALLOC_BUFFER_SZ){
        printf("Insufficient buffer size. Required %d bytes when available %d bytes.\n",
               s, ALLOC_BUFFER_SZ-buffer[1-id].size);
        return NULL;
    }

    if (++id == MAX_BUFFER){
        id = 0;
    }

    buffer[id].size = s;

    // Buffer 0 grows from left, buffer 1 from right
    if (id==0){ //buffer from left
        buffer[id].data = pool_buffer;
    }else{
        buffer[id].data = pool_buffer+ALLOC_BUFFER_SZ-s ;
    }

    printf("*******************************\n memory allocated: %d bytes\n*******************************\n", buffer[0].size+buffer[1].size);
    return buffer[id].data;
}



/**
 * argmax(): Finds the class with highest probability in ML output.
 * Used in classification layers to get final prediction.
 * Finds the index of the largest value within a vector of data (data1d_t)
 */
uint32_t argmax(data1d_t data){
    float max = data.data[0];
    uint32_t i, pos = 0;

    for(i=1;i<data.length;i++){
        if(data.data[i]>max){
            max = data.data[i];
            pos = i;
        }
    }

    return pos;
}