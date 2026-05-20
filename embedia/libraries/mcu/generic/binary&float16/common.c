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
#include "common.h"

typedef struct{
    size_t  size;
    void  * data;
} raw_buffer;

#define MAX_BUFFER 2

static unsigned char id = MAX_BUFFER-1;
static raw_buffer buffer[MAX_BUFFER];

void prepare_buffers(){
    id = MAX_BUFFER-1;
}

void * swap_alloc(size_t s){

    if (++id == MAX_BUFFER){
        id = 0;
    }

    if (buffer[id].size < s){
        buffer[id].data = realloc(buffer[id].data, s);
        buffer[id].size = s;
    }
    return buffer[id].data;
}