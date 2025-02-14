#include "common.h"

typedef struct{
    size_t  size;
    void  * data;
} raw_buffer;

#define MAX_BUFFER 2

static unsigned char id = MAX_BUFFER-1;
static raw_buffer buffer[MAX_BUFFER] = {0};

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