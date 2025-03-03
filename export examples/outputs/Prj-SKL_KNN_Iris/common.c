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

/*
 * argmax()
 *  Finds the index of the largest value within a vector of data (data1d_t)
 * Parameters:
 *  data => data of type data1d_t to search for max.
 * Returns:
 *  search result - index of the maximum value
 */

/*
 * argmax()
 *  Finds the index of the largest value within a vector of data (data1d_t)
 * Parameters:
 *  data => data of type data1d_t to search for max.
 * Returns:
 *  search result - index of the maximum value
 */
uint32_t argmax(data1d_t data){
    fixed max = data.data[0];
    uint32_t i, pos = 0;

    for(i=1;i<data.length;i++){
        if(data.data[i]>max){
            max = data.data[i];
            pos = i;
        }
    }

    return pos;
}
