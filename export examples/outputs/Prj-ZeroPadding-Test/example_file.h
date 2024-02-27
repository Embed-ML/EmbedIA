#ifndef _EXAMPLE_FILE_H
#define _EXAMPLE_FILE_H

#include "embedia.h"



#define MAX_SAMPLE 0

#define SELECT_SAMPLE 0

#if SELECT_SAMPLE == 0
        
uint16_t sample_data_id = 0;

static float sample_data[]= {
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
};

#endif


#endif