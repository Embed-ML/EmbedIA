#ifndef _EXAMPLE_FILE_H
#define _EXAMPLE_FILE_H

#include "embedia.h"



#define MAX_SAMPLE 11

#define SELECT_SAMPLE 0

#if SELECT_SAMPLE == 0
uint16_t sample_data_id = 0.0;

static float sample_data[]= {
  34.0, -1.0, 1005.63, 24.0, 4.0, 43.0, 26.0, 9.0, -10.0, 1009.0, 999.0
};

#endif
#if SELECT_SAMPLE == 1
uint16_t sample_data_id = 0.0;

static float sample_data[]= {
  36.0, 4.0, 1005.46, 21.0, 6.0, 43.0, 29.0, 10.0, -2.0, 1008.0, 1001.0
};

#endif
#if SELECT_SAMPLE == 2
uint16_t sample_data_id = 0.0;

static float sample_data[]= {
  35.0, 6.0, 1006.0, 27.0, 5.0, 41.0, 29.0, 12.0, -2.0, 1009.0, 1000.0
};

#endif
#if SELECT_SAMPLE == 3
uint16_t sample_data_id = 0.0;

static float sample_data[]= {
  34.0, 7.0, 1005.65, 29.0, 6.0, 41.0, 27.0, 13.0, 0.0, 1008.0, 1001.0
};

#endif
#if SELECT_SAMPLE == 4
uint16_t sample_data_id = 1.0;

static float sample_data[]= {
  31.0, 11.0, 1007.94, 61.0, 13.0, 38.0, 24.0, 16.0, 6.0, 1011.0, 1003.0
};

#endif
#if SELECT_SAMPLE == 5
uint16_t sample_data_id = 0.0;

static float sample_data[]= {
  28.0, 13.0, 1008.39, 69.0, 18.0, 34.0, 21.0, 17.0, 9.0, 1011.0, 1004.0
};

#endif
#if SELECT_SAMPLE == 6
uint16_t sample_data_id = 0.0;

static float sample_data[]= {
  30.0, 10.0, 1007.62, 50.0, 8.0, 38.0, 23.0, 14.0, 6.0, 1010.0, 1002.0
};

#endif
#if SELECT_SAMPLE == 7
uint16_t sample_data_id = 0.0;

static float sample_data[]= {
  34.0, 8.0, 1006.73, 32.0, 7.0, 41.0, 26.0, 12.0, 6.0, 1010.0, 1002.0
};

#endif
#if SELECT_SAMPLE == 8
uint16_t sample_data_id = 1.0;

static float sample_data[]= {
  34.0, 11.0, 1005.75, 45.0, 7.0, 42.0, 27.0, 16.0, 7.0, 1008.0, 1000.0
};

#endif
#if SELECT_SAMPLE == 9
uint16_t sample_data_id = 1.0;

static float sample_data[]= {
  34.0, 16.0, 1007.1, 51.0, 12.0, 41.0, 27.0, 18.0, 13.0, 1010.0, 1002.0
};

#endif
#if SELECT_SAMPLE == 10
uint16_t sample_data_id = 1.0;

static float sample_data[]= {
  32.0, 16.0, 1006.78, 66.0, 16.0, 40.0, 25.0, 22.0, 10.0, 1011.0, 1001.0
};

#endif
#if SELECT_SAMPLE == 11
uint16_t sample_data_id = 0.1;

static float sample_data[]= {
  34.0, 13.0, 1003.83, 58.0, 9.0, 42.0, 27.0, 20.0, 10.0, 1007.0, 998.0
};

#endif


#endif