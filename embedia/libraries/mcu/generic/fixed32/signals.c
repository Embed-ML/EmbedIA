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

#include <stdlib.h>
#include <math.h>
#include "signals.h"



/* ------------------------------ Spectrogram ------------------------------ */



/*
 * void rearrange(fixed data_re[], fixed data_im[], const unsigned int N)
 * Performs the necessary reordering of the data before applying the FFT.
 * Parameters:
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data.
 *   - N: Amount of samples to perform the FFT over.
 */
static inline void rearrange(fixed data_re[], fixed data_im[], const unsigned int N) {
    register unsigned int position;
    unsigned int target = 0;

    for(position = 0; position < N; position++) {
        if(target > position) {
            const fixed temp_re = data_re[target];
            const fixed temp_im = data_im[target];
            data_re[target] = data_re[position];
            data_im[target] = data_im[position];
            data_re[position] = temp_re;
            data_im[position] = temp_im;
        }
        unsigned int mask = N;
        while(target & (mask >>= 1))
            target &= ~mask;
        target |= mask;
    }
}

/*
 * void compute(fixed data_re[], fixed data_im[], const unsigned int N)
 * Contains the FFT calculation core, applying the Fourier transforms for
 * each recursive step.
 * Parameters:
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data.
 *   - N: Amount of samples to perform the FFT over.
 */

static inline void compute(fixed data_re[], fixed data_im[], const unsigned int N) {
    const fixed pi = FIXED_NEG(FIX_PI);
    register unsigned int step, group, pair;

    for(step = 1; step < N; step <<= 1) {
        const unsigned int jump = step << 1;
        const fixed step_d = INT_TO_FIXED(step);
        fixed twiddle_re = FIX_ONE;
        fixed twiddle_im = FIX_ZERO;
        int max_val = 0;

        for(group = 0; group < step; group++) {
            for(pair = group; pair < N; pair += jump) {
                const unsigned int match = pair + step;

                // Multiplicación con precisión extendida
                dfixed product_re_tmp = ((dfixed)twiddle_re * data_re[match] - (dfixed)twiddle_im * data_im[match]) >> FIX_FRC_SZ;
                dfixed product_im_tmp = ((dfixed)twiddle_im * data_re[match] + (dfixed)twiddle_re * data_im[match]) >> FIX_FRC_SZ;

                // Detección de desbordamiento
                fixed product_re = (fixed)product_re_tmp;
                fixed product_im = (fixed)product_im_tmp;

                fixed new_re = data_re[pair] - product_re;
                fixed new_im = data_im[pair] - product_im;
                data_re[pair] = data_re[pair] + product_re;
                data_im[pair] = data_im[pair] + product_im;
                data_re[match] = new_re;
                data_im[match] = new_im;

                // Actualizar máximo valor
                int current_max = FIXED_MAX(FIXED_ABS(data_re[pair]), FIXED_ABS(data_im[pair]));
                current_max = FIXED_MAX(current_max, FIXED_MAX(FIXED_ABS(data_re[match]), FIXED_ABS(data_im[match])));
                max_val = FIXED_MAX(max_val, current_max);
            }

            if(group + 1 == step) continue;

            fixed angle = FIXED_DIV(FIXED_MUL(pi, INT_TO_FIXED(group + 1)), step_d);
            twiddle_re = fixed_cos(angle);
            twiddle_im = fixed_sin(angle);
        }
    }

}


/*
 * void fft(fixed data_re[], fixed data_im[], const unsigned int N)
 * Performs a Fast Fourier Transform (FFT) on the complex data passed as parameters.
 * Parameters:
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data.
 *   - N: Amount of samples to perform the FFT over.
 * First performs a reordering of the data and then applies the FFT calculations.
 */
static inline void fft(fixed data_re[], fixed data_im[], const unsigned int N) {
    rearrange(data_re, data_im, N);
    compute(data_re, data_im, N);
}



static inline void apply_symm_window(fixed* input,
                                   fixed* output_re, fixed* output_im,
                                   unsigned int input_size,
                                   const window_t* window,   // ← cambio
                                   unsigned int frame_length,
                                   unsigned int start, int gain_compensation) {
    const unsigned int total_length = frame_length;
    const unsigned int midpoint = frame_length >> 1;
    const int right_shift = gain_compensation;
    unsigned int j;

  for (j = 0; j < total_length; j++) {
    if (start + j < input_size) {
      window_t win_value;
      if (j < midpoint) {
        win_value = window[j];
      } else {
        win_value = window[frame_length - 1 - j];
      }
      dfixed temp =
          (dfixed)input[start + j] * (dfixed)win_value; // ← cast explícito
      output_re[j] = (fixed)((temp + (1 << (right_shift - 1))) >> right_shift);
    } else {
      output_re[j] = FIX_ZERO;
    }
    output_im[j] = FIX_ZERO;
  }
}

/*
 * void create_spectrogram(spectrogram_layer_t config, data2d_t input, data3d_t *output)
 * Generates the spectrogram from the input signal by applying FFTs.
 * Parameters:
 *   - config: Spectrogram layer configuration
 *   - input:  2D input signal
 *   - output: 3D output spectrogram (W = frame_length//2, H = n_frames, Ch = 1)
 */
#define DEBUG_STFT 0
#if DEBUG_STFT
#include <stdio.h>
void printf_vector(char * name_vector_debug, fixed vector[], int n) {
        int i;
        printf("#%s:\n", name_vector_debug);
        printf("%s = np.array([ ", name_vector_debug);
        for(i = 0; i < n; i++) {
            printf("%f, ", FIXED_TO_FLOAT(vector[i]));
        }
        printf("])\n\n");
    }
#endif // DEBUG_STFT

void multi_stft_layer(spectrogram_layer_t config, data2d_t input, data3d_t *output) {
    #if DEBUG_STFT
    void printf_vector(char * name_vector_debug, fixed vector[], int n) {
        int i;
        printf("#%s:\n", name_vector_debug);
        printf("%s = np.array([ ", name_vector_debug);
        for(i = 0; i < n; i++) {
            printf("%f, ", FIXED_TO_FLOAT(vector[i]));
        }
        printf("])\n\n");
    }
    char name_vector_debug[30];
    #endif

    register int i, j, c = 0;
    int aux_n_fft = 0;

    /*
    fixed *data_re = (fixed *)malloc((config.frame_length + aux_n_fft) * sizeof(fixed));
    fixed *data_im = (fixed *)malloc((config.frame_length + aux_n_fft) * sizeof(fixed));

    output->height   = config.n_frames;
    output->width    = config.n_fft_table;
    output->channels = input.height;
    output->data     = (fixed *)swap_alloc(sizeof(fixed) * output->height * output->width * output->channels);
    */

    const uint16_t frm_len = config.frame_length;
    const uint16_t n_ch      = input.channels;
    const uint16_t n_frms    = config.n_frames;
    const uint16_t n_bins    = config.n_fft_table;

    const uint32_t sz_re  = (uint32_t)frm_len * sizeof(fixed);
    const uint32_t sz_im  = (uint32_t)frm_len * sizeof(fixed);
    const uint32_t sz_out = (uint32_t)n_ch * n_frms * n_bins * sizeof(fixed);

    output->height   = n_frms;
    output->width    = n_bins;
    output->channels = n_ch;

    fixed *data_re, *data_im;
    swap_alloc_slice3(sz_re, sz_im, sz_out, (void**)&data_re, (void**)&data_im, (void**)&(output->data));


    for (c = 0; c < input.height; c++) {
        // Calcular media (usando dfixed para mayor precisión)
        dfixed sum = 0;
        for (i = 0; i < input.width; i++)
            sum += (dfixed)input.data[c * input.width + i];
        fixed mean = (fixed)(sum / input.width);

        // Restar media (sin escalado adicional)
        fixed channel_data[input.width];
        for (i = 0; i < input.width; i++) {
            channel_data[i] = FIXED_SUB(input.data[c * input.width + i], mean);
        }

        #if DEBUG_STFT
        printf("Channel %d:\n", c);
        printf_vector("senial", channel_data, input.width);
        #endif

        for (i = 0; i < config.n_frames; i++) {
            const unsigned int start = i * config.hop_length;

            apply_symm_window(channel_data, data_re, data_im, input.width,
                              config.window, config.frame_length, start, config.window_shift-0);


            #if DEBUG_STFT
            sprintf(name_vector_debug, "senial_bloque_%d", i);
            printf_vector(name_vector_debug, data_re, config.frame_length);
            #endif

            // FFT
            fft(data_re, data_im, config.frame_length + aux_n_fft);


            #if DEBUG_STFT
            sprintf(name_vector_debug, "fft_real_bloque_%d", i);
            printf_vector(name_vector_debug, data_re, config.frame_length);
            sprintf(name_vector_debug, "fft_imag_bloque_%d", i);
            printf_vector(name_vector_debug, data_im, config.frame_length);
            #endif

            // Magnitud (sin normalización adicional)
            for (j = 0; j < config.n_fft_table; j++) {
                dfixed aux_re = (dfixed)data_re[j];
                dfixed aux_im = (dfixed)data_im[j];

                dfixed re_sq = (aux_re * aux_re);
                dfixed im_sq = (aux_im * aux_im);
                dfixed sum_sq = (re_sq + im_sq);
                data_re[j] = fixed_sqrt(sum_sq >> FIX_FRC_SZ);

            }

            // Conversión a dB (opcional)
            if (config.convert_to_db) {
                fixed log10_scale = FL2FX_CONST(20.0f);
                fixed eps = FIX_MIN;//FL2FX_CONST(1e-6f);  // 1e-8 no es representable en Q17

                // Primera pasada: calcular dB y buscar pico
                fixed peak_db = FIX_MIN;
                for (j = 0; j < config.n_fft_table; j++) {
                    fixed val = FIXED_MAX(data_re[j], eps);
                    data_re[j] = FIXED_MUL(log10_scale, fixed_logn(val, FL2FX_CONST(10.0f)));
                    if (data_re[j] > peak_db) peak_db = data_re[j];
                }

                // Segunda pasada: clampear a top_db relativo al pico
                fixed floor_db = peak_db - FL2FX_CONST(80.0f);
                for (j = 0; j < config.n_fft_table; j++) {
                    data_re[j] = FIXED_MAX(data_re[j], floor_db);
                }
            }

            // Guardar en output
            for (j = 0; j < config.n_fft_table; j++) {
                output->data[(c * output->height * output->width) + (i * output->width) + j] = data_re[j];
            }

            #if DEBUG_STFT
            sprintf(name_vector_debug, "bloque_%d", i);
            printf_vector(name_vector_debug, data_re, config.n_fft_table);
            #endif
        }
    }

    #if DEBUG_STFT
    printf_vector("array", output->data, output->height * output->width * output->channels);
    #endif
}

void stft_layer(spectrogram_layer_t config, data1d_t input, data2d_t *output) {
    data2d_t inp_2d;
    data3d_t out_3d;

    inp_2d.height = 1;
    inp_2d.width  = input.length;
    inp_2d.data   = input.data;

    multi_stft_layer(config, inp_2d, &out_3d);

    output->data   = out_3d.data;
    output->width  = out_3d.width;
    output->height = out_3d.height;
}