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
 * void fft(float data_re[], float data_im[], const unsigned int N)
 * Performs a Fast Fourier Transform (FFT) on the complex data passed as parameters.
 * Parameters:
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data.
 *   - N: Amount of samples to perform the FFT over.
 * First performs a reordering of the data and then applies the FFT calculations.
 */
void fft(float data_re[], float data_im[], const unsigned int N){
    rearrange(data_re, data_im, N);
    compute(data_re, data_im, N);
}

/*
 * void rearrange(float data_re[], float data_im[], const unsigned int N)
 * Performs the necessary reordering of the data before applying the FFT.
 * Parameters:
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data.
 *   - N: Amount of samples to perform the FFT over.
 */
void rearrange(float data_re[], float data_im[], const unsigned int N){
  register unsigned int position;
  unsigned int target = 0;

  for(position=0; position<N;position++){
      if(target>position) {
        const float temp_re = data_re[target];
        const float temp_im = data_im[target];
        data_re[target] = data_re[position];
        data_im[target] = data_im[position];
        data_re[position] = temp_re;
        data_im[position] = temp_im;
      }
      unsigned int mask = N;
      while(target & (mask >>=1))
        target &= ~mask;
      target |= mask;
    }
}

/*
 * void compute(float data_re[], float data_im[], const unsigned int N)
 * Contains the FFT calculation core, applying the Fourier transforms for
 * each recursive step.
 * Parameters:
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data.
 *   - N: Amount of samples to perform the FFT over.
 */
void compute(float data_re[], float data_im[], const unsigned int N){
  const float pi = -3.14159265358979323846;
  register unsigned int step,group,pair;

  for(step=1; step<N; step <<=1) {
    const unsigned int jump = step << 1;
    const float step_d = (float) step;
    float twiddle_re = 1.0;
    float twiddle_im = 0.0;
    for(group=0; group<step; group++){
        for(pair=group; pair<N; pair+=jump){
            const unsigned int match = pair + step;
            const float product_re = twiddle_re*data_re[match]-twiddle_im*data_im[match];
            const float product_im = twiddle_im*data_re[match]+twiddle_re*data_im[match];
            data_re[match] = data_re[pair]-product_re;
            data_im[match] = data_im[pair]-product_im;
            data_re[pair] += product_re;
            data_im[pair] += product_im;
        }

        // we need the factors below for the next iteration
        // if we don't iterate then don't compute
        if(group+1 == step){
            continue;
        }

        float angle = pi*((float) group+1)/step_d;
        twiddle_re = cosf(angle);
        twiddle_im = sinf(angle);
    }
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
#endif // DEBUG_STFT

void multi_stft_layer(spectrogram_layer_t config, data2d_t input, data3d_t *output) {
    #if DEBUG_STFT
    #include <stdio.h>
    void printf_vector(char * name_vector_debug, float vector[], int n){
        int i;
        printf("#%s:\n", name_vector_debug);
        printf("%s = np.array([ ", name_vector_debug);
        for(i=0; i<n; i++){
            printf("%f, ", vector[i]);
        }
        printf("])\n\n");
    }
    char name_vector_debug[20];
    #endif

    register int i, j, c=0;
    float aux_re, aux_im;
    int aux_n_fft = 0;

    // Temporary arrays for FFT input
    // float data_re[config.frame_length + aux_n_fft];
    // float data_im[config.frame_length + aux_n_fft];

    // For hanning window
    float win;
    int midpoint;

    const uint16_t frm_len = config.frame_length;
    const uint16_t n_ch      = input.channels;
    const uint16_t n_frms    = config.n_frames;
    const uint16_t n_bins    = config.n_fft_table;

    const uint32_t sz_re  = (uint32_t)frm_len * sizeof(float);
    const uint32_t sz_im  = (uint32_t)frm_len * sizeof(float);
    const uint32_t sz_out = (uint32_t)n_ch * n_frms * n_bins * sizeof(float);

    output->height   = n_frms;
    output->width    = n_bins;
    output->channels = n_ch;

    float *data_re, *data_im;
    swap_alloc_slice3(sz_re, sz_im, sz_out, (void**)&data_re, (void**)&data_im, (void**)&(output->data));


    // Procesar cada canal por separado
    for (c = 0; c < input.height; c++) {
        // Calculate mean for this channel
        float mean = 0.0f;
        for (i = 0; i < input.width; i++) {
            mean += input.data[c * input.width + i];
        }
        mean /= input.width;

        // Subtract mean from input signal (like in Python)
        float channel_data[input.width];
        for (i = 0; i < input.width; i++) {
            channel_data[i] = input.data[c * input.width + i] - mean;
        }

        #if DEBUG_STFT
        printf("Channel %d:\n", c);
        printf_vector("senial", channel_data, input.width);
        #endif // DEBUG_STFT

        midpoint = config.frame_length / 2;

        for (i = 0; i < config.n_frames; i++) {
            const unsigned int start = i * config.hop_length;

            // Copy windowed signal to FFT input
            for (j = 0; j < config.frame_length+aux_n_fft; j++) {
                if (start + j < input.width) {
                    if (j < midpoint) {
                        win = config.window[j];
                    } else {
                        win = config.window[config.frame_length - 1 - j];
                    }
                    data_re[j] = channel_data[start + j] * win;
                } else {
                    data_re[j] = 0.0f; // Zero-padding if out of bounds
                }
                data_im[j] = 0.0f;
            }

            #if DEBUG_STFT
            sprintf(name_vector_debug, "senial_bloque_%d", i);
            printf_vector(name_vector_debug, data_re, config.frame_length);
            #endif // DEBUG_STFT

            // Compute FFT
            fft(data_re, data_im, config.frame_length+aux_n_fft);

            #if DEBUG_STFT
            sprintf(name_vector_debug, "fft_real_bloque_%d", i);
            printf_vector(name_vector_debug, data_re, config.frame_length);

            sprintf(name_vector_debug, "fft_imag_bloque_%d", i);
            printf_vector(name_vector_debug, data_im, config.frame_length);
            #endif // DEBUG_STFT

            // Compute magnitude spectrum
            for (j = 0; j < config.frame_length; j++) {
                aux_re = data_re[j];
                aux_im = data_im[j];
                data_re[j] = sqrt(aux_re * aux_re + aux_im * aux_im);
            }

            // Optionally convert to dB
            if (config.convert_to_db) {
                #define EPS 1e-8f
                // Encontrar máximo
                float max_mag = 0.0f;
                for (j = 0; j < config.n_fft_table; j++) {
                    if (data_re[j] > max_mag) max_mag = data_re[j];
                }
                float max_db = 20.0f * log10f(fmaxf(max_mag, EPS));
                float min_allowed_db = max_db - config.top_db;

                for (j = 0; j < config.n_fft_table; j++) {
                    float mag_db = 20.0f * log10f(fmaxf(data_re[j], EPS));
                    data_re[j] = (mag_db < min_allowed_db) ? min_allowed_db : mag_db;
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