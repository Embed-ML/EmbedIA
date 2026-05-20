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
#include <stdio.h>
/* ------------------------------ Spectrogram ------------------------------ */



/*
 * void rearrange(fixed data_re[], fixed data_im[], const unsigned int N)
 * Performs the necessary reordering of the data before applying the FFT.
 * Parameters:
 *   - data_re: Array containing the real part of the complex data.
 *   - data_im: Array containing the imaginary part of the complex data.
 *   - N: Amount of samples to perform the FFT over.
 */
static inline void rearrange(int32_t data_re[], int32_t data_im[], const unsigned int N) {
    register unsigned int position;
    unsigned int target = 0;

    for(position = 0; position < N; position++) {
        if(target > position) {
            const int32_t temp_re = data_re[target];
            const int32_t temp_im = data_im[target];
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

// Macro optimizada para división de enteros pequeños (1-1024) en Q16.16
#define DFIXED_INT_DIV(A, B) \
    ((dfixed)(((dfixed)(A) << (2*FIX_FRC_SZ)) / (dfixed)(B)))


#define DFIXED_DMUL(A,B)            \
    ((dfixed)(((A)>>(FIX_FRC_SZ/2))*((B)>>(FIX_FRC_SZ/2)))>>FIX_FRC_SZ)

#define DFIX_PI FL2DFX(M_PI)

static inline void compute(int32_t data_re[], int32_t data_im[], const unsigned int N) {
    const dfixed pi = -DFIX_PI;
    const fixed N_INV = FIXED_DIV(FIX_ONE, N);

    register unsigned int step, group, pair;

    for(step = 1; step < N; step <<= 1) {
        const unsigned int jump = step << 1;
        int32_t twiddle_re = 65536; // Q16 int
        int32_t twiddle_im = 0;

        dfixed tmp_re, tmp_im;

        for(group = 0; group < step; group++) {
            
            if (group > 0) {
                float angle_fl = -3.14159265358979f * (float)group / (float)step;
                twiddle_re = (int32_t)(cosf(angle_fl) * 65536.0f);
                twiddle_im = (int32_t)(sinf(angle_fl) * 65536.0f);
            }

            for(pair = group; pair < N; pair += jump) {
                const unsigned int match = pair + step;
                
                // Multiplicar twiddle de Q16 por data de Q8 y regresar a Q8
                const int32_t product_re = (int32_t)( (((int64_t)twiddle_re * data_re[match]) - ((int64_t)twiddle_im * data_im[match]) + (1 << 15)) >> 16 );
                const int32_t product_im = (int32_t)( (((int64_t)twiddle_im * data_re[match]) + ((int64_t)twiddle_re * data_im[match]) + (1 << 15)) >> 16 );
                
                int32_t tr = data_re[pair];
                int32_t ti = data_im[pair];

                data_re[match] = tr - product_re;
                data_im[match] = ti - product_im;
                data_re[pair]  = tr + product_re;
                data_im[pair]  = ti + product_im;
            }
        }
    }
    for(unsigned int i = 0; i < N; i++) {
        data_re[i] = (int32_t)( ((int64_t)data_re[i] * N_INV + (1 << (FIX_FRC_SZ - 1))) >> FIX_FRC_SZ );
        data_im[i] = (int32_t)( ((int64_t)data_im[i] * N_INV + (1 << (FIX_FRC_SZ - 1))) >> FIX_FRC_SZ );
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
static inline void fft(int32_t data_re[], int32_t data_im[], const unsigned int N) {
    rearrange(data_re, data_im, N);
    compute(data_re, data_im, N);
}


static inline void apply_symm_window1(fixed* input, fixed* output_re, fixed* output_im,
                                   const fixed* window, unsigned int frame_length,
                                   unsigned int start, int gain_compensation) {
    const unsigned int midpoint = frame_length >> 1;
    const int right_shift = FIX_FRC_SZ - gain_compensation;
    unsigned int j;

    // Primera mitad
    for(j = 0; j < midpoint; j++) {
        dfixed temp = (dfixed)input[start + j] * window[j];
        output_re[j] = (fixed)(temp >> right_shift);
        output_im[j] = FIX_ZERO;
    }

    // Segunda mitad
    for(j = midpoint; j < frame_length; j++) {
        unsigned int window_idx = frame_length - 1 - j;
        dfixed temp = (dfixed)input[start + j] * window[window_idx];
        output_re[j] = (fixed)(temp >> right_shift);
        output_im[j] = FIX_ZERO;
    }
}

static inline void apply_symm_window(fixed* input,
                                   int32_t* output_re, int32_t* output_im,
                                   unsigned int input_size, // Se necesita el tamaño de entrada
                                   const fixed* window, unsigned int frame_length,
                                   unsigned int start, int gain_compensation) {
    const unsigned int aux_n_fft = 0; // Añadido aux_n_fft
    const unsigned int total_length = frame_length + aux_n_fft;
    const unsigned int midpoint = frame_length >> 1;
    const int right_shift = FIX_FRC_SZ - gain_compensation; // Ajuste por escala fija
    unsigned int j;

    // Iterar sobre toda la longitud de salida (incluyendo zero-padding)
    for (j = 0; j < total_length; j++) {
        // Verificar límites para el acceso al array de entrada
        if (start + j < input_size) {
            fixed win_value;
            // Aplicar ventana simétrica
            if (j < midpoint) {
                win_value = window[j];
            } else { // Asegurarse de que j < frame_length para el cálculo del índice
                win_value = window[frame_length - 1 - j];
            }

            dfixed temp = (dfixed)input[start + j] * win_value;
            output_re[j] = (int32_t)(temp >> right_shift);
            
        } else {
            output_re[j] = FIX_ZERO;
        }

        // Parte imaginaria siempre es cero
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

    // Temporary arrays for FFT input
    // fixed data_re[config.frame_length + aux_n_fft];
    // fixed data_im[config.frame_length + aux_n_fft];
    int32_t * data_re = (int32_t *) malloc((config.frame_length+aux_n_fft)*sizeof(int32_t));
    int32_t * data_im = (int32_t *) malloc((config.frame_length+aux_n_fft)*sizeof(int32_t));

    // output dimension and allocate
    output->height = config.n_frames;
    output->width = config.n_fft_table;
    output->channels = input.height;  // Un canal por cada canal de entrada
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->height * output->width * output->channels);

    // Procesar cada canal por separado
    for (c = 0; c < input.height; c++) {
        // Calcular media (usando dfixed para mayor precisión)
        dfixed sum = 0;
        for (i = 0; i < input.width; i++)
            sum += (dfixed)input.data[c * input.width + i];
        fixed mean = (fixed)(sum / input.width);

        // Subtract mean from input signal (like in Python)
        fixed channel_data[input.width];
        for (i = 0; i < input.width; i++) {
            channel_data[i] = FIXED_SUB(input.data[c * input.width + i], mean);
        }

        #if DEBUG_STFT
        printf("Channel %d:\n", c);
        printf_vector("senial", channel_data, input.width);
        #endif // DEBUG_STFT

        for (i = 0; i < config.n_frames; i++) {
            const unsigned int start = i * config.hop_length;

            // apply_symm_window1(channel_data, data_re, data_im, config.window, config.frame_length, start, 0);
            apply_symm_window(channel_data, data_re, data_im, input.width, config.window, config.frame_length, start, 0);

            #if DEBUG_STFT
            sprintf(name_vector_debug, "senial_bloque_%d", i);
            printf_vector(name_vector_debug, data_re, config.frame_length);
            #endif // DEBUG_STFT

            // Compute FFT
            fft(data_re, data_im, config.frame_length + aux_n_fft);

            #if DEBUG_STFT
            sprintf(name_vector_debug, "fft_real_bloque_%d", i);
            printf_vector(name_vector_debug, data_re, config.frame_length);

            sprintf(name_vector_debug, "fft_imag_bloque_%d", i);
            printf_vector(name_vector_debug, data_im, config.frame_length);
            #endif // DEBUG_STFT

            // Magnitud (sin normalización adicional)
            for (j = 0; j < config.n_fft_table; j++) {
                dfixed aux_re = (dfixed)data_re[j];
                dfixed aux_im = (dfixed)data_im[j];
                dfixed re_sq = (aux_re * aux_re) >> FIX_FRC_SZ;
                dfixed im_sq = (aux_im * aux_im) >> FIX_FRC_SZ;
                
                dfixed sum_sq = re_sq + im_sq; 
                
                // Adaptive scaling to prevent overflow when passing to 16-bit fixed_sqrt
                int shift = 0;
                while (sum_sq > FIX_MAX) {
                    sum_sq >>= 2; // Divide square by 4
                    shift++;      // Multiply sqrt by 2
                }
                
                data_re[j] = fixed_sqrt((fixed)sum_sq) << shift;
            }

            // Optionally convert to dB
            if (config.convert_to_db) {
                fixed log10_scale = FL2FX_CONST(20.0f);
                for (j = 0; j < config.n_fft_table; j++) {
                    fixed log_val = fixed_logn(data_re[j], FL2FX_CONST(10.0f));
                    data_re[j] = FIXED_MUL(log10_scale, log_val);
                }
            }

            // Store in output (only first half: 0 to n_fft//2)
            for (j = 0; j < config.n_fft_table; j++) {
                // La estructura de datos es [canal][bloque][frecuencia]
                output->data[(c * output->height * output->width) + (i * output->width) + j] = data_re[j];
            }

            #if DEBUG_STFT
            sprintf(name_vector_debug, "bloque_%d", i);
            printf_vector(name_vector_debug, data_re, config.n_fft_table);
            #endif // DEBUG_STFT
        }
    }

    #if DEBUG_STFT
    printf_vector("array", output->data, output->height * output->width * output->channels);
    #endif // DEBUG_STFT
    
    free(data_re); 
    free(data_im); 
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