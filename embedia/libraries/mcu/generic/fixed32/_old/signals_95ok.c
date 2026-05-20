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
/*
Test result:  95.000 %
 Acc. error: 1498.599
 Elem count: 1920
*/

#include <stdlib.h>
#include <math.h>
#include "signals.h"
#include <stdio.h>
/* ------------------------------ Spectrogram ------------------------------ */



/*
 * EmbedIA - stft_layer.c
 * Fixed-point Q16.16 implementation of multi_stft_layer.
 *
 * Fiel al comportamiento de signals.c (float):
 *   1. Resta de media del canal
 *   2. Ventana SIMÉTRICA: win[j] para j < midpoint, win[frame_len-1-j] para j >= midpoint
 *   3. FFT radix-2 DIT sin escalado (equivalente a numpy.fft, sin normalización)
 *   4. Magnitud: fixed_sqrt(re² + im²)  — NO potencia
 *   5. dB opcional: 20·log10(max(mag, EPS))
 *   6. Solo primer mitad del espectro: bins 0..n_fft_table-1
 *   7. Layout salida: [canal][frame][bin]
 *
 * Parámetros relevantes de spectrogram_layer_t:
 *   frame_length   = N del FFT (256 en el ejemplo)
 *   hop_length     = paso entre frames (128)
 *   n_frames       = número de frames (15)
 *   n_fft_table    = bins de salida = frame_length/2 (128)
 *   window         = ventana Q16.16, longitud frame_length
 *   convert_to_db  = 0 magnitud, 1 dB
 *   n_channels     = canales (input.height)
 *
 * NOTA: len_nfft_nmels no se usa — el FFT opera sobre frame_length muestras.
 */


/* -----------------------------------------------------------------------
 * Constantes internas
 * --------------------------------------------------------------------- */

/* EPS para dB equivalente a 1e-8f en float */
#define STFT_EPS      FL2FX_CONST(1e-8)

/* 20 en Q16.16 para conversión dB: 20·log10(mag) */
#define FIX_TWENTY    FL2FX_CONST(20.0)

/* log10(e) en Q16.16 */
#define FIX_LOG10E    FL2FX_CONST(0.4342944819032518)

/* -----------------------------------------------------------------------
 * Twiddle factors en dfixed para mayor precisión
 * --------------------------------------------------------------------- */

/* 2π en Q32.32 como literal entero sin double en runtime */
static const dfixed DFIX_2PI_CONST = (dfixed)26986075409LL;

static fixed twiddle_cos(dfixed angle_q32)
{
    while (angle_q32 < -DFIX_2PI_CONST) angle_q32 += DFIX_2PI_CONST;
    fixed fx = (fixed)((angle_q32 + (dfixed)0x8000) >> 16);
    return fixed_cos(fx);
}

static fixed twiddle_sin(dfixed angle_q32)
{
    while (angle_q32 < -DFIX_2PI_CONST) angle_q32 += DFIX_2PI_CONST;
    fixed fx = (fixed)((angle_q32 + (dfixed)0x8000) >> 16);
    return fixed_sin(fx);
}

/* -----------------------------------------------------------------------
 * Bit-reversal (in-place)
 * --------------------------------------------------------------------- */
static void bit_reverse(fixed *re, fixed *im, uint16_t N)
{
    uint16_t j = 0;
    for (uint16_t i = 1; i < N; i++) {
        uint16_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            fixed tmp;
            tmp = re[i]; re[i] = re[j]; re[j] = tmp;
            tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
    }
}

/* -----------------------------------------------------------------------
 * FFT radix-2 DIT — replica exacta de compute() en signals.c
 *
 * El algoritmo de signals.c inicializa twiddle = (1,0) para group=0
 * y actualiza para cada group siguiente con:
 *   angle = pi * (group+1) / step   (pi negativo en signals.c)
 * Replicamos eso exactamente.
 * --------------------------------------------------------------------- */
static void fft_fixed(fixed *re, fixed *im, uint16_t N)
{
    bit_reverse(re, im, N);

    for (uint16_t step = 1; step < N; step <<= 1) {
        const uint16_t jump = step << 1;

        fixed tw_re = FIX_ONE;
        fixed tw_im = FIX_ZERO;

        for (uint16_t group = 0; group < step; group++) {

            for (uint16_t pair = group; pair < N; pair += jump) {
                const uint16_t match = pair + step;

                fixed product_re = FIXED_MUL(tw_re, re[match])
                                 - FIXED_MUL(tw_im, im[match]);
                fixed product_im = FIXED_MUL(tw_im, re[match])
                                 + FIXED_MUL(tw_re, im[match]);

                re[match] = re[pair] - product_re;
                im[match] = im[pair] - product_im;
                re[pair]  = re[pair] + product_re;
                im[pair]  = im[pair] + product_im;
            }

            if (group + 1 == step) continue;

            /*
             * Replica exacta de compute() en signals.c:
             *   angle = pi * (group+1) / step_d   (pi negativo)
             *
             * Calculamos el numerador en Q16.16 y la división entera
             * se hace en dfixed para mantener precisión:
             *   angle = -FIX_PI * (group+1) / step
             *
             * Usamos multiplicación entera exacta: FIX_PI * (group+1)
             * no puede desbordar int32 para group < 128 (step <= 128).
             */
            dfixed num = (dfixed)FIX_PI * (dfixed)(group + 1);
            dfixed den = (dfixed)step << 16;  /* step en Q16.16 */
            dfixed angle_q32 = -(num << 16) / den;
            tw_re = twiddle_cos(angle_q32);
            tw_im = twiddle_sin(angle_q32);
        }
    }
}

/* -----------------------------------------------------------------------
 * Magnitud: sqrt(re² + im²)
 * --------------------------------------------------------------------- */
static inline fixed magnitude_bin(fixed re, fixed im)
{
    return fixed_sqrt(FIXED_ADD(FIXED_MUL(re, re), FIXED_MUL(im, im)));
}

/* -----------------------------------------------------------------------
 * dB: 20·log10(max(mag, EPS))
 * --------------------------------------------------------------------- */
static inline fixed mag_to_db(fixed mag)
{
    fixed m = (mag < STFT_EPS) ? STFT_EPS : mag;
    return FIXED_MUL(FIX_TWENTY, FIXED_MUL(fixed_log(m), FIX_LOG10E));
}

/* -----------------------------------------------------------------------
 * Procesamiento de un canal — replica signals.c por canal
 * --------------------------------------------------------------------- */
static void stft_channel(const fixed        *signal,
                         uint16_t            sig_len,
                         spectrogram_layer_t config,
                         fixed              *re,
                         fixed              *im,
                         fixed              *out_ch)
{
    const uint16_t frame_len = config.frame_length;
    const uint16_t hop       = config.hop_length;
    const uint16_t n_fr      = config.n_frames;
    const uint16_t n_bins    = config.n_fft_table;
    const fixed   *win       = config.window;
    const uint16_t midpoint  = frame_len / 2;

    /* ---- Restar media del canal (igual que signals.c) ---- */
    dfixed sum = 0;
    for (uint16_t i = 0; i < sig_len; i++) {
        sum += (dfixed)signal[i];
    }
    fixed mean = (fixed)(sum / (dfixed)sig_len);

    /* ---- Procesar cada frame ---- */
    for (uint16_t f = 0; f < n_fr; f++) {
        const uint16_t start = f * hop;

        for (uint16_t j = 0; j < frame_len; j++) {
            im[j] = FIX_ZERO;
            uint16_t sig_idx = start + j;
            if (sig_idx < sig_len) {
                fixed s = signal[sig_idx] - mean;
                /* Ventana simétrica — igual que signals.c */
                fixed w = (j < midpoint) ? win[j] : win[frame_len - 1 - j];
                re[j] = FIXED_MUL(s, w);
            } else {
                re[j] = FIX_ZERO;
            }
        }

        fft_fixed(re, im, frame_len);

        fixed *frame_out = out_ch + (uint32_t)f * n_bins;
        for (uint16_t k = 0; k < n_bins; k++) {
            fixed mag = magnitude_bin(re[k], im[k]);
            frame_out[k] = config.convert_to_db ? mag_to_db(mag) : mag;
        }
    }
}

/* -----------------------------------------------------------------------
 * Punto de entrada público
 * --------------------------------------------------------------------- */
void multi_stft_layer(spectrogram_layer_t config,
                      data2d_t            input,
                      data3d_t           *output)
{
    const uint16_t frame_len = config.frame_length;
    const uint16_t n_ch      = input.height;
    const uint16_t n_fr      = config.n_frames;
    const uint16_t n_bins    = config.n_fft_table;

    uint32_t sz_re  = (uint32_t)frame_len * sizeof(fixed);
    uint32_t sz_im  = (uint32_t)frame_len * sizeof(fixed);
    uint32_t sz_out = (uint32_t)n_ch * n_fr * n_bins * sizeof(fixed);

    void *ptr_re, *ptr_im, *ptr_out;
    swap_alloc_slice3(sz_re, sz_im, sz_out, &ptr_re, &ptr_im, &ptr_out);

    fixed *re      = (fixed *)ptr_re;
    fixed *im      = (fixed *)ptr_im;
    fixed *out_buf = (fixed *)ptr_out;

    /* Layout salida igual que signals.c:
     *   height   = n_frames
     *   width    = n_fft_table
     *   channels = input.height                        */
    output->height   = n_fr;
    output->width    = n_bins;
    output->channels = n_ch;
    output->data     = (real_t *)out_buf;

    for (uint16_t c = 0; c < n_ch; c++) {
        const fixed *sig_ptr =
            (const fixed *)input.data + (uint32_t)c * input.width;
        fixed *out_ch =
            out_buf + (uint32_t)c * n_fr * n_bins;
        stft_channel(sig_ptr, input.width, config, re, im, out_ch);
    }
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
