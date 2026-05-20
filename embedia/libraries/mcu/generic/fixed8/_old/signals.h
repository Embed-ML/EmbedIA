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

#ifndef _SIGNALS_H
#define _SIGNALS_H

#include <stdint.h>
#include "common.h"
#include "fixed.h"

/*
 * spectrogram_layer_t struct
 *
 * Defines the necessary configuration to generate a spectrogram from an
 * audio signal. It groups all the key parameters into a single data type
 * to facilitate passing to the processing functions.
 *
 * Fields:
 * - n_channels: number of audio channels to process.
 * - sample_rate: sampling rate in Hz.
 * - ts_us: time step between samples in microseconds (µs).
 *
 * - frame_length: number of samples per frame.
 * - overlap_length: number of overlapping samples between frames.
 * - hop_length: number of samples to advance between frames (frame_length - overlap_length).
 * - window: pointer to the window function (e.g., Hann) to apply to each frame.
 *
 * - n_fft_table: number of FFT points used for the transform (may be >= frame_length).
 * - len_nfft_nmels: number of bins resulting from the FFT (e.g., n_fft_table / 2 + 1, or mel bins).
 *
 * - n_frames: number of time frames in the output spectrogram.
 * - spec_size: total number of values in the resulting spectrogram (e.g., n_frames × len_nfft_nmels).
 *
 * - convert_to_db: if non-zero, output is converted to decibel scale.
 */
/* Q0.8 window format */
typedef uint8_t window_t;

typedef struct {
    // Input parameters
    uint16_t n_channels;
    uint16_t sample_rate;
    uint16_t ts_us;

    // windows & transform config
    uint16_t frame_length;
    uint16_t overlap_length;
    uint16_t hop_length;
    const window_t *window;
    uint16_t window_shift;

    // FFT & spectrogram config
    uint16_t n_fft_table;
    uint16_t len_nfft_nmels;

    // output dimensions
    uint16_t n_frames;
    uint16_t spec_size;

    // Post-processing
    uint16_t convert_to_db;
    fixed top_db;
} spectrogram_layer_t;

/* Signal processing */

/* void fft(float data_re[], float data_im[], const unsigned int N)
 * Performs a Fast Fourier Transform (FFT) on the complex data passed as parameters.
 * Parameters:
 *  - data_re: Array containing the real part of the complex data.
 *  - data_im: Array containing the imaginary part of the complex data.
 *  - N: Amount of samples to perform the FFT over.
 */
//void fft(fixed data_re[], fixed data_im[], const unsigned int N);

/*
 * void rearrange(float data_re[], float data_im[], const unsigned int N)
 * Performs the necessary reordering of the data before applying the FFT.
 * Parameters:
 *  - data_re: Array containing the real part of the complex data.
 *  - data_im: Array containing the imaginary part of the complex data.
 *  - N: Amount of samples to perform the FFT over.
 */
//void rearrange(fixed data_re[],fixed data_im[],const unsigned int N);

/*
 * void compute(float data_re[], float data_im[], const unsigned int N)
 * Contains the FFT calculation core, applying the Fourier transforms for
 * each recursive step.
 * Parameters:
 *  - data_re: Array containing the real part of the complex data.
 *  - data_im: Array containing the imaginary part of the complex data.
 *  - N: Amount of samples to perform the FFT over.
 */
//void compute(fixed data_re[], fixed data_im[],const unsigned int N);

/*
 * void create_spectrogram(spectrogram_layer_t config, data1d_t input, data3d_t *output)
 * Generates the spectrogram from the input signal by applying FFTs
 * and further processing.
 * Parameters:
 *  - config: Spectrogram layer configuration
 *  - input:  2D input signal (channels, samples)
 *  - output: 3D output spectrogram (channels, , H = b_blocks, Ch = 1)
 */
void multi_stft_layer(spectrogram_layer_t config, data2d_t input, data3d_t * output);

void stft_layer(spectrogram_layer_t config, data1d_t input, data2d_t * output);


#endif