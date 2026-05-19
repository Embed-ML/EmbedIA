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
#include "neural_net.h"

/*
 * compute_padding()
 *   Computes the amount of padding required for a convolutional layer with given stride, input size,
 *   filter size, and output size.
 * Parameters:
 *   stride => Stride value for the convolution
 *   in_size => Size of the input data
 *   filter_size => Size of the convolutional filter
 *   out_size => Expected output size
 */
static uint16_t compute_padding(int stride, int in_size, int filter_size, int out_size){
    int dilation_rate = 1;
    // int offset = 0;
    int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    int total_padding = ((out_size - 1) * stride + effective_filter_size - in_size);
    total_padding = total_padding > 0 ? total_padding : 0;
    // *offset = total_padding % 2;
    return total_padding / 2;
}

/*
 * calc_alloc_conv2d_output()
 *   Calculates the output size for a convolutional layer and allocates memory for the output data.
 * Parameters:
 * n_filters => Number of filters in the convolutional layer
 * kernel_sz => Size of the convolutional kernel
 * strides => Strides for the convolution
 * padding => Type of padding (VALID or SAME)
 * input => Input data for the convolution
 * output => Pointer to store the output data
 */
static void calc_alloc_conv2d_output(uint16_t n_filters, size2d_t kernel_sz, size2d_t strides, uint8_t padding, data3d_t input, data3d_t *output){
/*    if (padding == PAD_VALID){
        // effective_filter_size = (filter_size - 1) * dilation_rate + 1 for dilation_rate=1 => kernel size
        output->height = (input.height + strides.h - kernel_sz.h) / strides.h;
        output->width  = (input.width  + strides.w - kernel_sz.w) / strides.w;
    }else{
        output->height = (input.height + strides.h - 1) / strides.h;
        output->width  = (input.width  + strides.w - 1) / strides.w;
    }
    output->channels = n_filters; // total of output channels
    output->data = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );
*/}

/*
 * conv2d_strides_layer()
 *   Performs a 2D convolution operation with strides on the input data using the provided
 *   convolutional layer parameters. This implementation assumes no padding in order to
 *   optimize speed avoiding checking input/output limits
 * Parameters:
 *   layer => Convolutional layer with loaded filters
 *   input => Input data for the convolution
 *   output => Pointer to store the output data
 */
void conv2d_strides_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output){
 /*   int32_t delta, i,j,k,l, f_pos, i_pos;
    int16_t f, c;
    float value;

    // calculate output size and allocate memory
    calc_alloc_conv2d_output(layer.n_filters, layer.kernel, layer.strides, layer.padding, input, output);

    for(f=0; f<layer.n_filters; f++){
        delta = f*(output->height)*(output->width);

        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){
                value = 0;
                for(c=0; c<layer.channels; c++){
                    for(k=0; k<layer.kernel.h; k++){
                        for(l=0; l<layer.kernel.w; l++){
                            f_pos = (c*layer.kernel.h*layer.kernel.w)+k*layer.kernel.w+l;
                            i_pos = (c * input.height * input.width) +      // start of channel
                                    (i*layer.strides.h + k) * input.width + // start of row
                                    (j*layer.strides.w + l);                // offset from start

                            value += DEQUANTIZE(layer.filters[f].weights[f_pos], layer.qparam) * input.data[i_pos];
                        }
                    }
                }
                output->data[delta + i*output->width + j] = value + DEQUANTIZE(layer.filters[f].bias, layer.qparam);
            }
        }
    }*/
}

/*
 * conv2d_padding_layer()
 *   Performs a 2D convolution operation with padding on the input data using the provided convolutional
 *   layer parameters. This is a general implementation that assumes padding > 0 and strides >1
 * Parameters:
 *   layer => Convolutional layer with loaded filters
 *   input => Input data for the convolution
 *   output => Pointer to store the output data
 */
void conv2d_padding_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output){
 /*   int32_t delta, i,j,k,l, f_pos, i_pos;
    int16_t f, c, i_pad, j_pad, pad_h, pad_w;
    float value;

    // calculate output size and allocate memory
    calc_alloc_conv2d_output(layer.n_filters, layer.kernel, layer.strides, layer.padding, input, output);

    pad_h = compute_padding(layer.strides.h, input.height, layer.kernel.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.kernel.w, output->width);

    for(f=0; f<layer.n_filters; f++){
        delta = f*(output->height)*(output->width);

        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){
                value = 0;
                for(c=0; c<layer.channels; c++){
                    for(k=0; k<layer.kernel.h; k++){
                        for(l=0; l<layer.kernel.w; l++){
                            i_pad = i * layer.strides.h + k - pad_h;
                            j_pad = j * layer.strides.w + l - pad_w;
                            // Check for valid input access within padded bounds
                            if (i_pad >= 0 && i_pad < input.height && j_pad >= 0 && j_pad < input.width) {
                                f_pos = (c * layer.kernel.h * layer.kernel.w) + k * layer.kernel.w + l;
                                i_pos = (c * input.height * input.width) + i_pad * input.width + j_pad;
                                value += DEQUANTIZE(layer.filters[f].weights[f_pos], layer.qparam) * input.data[i_pos];
                            }
                        }
                    }
                }
                output->data[delta + i*output->width + j] = value + DEQUANTIZE(layer.filters[f].bias, layer.qparam);
            }
        }
    }*/
}

/*
 * conv2d_layer()
 *   Function in charge of applying the convolution of a filter layer on a given input data set.
 *   This implementation assumes no padding and strides = 1 in order to optimize speed avoiding
 *   checking input/output limits and most inner multiplication of loop related to strides
 * Parameters:
 *   layer => convolutional layer with loaded filters.
 *   input => input data of type data3d_t
 *   *output => pointer to the data3d_t structure where the result will be saved.
 */
void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output){
 /*   int32_t delta, i,j,k,l, f_pos, i_pos;
    int16_t f, c;
    float value;

    // calculate output size and allocate memory
    calc_alloc_conv2d_output(layer.n_filters, layer.kernel, layer.strides, layer.padding, input, output);

    for(f=0; f<layer.n_filters; f++){
        delta = f*(output->height)*(output->width);

        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){
                value = 0;
                for(c=0; c<layer.channels; c++){
                    for(k=0; k<layer.kernel.h; k++){
                        for(l=0; l<layer.kernel.w; l++){
                            f_pos = (c*layer.kernel.h*layer.kernel.w)+k*layer.kernel.w+l; // assumes strides=1
                            i_pos = (c * input.height * input.width) + // start of channel
                                    (i + k) * input.width +            // start of row
                                    (j + l);                           // offset from start

                            value += DEQUANTIZE(layer.filters[f].weights[f_pos],layer.qparam) * input.data[i_pos];
                        }
                    }
                }
                output->data[delta + i*output->width + j] = value + DEQUANTIZE(layer.filters[f].bias,layer.qparam);
            }
        }
    }*/
}

/*
 * depthwise()
 *   Performs the depthwise convolution operation in a depthwise separable convolution layer.
 * Parameters:
 *   layer => Separable convolutional layer parameters
 *   filter => Filter weights for the depthwise convolution
 *   input => Input data for the convolution
 *   *output => Pointer to store the output data
 */
 static void depthwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t * output){
  /*  uint32_t i,j,k,l,c, f_pos, i_pos, i_pad, j_pad;
    uint8_t pad_h, pad_w;
    float sum, value;

    pad_h = compute_padding(layer.strides.h, input.height, layer.depth_kernel_sz.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.depth_kernel_sz.w, output->width);

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            for (c = 0; c < layer.depth_channels; c++) {
                sum = 0;
                for (k = 0; k < layer.depth_kernel_sz.h; k++) {
                    for (l = 0; l < layer.depth_kernel_sz.w; l++) {
                            i_pad = i * layer.strides.h + k - pad_h;
                            j_pad = j * layer.strides.w + l - pad_w;
                            // Check for valid input access within padded bounds
                            if (i_pad >= 0 && i_pad < input.height && j_pad >= 0 && j_pad < input.width) {
                                f_pos = (c * layer.depth_kernel_sz.h * layer.depth_kernel_sz.w) + k * layer.depth_kernel_sz.w + l;
                                i_pos = (c * input.height * input.width) + i_pad * input.width + j_pad;
                                sum += DEQUANTIZE(filter.weights[f_pos], layer.qparam) * input.data[i_pos];
                            }
                    }
                }
                output->data[c*output->width*output->height + i*output->width + j] = sum;
            }
        }
    }*/
}

/*
 * pointwise()
 *   Performs the pointwise convolution operation in a depthwise separable convolution layer.
 * Parameters:
 *   layer => Separable convolutional layer parameters
 *   filter => Filter weights for the pointwise convolution
 *   input => Input data for the convolution
 *   *output => Pointer to store the output data
 *   delta => Offset for writing to the output data
 */
 static void pointwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t * output, uint32_t delta){
  /*  uint32_t i, j, c, i_pos;
    float sum;

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            sum = 0;
            for (c = 0; c < layer.point_channels; c++) {
                i_pos = (c * input.height * input.width) + (i * 1) * input.width + (j * 1);
                sum += (DEQUANTIZE(filter.weights[c], layer.qparam) * input.data[i_pos]);
            }
            output->data[delta + i*output->width + j] = sum + DEQUANTIZE(filter.bias, layer.qparam);
        }
    }*/
}



/*
 * separable_conv2d_layer()
 *  Function in charge of applying the convolution of a filter layer (conv_layer_t) on a given input data set.
 * Parameters:
 *  layer => convolutional layer with loaded filters.
 *  input => input data of type data3d_t
 *  *output => pointer to the data3d_t structure where the result will be saved.
 */

void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t * output){
 /*   uint32_t delta, i;
    data3d_t depth_output;

    calc_alloc_conv2d_output(layer.depth_channels, layer.depth_kernel_sz, layer.strides, layer.padding, input, &depth_output);

    depthwise(layer, layer.depth_filter, input, &depth_output);

    output->channels = layer.n_filters; //cantidad de filtros
    output->height   = depth_output.height;
    output->width    = depth_output.width;
    output->data     = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );

    for(i=0; i<layer.n_filters; i++){
        delta = i*(output->height)*(output->width);
        pointwise(layer, layer.point_filters[i], depth_output,output,delta);
    }*/
}

/*
 * depthwise_bias()
 *   Performs the depthwise convolution operation with bias in a depthwise convolutional layer.
 * Parameters:
 *   layer => Depthwise convolutional layer parameters
 *   input => Input data for the convolution
 *   *output => Pointer to store the output data
 */
static void depthwise_bias(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output){
 /*   uint32_t i, j, k, l, c, f_pos, i_pos, pad_h, pad_w, j_pad, i_pad;
    float sum;

    pad_h = compute_padding(layer.strides.h, input.height, layer.kernel_sz.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.kernel_sz.w, output->width);

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            for (c = 0; c < layer.channels; c++) {
                sum = 0;
                for (k = 0; k < layer.kernel_sz.h; k++) {
                    for (l = 0; l < layer.kernel_sz.w; l++) {

                            i_pad = i * layer.strides.h + k - pad_h;
                            j_pad = j * layer.strides.w + l - pad_w;
                            // Check for valid input access within padded bounds
                            if (i_pad >= 0 && i_pad < input.height && j_pad >= 0 && j_pad < input.width) {
                                f_pos = (c * layer.kernel_sz.h * layer.kernel_sz.w) + k * layer.kernel_sz.w + l;
                                i_pos = (c * input.height * input.width) + i_pad * input.width + j_pad;
                                sum += DEQUANTIZE(layer.weights[f_pos], layer.w_qparam) * input.data[i_pos];
                              }


                    }
                }
                output->data[c*output->width*output->height + i*output->width + j]= sum + DEQUANTIZE(layer.bias[c], layer.b_qparam);
            }
        }
    }*/
}


/*
 * depthwise_conv2d_layer()
 *  Function in charge of applying the depthwise of a filter layer with bias (depthwise_conv2d_layer_t) on a given input data set.
 * Parameters:
 *  layer => depthwise layer with loaded filters.
 *  input => input data of type data3d_t
 *  *output => pointer to the data3d_t structure where the result will be saved.
 */

void depthwise_conv2d_layer(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output){

    calc_alloc_conv2d_output(layer.channels, layer.kernel_sz, layer.strides, layer.padding, input, output);

    depthwise_bias(layer, input, output);
}



/*
 * dense_layer()
 *  Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *  dense_layer => structure with the weights of the neurons of the dense layer.
 *  input       => structure data1d_t with the input data to process.
 *  *output     => structure data1d_t to store the output result.
 */
/*void dense_layer(dense_layer_t *layer, data1d_t *input, data1d_t * output){
    // Configurar salida
    output->length = layer->output_size;
    output->qparam = layer->output_qparam;
    output->data = (quant8*)swap_alloc(sizeof(quant8)*output->length);

    // MAC (Multiply-Accumulate) vectorizado
    for (uint16_t i = 0; i < layer->output_size; i++) {
        int32_t acc = 0;

        // Descuantizar bias (una vez por neurona)
        int32_t bias = (layer->biases[i] - layer->weights_qparam.zero_point)
                      * layer->weights_qparam.scale;

        // Producto punto input · weights_col
        for (uint16_t j = 0; j < layer->input_size; j++) {
            int32_t w = layer->weights[i * layer->input_size + j] - layer->weights_qparam.zero_point;
            int32_t x = input->data[j] - input->qparam.zero_point;
            acc += (w * x) >> Q_FRAC_BITS;
        }

        // Requantización a int8
        int32_t scaled_acc = (acc * layer->output_qparam.inv_scale) >> Q_FRAC_BITS;
        output->data[i] = (quant8)(scaled_acc + layer->output_qparam.zero_point);
    }
}*/
void dense_layer(dense_layer_t *layer, data1d_t *input, data1d_t *output) {
    output->length = layer->output_size;
    output->qparam = layer->output_qparam;
    output->data = (quant8*)swap_alloc(sizeof(quant8) * output->length);

    // Escala efectiva: (input_scale * weight_scale) / output_scale
    int64_t scale_mul = (int64_t)input->qparam.scale * layer->weights_qparam.scale;
    int32_t effective_scale = (int32_t)((scale_mul + (layer->output_qparam.scale / 2)) / layer->output_qparam.scale);
    // Está en Q_FRAC_BITS (Q17 en tu caso)

    for (uint16_t i = 0; i < layer->output_size; i++) {
        int32_t acc = 0;

        for (uint16_t j = 0; j < layer->input_size; j++) {
            int32_t w = (int32_t)layer->weights[i * layer->input_size + j] - layer->weights_qparam.zero_point;
            int32_t x = (int32_t)input->data[j] - input->qparam.zero_point;
            acc += w * x;  // Producto escalar, sin escalar todavía
        }

        // Sumar bias si está presente (bias en int32_t ya escalado al mismo dominio que acc)
        if (layer->biases) {
            acc += layer->biases[i];
        }

        // Escalar acumulador (Q0.30) con escala efectiva en Q_FRAC_BITS → resultado sigue en Q0.30
        int64_t scaled = (int64_t)acc * effective_scale;

        // Redondear antes de hacer shift
        int32_t shifted = (int32_t)((scaled + (1LL << (Q_FRAC_BITS - 1))) >> Q_FRAC_BITS);

        // Agregar zero_point de salida y saturar a int8
        output->data[i] = Q_CLAMP(shifted + output->qparam.zero_point);
    }
}






/*
 * max_pooling2d_layer()
 *  Maxpooling layer, for now supports square size and stride. No support for padding
 * Parameters:
 *  pool_size => size for pooling
 *  stride    => stride for pooling
 *  input     => input data
 *  output    => output data
 */
void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
  /*  uint32_t c,i,j,aux1,aux2;
    float max = -INFINITY;
    float num;

    output->height = ((uint16_t) ((input.height - pool.size)/pool.strides)) + 1;
    output->width  = ((uint16_t) ((input.width - pool.size)/pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (float*)swap_alloc(sizeof(float)*(output->channels)*(output->height)*(output->width));

    for(c=0; c<output->channels; c++){
        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){

                max = -INFINITY;

                for(aux1=0; aux1<pool.size; aux1++){
                        for(aux2=0; aux2<pool.size; aux2++){

                        num = input.data[c*input.width*input.height + (i*pool.strides + aux1)*input.width + j*pool.strides + aux2];

                        if(num>max){
                            max = num;
                        }
                    }
                }

                output->data[c*output->width*output->height + i*output->width + j] = max;
            }
        }
    }*/
}


/*
 * average_pooling_2d()
 *  Function that applies an average pooling to an input with a window size of received
 *  by parameter (uint16_t strides)
 * Parameters:
 *  input => input data of type data3d_t.
 *  *output => pointer to the data3d_t structure where the result will be stored.
 */

void average_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
/*    uint32_t c,i,j,aux1,aux2;
    uint32_t cant = pool.size*pool.size;
    float avg = 0;
    float num;

    // output->height = (input.height)/strides ;
    // output->width =  (input.width )/strides ;
    output->height = ((uint32_t) ((input.height - pool.size)/pool.strides)) + 1;
    output->width  = ((uint32_t) ((input.width - pool.size)/pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (float*)swap_alloc(sizeof(float)*(output->channels)*(output->height)*(output->width));

    for(c=0; c<output->channels; c++){
        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){

                avg = 0;

                for(aux1=0; aux1<pool.size; aux1++){
                    for(aux2=0; aux2<pool.size; aux2++){
                        num = input.data[c*input.width*input.height + (i*pool.strides + aux1)*input.width + j*pool.strides + aux2];
                        avg += num;
                    }
                }

                output->data[c*output->width*output->height + i*output->width + j] = avg/cant;
            }
        }
    }*/
}

/*
 * softmax activation function
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
// ==== VERSIÓN ULTRA OPTIMIZADA CON LOOKUP INLINE ====
// Para casos donde el rango de entrada es predecible: softmax_shift_inplace_ultra
/*void softmax_activation(int8_t* data, uint32_t length, qparam_t* qparam) {

    // 1. Encontrar máximo con optimización de loop
    int8_t max_val = data[0];
    for (int i = 1; i < length; i++) {
        max_val = (data[i] > max_val) ? data[i] : max_val;  // Branchless
    }

    // 2. Lookup table inline para shifts comunes (mejor que cálculo)
    static const uint16_t shift_lut[9] = {
        256, 128, 64, 32, 16, 8, 4, 2, 1  // 2^(8-i) para i=0..8
    };

    uint32_t sum = 0;

    // 3. Primera pasada optimizada
    for (int i = 0; i < length; i++) {
        int16_t diff = data[i] - max_val;  // diff <= 0

        uint16_t exp_approx;
        if (diff <= -8) {
            exp_approx = 1;
        } else {
            exp_approx = shift_lut[-diff];  // LUT en lugar de shift
        }
        sum += exp_approx;
    }

    // 4. Segunda pasada con división optimizada
    uint32_t inv_sum = (1UL << 24) / sum;  // Precalcular 1/sum en fixed point

    for (int i = 0; i < length; i++) {
        int16_t diff = data[i] - max_val;

        uint16_t exp_approx = (diff <= -8) ? 1 : shift_lut[-diff];

        // Usar multiplicación en lugar de división
        uint32_t normalized = (exp_approx * inv_sum) >> 16;

        int32_t quantized = (normalized * qparam->inv_scale) >> 8;
        quantized += qparam->zero_point;

        data[i] = (int8_t)((quantized > 127) ? 127 :
                          (quantized < -128) ? -128 : quantized);
    }
}*/
void softmax_activation(int8_t* data, uint32_t length, qparam_t* qparam) {
    // 1. Encontrar máximo (optimizado)
    int8_t max_val = data[0];
    for (uint32_t i = 1; i < length; i++) {
        max_val = (data[i] > max_val) ? data[i] : max_val;  // Branchless
    }

    // 2. LUT para exponencial (2^-x)
    static const uint16_t exp_lut[9] = {256, 128, 64, 32, 16, 8, 4, 2, 1}; // 2^(8-i)

    // 3. Primera pasada: calcular sum(exp(x - max_val))
    uint32_t sum = 0;
    for (uint32_t i = 0; i < length; i++) {
        int16_t diff = data[i] - max_val;
        uint16_t exp_approx = (diff <= -8) ? 1 : exp_lut[-diff];
        sum += exp_approx;
    }

    // 4. Precalcular 1/sum en punto fijo (Q24.8)
    uint32_t inv_sum = (1UL << 24) / sum;  // Equivale a (1.0/sum) * 2^24

    // 5. Segunda pasada: normalizar y cuantizar
    for (uint32_t i = 0; i < length; i++) {
        int16_t diff = data[i] - max_val;
        uint16_t exp_approx = (diff <= -8) ? 1 : exp_lut[-diff];

        // Normalización (Q24.8 * Q8.8 -> Q32.16, luego >>16 para Q16.0)
        uint32_t normalized = (exp_approx * inv_sum) >> 16;

        // Requantización: (Q16.0 * scale_fixed) >> Q_FRAC_BITS
        int32_t quantized = (normalized * qparam->scale) >> Q_FRAC_BITS;
        quantized += qparam->zero_point;

        // Saturación a int8_t
        data[i] = (int8_t)((quantized > 127) ? 127 :
                          (quantized < -128) ? -128 : quantized);
    }
}



/*
 * relu activation function
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 *  qp => quantization parameters
 */
void relu_activation(quant8 *data, uint32_t length, qparam_t* qp) {
    uint32_t i;
    const int8_t zero_point = qp->zero_point;

    for(i = 0; i < length; i++) {
        // Para datos cuantizados, el "0" real corresponde al zero_point
        data[i] = data[i] < zero_point ? zero_point : data[i];
    }
}

/* // relu con saturacion
void relu_activation_sat(quant8 *data, uint32_t length, qparam_t qp) {
    uint32_t i;
    const int8_t zero_point = qp.zero_point;
    const int8_t max_val = 127; // Valor máximo para int8_t

    for(i = 0; i < length; i++) {
        data[i] = (data[i] < zero_point) ? zero_point :
                 (data[i] > max_val) ? max_val : data[i];
    }
}

// Versión optimizada para ARM Cortex-M (usando instrucciones SIMD):
void relu_activation_opt(quant8 *data, uint32_t length, qparam_t qp) {
    uint32_t i;
    const int8x16_t zero_point_v = vdupq_n_s8(qp.zero_point);
    uint32_t chunks = length / 16;

    // Procesamiento en bloques de 16 elementos
    for(i = 0; i < chunks * 16; i += 16) {
        int8x16_t vec = vld1q_s8(data + i);
        int8x16_t res = vmaxq_s8(vec, zero_point_v);
        vst1q_s8(data + i, res);
    }

    // Procesar elementos restantes
    for(i = chunks * 16; i < length; i++) {
        data[i] = data[i] < qp.zero_point ? qp.zero_point : data[i];
    }
}
*/

/*
 * relu6 activation function (float version)
 * Parameters:
 *  *data  => array of float values to update
 *  length => number of values to update
 */
void relu6_activation(float *data, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        if (data[i] < 0.0)
            data[i] = 0.0;
        else if (data[i] > 6.0)
            data[i] = 6.0;
    }
}
/*
 * leaky relu activation function
 * Parameters:
 *  alfa   => coefficient to multiply negative values
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void leakyrelu_activation(float *data, uint32_t length, float alpha){
    uint32_t i;

    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? alpha*data[i] : data[i];
    }
}


/*
 * tanh activation function: (2 / (1+e^(-2x)) -1
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void tanh_activation(float *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = 2/(1+exp(-2*data[i])) - 1;
    }
}

/*
 * sigmoid activation function: 1 / (1 + exp(-x))
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void sigmoid_activation(quant8 *data, uint32_t length, qparam_t *qp) {
    // Asume que los datos ya están preescalados adecuadamente
    const fixed scale = qp->scale;
    const fixed zero_point = (fixed)qp->zero_point << Q_FRAC_BITS;

    uint32_t i;

    for(i = 0; i < length; i++) {
        fixed input = ((fixed)data[i] << Q_FRAC_BITS) - zero_point;

        // Versión simplificada con exp() preajustada
        fixed exp_val = fixed_exp(-(input >> 2));  // Reduce rango dinámico

        fixed denominator = FIX_ONE + exp_val;
        fixed output = FIXED_DIV(FIX_ONE, denominator);

        // Re-escalado rápido con saturación
        int32_t result = ((output * scale) >> Q_FRAC_BITS) + qp->zero_point;
        data[i] = Q_CLAMP(result);
    }
}

/*
 * softsign activation function: x / (|x| + 1)
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softsign_activation(float *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = data[i] / (fabs(data[i])+1);
    }
}

/*
 * softplus activation function: log(e^x + 1)
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softplus_activation(float *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = log( exp(data[i])+1 );
    }
}


/*
 * flatten3d_layer()
 *  Performs a variable shape change.
 *  Converts the data format from data3d_t array format to data1d_t vector.
 *  (prepares data for input into a layer of type dense_layer_t).
 * Parameters:
 *  input => input data of type data3d_t.
 *  *output => pointer to the data1d_t structure where the result will be stored.
 */
void flatten3d_layer(data3d_t input, data1d_t * output){
 /*   uint32_t c,i,j;
    uint32_t cantidad = 0;

    output->length = input.channels * input.height * input.width;
    output->data = (float*)swap_alloc(sizeof(float)*output->length);

    for(i=0;i<input.height;i++){
        for(j=0;j<input.width;j++){
            for(c=0;c<input.channels;c++){
                output->data[cantidad++] = input.data[(c*input.width*input.height)+(i*input.width)+j];
            }
        }
    }*/
}



/*********************************************************************************************************************************/
/* Normalization layers */


void normalization1(normalization_layer_t n, data1d_t input, data1d_t * output){
/*
    uint32_t i;

    output->length = input.length;
    output->data = (float*)swap_alloc(sizeof(float)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = (input.data[i]-n.sub_val[i])*n.inv_div_val[i];
    }*/
}

void normalization2(normalization_layer_t n, data1d_t input, data1d_t * output){
/*
    uint32_t i;

    output->length = input.length;
    output->data = (float*)swap_alloc(sizeof(float)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = input.data[i]*n.inv_div_val[i];
    }*/
}

/*
 * batch_normalization{X}d_layer()
 * Keras Batch Normalization
 * Parameters:
 *      batch_normlization_t norm =>  structure with batch normalization layer parameters
 *      *data  =>   pointer to data{X}d_t
 */

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data) {
   /* uint32_t i;

    for(i = 0; i < data->length; i++) {
        data->data[i] = data->data[i] * layer.moving_inv_std_dev[i] + layer.std_beta[i];
    }*/
}


void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data) {
/*    uint32_t i, j, ilen = 0;
    uint32_t length = data->height * data->width;
    float scale, offset;

    for(i = 0; i < data->channels; i++, ilen += length) {
        scale = DEQUANTIZE(layer.moving_inv_std_dev[i],layer.mov_qparam);
        offset= DEQUANTIZE(layer.std_beta[i], layer.std_qparam);
        for(j = 0; j < length; j++) {
            data->data[ilen+j] = data->data[ilen+j] * scale + offset;
        }
    }*/
}



/*
 * void initialize_zero_padding(uint8_t pad_h, uint8_t pad_w, data3d_t *output)
 * Initializes the zero-padding areas in the given 3D data structure with zeros.
 * Parameters:
 *   - pad_h: Number of zero-padding rows at the top and bottom.
 *   - pad_w: Number of zero-padding columns at the left and right.
 *   - output: Pointer to a 3D data structure where the zero-padding will be initialized.
 * Description:
 *   This function initializes the zero-padding areas in a 3D data structure with zeros.
 *   It adds the specified number of zero rows at the top and bottom (pad_h) and zero columns
 *   at the left and right (pad_w). The initialization is performed in-place on the output data.
 */
static void zero_padding2d_init(uint8_t pad_h, uint8_t pad_w, data3d_t *output){
 /*   uint32_t c, i, j;

    for (c = 0; c < output->channels; c++) {
        for (i = 0; i < output->height; i++) {
            for (j = 0; j < pad_w; j++) {
                output->data[(c * output->height + i) * output->width + j] = 0.0; // left fill
                output->data[(c * output->height + i) * output->width + output->width - 1 - j] = 0.0; // right fill
            }
        }
    }

    for (c = 0; c < output->channels; c++) {
        for (i = 0; i < pad_h; i++) {
            // top fill
            for (j = 0; j < output->width; j++) {
                output->data[(c * output->height + i) * output->width + j] = 0.0;
            }
            // bottom fill
            for (j = 0; j < output->width; j++) {
                output->data[(c * output->height + output->height - 1 - i) * output->width + j] = 0.0;
            }
        }
    }*/
}

/*
 * void zero_padding2d_layer(uint8_t pad_h, uint8_t pad_w, data3d_t input, data3d_t *output)
 * Applies zero-padding to a 2D input data array.
 * Parameters:
 *   - pad_h: Number of zero-padding rows to add at the top and bottom.
 *   - pad_w: Number of zero-padding columns to add at the left and right.
 *   - input: 3D data structure representing the input data.
 *   - output: Pointer to a 3D data structure where the zero-padded output will be stored.
 * Description:
 *   This function performs zero-padding on a 2D input data array. It adds the specified
 *   number of zero rows at the top and bottom (pad_h) and zero columns at the left and right (pad_w).
 *   The result is stored in the output data structure.
 */
void zero_padding2d_layer(uint8_t pad_h, uint8_t pad_w, data3d_t input, data3d_t *output) {
 /*   uint32_t c, i, j, output_index, input_index;

    // Calc output dimension
    output->channels = input.channels;
    output->width = input.width + 2 * pad_w;
    output->height = input.height + 2 * pad_h;

    size_t output_size = output->channels * output->width * output->height;
    output->data = (float *)swap_alloc(output_size * sizeof(float));

    // Copy input data to the center of output data
    for (c = 0; c < input.channels; c++) {
        for (i = 0; i < input.height; i++) {
            for (j = 0; j < input.width; j++) {
                output_index = (c * output->height + (i + pad_h)) * output->width + j + pad_w;
                input_index = (c * input.height + i) * input.width + j;
                output->data[output_index] = input.data[input_index];
            }
        }
    }

    zero_padding2d_init(pad_h, pad_w, output);*/
}

/* channel_adapt_layer()
 *  Converts Tensorflow/Keras Image (Height, Width, Channel) to Embedia format (Channel, Height, Width).
 *  Usually required for first convolutional layer
 * Parameters:
 *  input   => input data of type data3d_t.
 *  *output => pointer to the data3d_t structure where the result will be stored.
 */
void channel_adapt_layer(data3d_t input, data3d_t * output){

  /*  uint32_t i, j, c, l;

    output->channels = input.channels;
    output->height   = input.height;
    output->width    = input.width;
    output->data     = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );

    for(c=0, l=0; c < input.channels; c++){
        for(i=0; i < input.height; i++) {
            for(j=0; j < input.width; j++, l++ ){
                output->data[l] = input.data[i*input.channels*input.width+input.channels*j+c];
            }
        }
    }*/
 }
