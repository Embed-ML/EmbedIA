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
    if (padding == PAD_VALID){
        // effective_filter_size = (filter_size - 1) * dilation_rate + 1 for dilation_rate=1 => kernel size
        output->height = (input.height + strides.h - kernel_sz.h) / strides.h;
        output->width  = (input.width  + strides.w - kernel_sz.w) / strides.w;
    }else{
        output->height = (input.height + strides.h - 1) / strides.h;
        output->width  = (input.width  + strides.w - 1) / strides.w;
    }
    output->channels = n_filters; // total of output channels
    output->data = (fixed*)swap_alloc( sizeof(fixed)*output->channels*output->height*output->width );
}

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
    int32_t delta, i,j,k,l, f_pos, i_pos;
    int16_t f, c;
    fixed value;

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

                            value += FIXED_MUL(layer.filters[f].weights[f_pos], input.data[i_pos]);
                        }
                    }
                }
                output->data[delta + i*output->width + j] = value + layer.filters[f].bias;
            }
        }
    }
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
    int32_t delta, i,j,k,l, f_pos, i_pos;
    int16_t f, c, i_pad, j_pad, pad_h, pad_w;
    dfixed value;

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
                                value += DFIXED_MUL(layer.filters[f].weights[f_pos], input.data[i_pos]);
                            }
                        }
                    }
                }
                value = value + FIXED_TO_DFIXED(layer.filters[f].bias);
                if (value > DFIX_MAX)
                    value = FIX_MAX;
                else if (value < DFIX_MIN)
                    value = FIX_MIN;
                else value = DFIXED_TO_FIXED(value);

                output->data[delta + i*output->width + j] = value;
		    }
		}
	}
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
    int32_t delta, i,j,k,l, f_pos, i_pos;
    int16_t f, c;
    fixed value;

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

                            value += FIXED_MUL(layer.filters[f].weights[f_pos], input.data[i_pos]);
                        }
                    }
                }
                output->data[delta + i*output->width + j] = value + layer.filters[f].bias;
            }
        }
    }
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
static void depthwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t *output) {
    uint32_t i, j, k, l, c, f_pos, i_pos, pad_h, pad_w, j_pad, i_pad;
    dfixed sum;

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
                                sum += DFIXED_MUL(filter.weights[f_pos], input.data[i_pos]);
                            }
                    }
                }
				if (sum > DFIX_MAX)
					sum = FIX_MAX;
				else if (sum < DFIX_MIN)
					sum = FIX_MIN;
				else sum = DFIXED_TO_FIXED(sum);

				output->data[c*output->width*output->height + i*output->width + j] = sum;
			}
		}
	}
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
static void pointwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t *output, uint32_t delta) {
    uint32_t i, j, c, i_pos;
    dfixed sum;

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            sum = 0;
            for (c = 0; c < layer.point_channels; c++) {
                i_pos = (c * input.height * input.width) + (i * 1) * input.width + (j * 1);
                sum += DFIXED_MUL(filter.weights[c], input.data[i_pos]);
            }
			sum = sum + FIXED_TO_DFIXED(filter.bias);
			if (sum > DFIX_MAX)
				sum = FIX_MAX;
			else if (sum < DFIX_MIN)
				sum = FIX_MIN;
			else sum = DFIXED_TO_FIXED(sum);
			
			output->data[delta + i*output->width + j] = sum;
        }
    }
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
    uint32_t delta, i;
    data3d_t depth_output;

    calc_alloc_conv2d_output(layer.depth_channels, layer.depth_kernel_sz, layer.strides, layer.padding, input, &depth_output);

    depthwise(layer, layer.depth_filter, input, &depth_output);

    output->channels = layer.n_filters; //cantidad de filtros
    output->height   = depth_output.height;
    output->width    = depth_output.width;
    output->data     = (fixed*)swap_alloc( sizeof(fixed)*output->channels*output->height*output->width );

    for(i=0; i<layer.n_filters; i++){
        delta = i*(output->height)*(output->width);
        pointwise(layer, layer.point_filters[i], depth_output,output,delta);
    }
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
    uint32_t i, j, k, l, c, f_pos, i_pos, pad_h, pad_w, j_pad, i_pad;
    fixed sum;

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
                                sum += FIXED_MUL(layer.weights[f_pos], input.data[i_pos]);
                            }


                    }
                }
                output->data[c * output->width * output->height + i * output->width + j] = sum + layer.bias[c];
            }
        }
    }
}

/*
 * Depthwise Convolution 2D Layer
 * - Implements channel-wise spatial convolution with independent filters
 * - Each input channel has its own set of filter weights
 * - More efficient than standard convolution for depthwise operations
 *
 *   Input tensor in data3d_t format (channels, height, width)
 *   output  => Pointer to output tensor (pre-allocated by calc_alloc_conv2d_output)
 *
 * Operation:
 *   1. Calculates output dimensions and allocates memory
 *   2. Applies depthwise convolution with per-channel bias
 */
void depthwise_conv2d_layer(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t * output) {
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
void dense_layer(dense_layer_t *layer, data1d_t *input, data1d_t *output) {
    output->length = layer->output_size;
    output->data = (fixed*)swap_alloc(sizeof(fixed) * output->length);

    for (uint32_t i = 0; i < layer->output_size; i++) {
        // Get the weights for the i-th neuron: strides across input_size
        const fixed *neuron_weights = &layer->weights[i * layer->input_size];
        const dfixed res = dot_product_bias(
            neuron_weights,           // Peso del i-ésimo neurón
            input->data,              // Datos de entrada
            input->length,            // Tamaño del vector de entrada
            layer->biases[i]          // Bias del i-ésimo neurón
        );

        if (res> DFIX_MAX)
            output->data[i] = FIX_MAX;
        else if (res < DFIX_MIN)
                 output->data[i] = FIX_MIN;
             else
                 output->data[i] = DFIXED_TO_FIXED(res);
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
    uint32_t c,i,j,aux1,aux2;
    fixed max = -FIX_MAX;
    fixed num;

    // output->height = (input.height)/pool_size ;
    // output->width =  (input.width )/pool_size ;
    output->height = ((uint16_t) ((input.height - pool.size)/pool.strides)) + 1;
    output->width  = ((uint16_t) ((input.width - pool.size)/pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*(output->channels)*(output->height)*(output->width));

    for(c=0; c<output->channels; c++){
        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){

                max = -FIX_MAX;

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
    }
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
    uint32_t c,i,j,aux1,aux2;
    dfixed cant = INT_TO_FIXED(pool.size*pool.size);
    dfixed avg = 0;
    fixed num;

    // output->height = (input.height)/strides ;
    // output->width =  (input.width )/strides ;
    output->height = ((uint32_t) ((input.height - pool.size)/pool.strides)) + 1;
    output->width  = ((uint32_t) ((input.width - pool.size)/pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*(output->channels)*(output->height)*(output->width));

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

                output->data[c*output->width*output->height + i*output->width + j] = FIXED_DIV(avg,cant);
            }
        }
    }
}

/*
 * softmax activation function
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softmax_activation(fixed *data, uint32_t length){
    uint32_t i;
    fixed m = -FIX_MAX;

    for(i = 0; i < length; i++) {
        if (data[i] > m) {
            m = data[i];
        }
    }

    fixed sum = FL2FX(0.0);
    for(i = 0; i < length; i++) {
        sum += fixed_exp(data[i] - m);
    }

    fixed offset = m + fixed_log(sum);
    for(i = 0; i < length; i++) {
        data[i] = fixed_exp(data[i] - offset);
    }
}


/*
 * relu activation function
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void relu_activation(fixed *data, uint32_t length){
    uint32_t i;

    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? 0 : data[i];
    }
}

void relu6_activation(fixed *data, uint32_t length) {
#define FIXED_SIX INT_TO_FIXED(6)
    for (uint32_t i = 0; i < length; i++) {
        if (data[i] < 0)
            data[i] = 0;
        else if (data[i] > FIXED_SIX)
            data[i] = FIXED_SIX;
    }
}

/*
 * leaky relu activation function
 * Parameters:
 *  alfa   => coeficient to multiply negative values
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void leakyrelu_activation(fixed *data, uint32_t length, fixed alpha){
    uint32_t i;

    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? FIXED_MUL(alpha, data[i]) : data[i];
    }
}


/*
 * tanh activation function: (2 / (1+e^(-2x)) -1
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void tanh_activation(fixed *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = fixed_tanh(data[i]);
    }
}

/*
 * sigmoid activation function: 1 / (1 + exp(-x))
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void sigmoid_activation(fixed *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = FIXED_DIV(FIX_ONE, FIX_ONE + fixed_exp(-data[i]));
    }
}

/*
 * softsign activation function: x / (|x| + 1)
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softsign_activation(fixed *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = FIXED_DIV(data[i],(fixed_abs(data[i])+FIX_ONE));
    }
}

/*
 * softplus activation function: log(e^x + 1)
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softplus_activation(fixed *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = fixed_log( fixed_exp(data[i])+1 );
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
    uint32_t c,i,j;
    uint32_t cantidad = 0;

    output->length = input.channels * input.height * input.width;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*output->length);

    for(i=0;i<input.height;i++){
        for(j=0;j<input.width;j++){
            for(c=0;c<input.channels;c++){
                output->data[cantidad++] = input.data[(c*input.width*input.height)+(i*input.width)+j];
            }
        }
    }
}




/*********************************************************************************************************************************/
/* Normalization layers */


void normalization1(normalization_layer_t n, data1d_t input, data1d_t * output){

    uint32_t i;

    output->length = input.length;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] =  FIXED_MUL((input.data[i]-n.sub_val[i]),n.inv_div_val[i]);
    }
}

void normalization2(normalization_layer_t n, data1d_t input, data1d_t * output){

    uint32_t i;

    output->length = input.length;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = FIXED_MUL(input.data[i],n.inv_div_val[i]);
    }
}

/*
 * batch_normalization{X}d_layer()
 * Keras Batch Normalization
 * Parameters:
 *      batch_normlization_t norm =>  structure with batch normalization layer parameters
 *      *data  =>   pointer to data{X}d_t
 */

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data) {
    uint32_t i;
	dfixed d_data;

	for(i = 0; i < data->length; i++) {
		d_data = DFIXED_MUL(data->data[i], layer.moving_inv_std_dev[i]) + layer.std_beta[i];

		if (d_data > DFIX_MAX)
			d_data = FIX_MAX;
		else if (d_data < DFIX_MIN)
			d_data = FIX_MIN;
		else 
			d_data = DFIXED_TO_FIXED(d_data);

		data->data[i] = d_data;
	}
}


void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data) {
    uint32_t i, j, ilen = 0;
    uint32_t length = data->height * data->width;
	dfixed d_data;

    for(i = 0; i < data->channels; i++, ilen += length) {
        for(j = 0; j < length; j++) {
            d_data = DFIXED_MUL(data->data[ilen+j], layer.moving_inv_std_dev[i]) + layer.std_beta[i];
			
			if (d_data > DFIX_MAX)
				d_data = FIX_MAX;
			else if (d_data < DFIX_MIN)
				d_data = FIX_MIN;
			else 
				d_data = DFIXED_TO_FIXED(d_data);

			data->data[ilen+j] = d_data;
		}
    }
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
    uint32_t c, i, j;

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
    }
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
    uint32_t c, i, j, output_index, input_index;

    // Calc output dimension
    output->channels = input.channels;
    output->width = input.width + 2 * pad_w;
    output->height = input.height + 2 * pad_h;

    size_t output_size = output->channels * output->width * output->height;
     output->data = (fixed *)swap_alloc(output_size * sizeof(fixed));

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

    zero_padding2d_init(pad_h, pad_w, output);
}

/* channel_adapt_layer()
 *  Converts Tensorflow/Keras Image (Height, Width, Channel) to Embedia format (Channel, Height, Width).
 *  Usually required for first convolutional layer
 * Parameters:
 *  input   => input data of type data3d_t.
 *  *output => pointer to the data3d_t structure where the result will be stored.
 */
void channel_adapt_layer(data3d_t input, data3d_t * output){

    uint32_t i, j, c, l;

    output->channels = input.channels;
    output->height   = input.height;
    output->width    = input.width;
    output->data     = (fixed*)swap_alloc( sizeof(fixed)*output->channels*output->height*output->width );

    for(c=0, l=0; c < input.channels; c++){
        for(i=0; i < input.height; i++) {
            for(j=0; j < input.width; j++, l++ ){
                output->data[l] = input.data[i*input.channels*input.width+input.channels*j+c];
            }
        }
    }
}
