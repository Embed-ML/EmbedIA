/*
 * EmbedIA
 * C LIBRARY FOR THE IMPLEMENTATION OF NEURAL NETWORKS ON MICROCONTROLLERS
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
    output->data = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );
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

                            value += layer.filters[f].weights[f_pos] * input.data[i_pos];
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
                                value += layer.filters[f].weights[f_pos] * input.data[i_pos];
                            }
                        }
                    }
                }
                output->data[delta + i*output->width + j] = value + layer.filters[f].bias;
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

                            value += layer.filters[f].weights[f_pos] * input.data[i_pos];
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
    float sum;

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
                                sum += filter.weights[f_pos] * input.data[i_pos];
                            }
                    }
                }
                output->data[c * output->width * output->height + i * output->width + j] = sum;
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
    float sum;

    for (i = 0; i < output->height; i++) {
        for (j = 0; j < output->width; j++) {
            sum = 0;
            for (c = 0; c < layer.point_channels; c++) {
                i_pos = (c * input.height * input.width) + (i * 1) * input.width + (j * 1);
                sum += (filter.weights[c] * input.data[i_pos]);
            }
            output->data[delta + i * output->width + j] = sum + filter.bias;
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
    output->data     = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );

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
                                sum += layer.weights[f_pos] * input.data[i_pos];
                            }


                    }
                }
                output->data[c * output->width * output->height + i * output->width + j] = sum + layer.bias[c];
            }
        }
    }
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
 * neuron_forward()
 *  Function that performs the forward of a neuron in front of a given set of input data.
 * Parameters:
 *  neuron_t neuron => neuron with its weights and bias loaded.
 *  flatten_data_t input => input data in vector form (flatten_data_t).
 * Returns:
 *  float => result of the operation
 */

static float neuron_forward(neuron_t neuron, data1d_t input){
    uint32_t i;
    float result = 0;

    for(i=0;i<input.length;i++){
        result += input.data[i]*neuron.weights[i];
    }

    return result + neuron.bias;
}

/*
 * dense_layer()
 *  Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *  dense_layer => structure with the weights of the neurons of the dense layer.
 *  input       => structure data1d_t with the input data to process.
 *  *output     => structure data1d_t to store the output result.
 */
void dense_layer(dense_layer_t dense_layer, data1d_t input, data1d_t * output){
    uint32_t i;

    output->length = dense_layer.n_neurons;
    output->data = (float*)swap_alloc(sizeof(float)*dense_layer.n_neurons);

    for(i=0;i<dense_layer.n_neurons;i++){
        output->data[i] = neuron_forward(dense_layer.neurons[i],input);
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
    }
}


/*
 * avg_pooling_2d()
 *  Function that applies an average pooling to an input with a window size of received
 *  by parameter (uint16_t strides)
 * Parameters:
 *  input => input data of type data3d_t.
 *  *output => pointer to the data3d_t structure where the result will be stored.
 */

void avg_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
    uint32_t c,i,j,aux1,aux2;
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
    }
}

/*
 * softmax activation function
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softmax_activation(float *data, uint32_t length){
    uint32_t i;
    float m = -INFINITY;
    for(i = 0; i < length; i++) {
        if (data[i] > m) {
            m = data[i];
        }
    }

    float sum = (0.0);
    for(i = 0; i < length; i++) {
        sum += exp(data[i] - m);
    }

    float offset = m + log(sum);
    for(i = 0; i < length; i++) {
        data[i] = exp(data[i] - offset);
    }
}


/*
 * relu activation function
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void relu_activation(float *data, uint32_t length){
    uint32_t i;

    for(i=0;i<(length);i++){
        data[i] = data[i] < 0 ? 0 : data[i];
    }
}

/*
 * leaky relu activation function
 * Parameters:
 *  alfa   => coeficient to multiply negative values
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
void sigmoid_activation(float *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = 1 / (1 + exp(-data[i]));
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
    uint32_t c,i,j;
    uint32_t cantidad = 0;

    output->length = input.channels * input.height * input.width;
    output->data = (float*)swap_alloc(sizeof(float)*output->length);

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
    output->data = (float*)swap_alloc(sizeof(float)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = (input.data[i]-n.sub_val[i])*n.inv_div_val[i];
    }
}

void normalization2(normalization_layer_t n, data1d_t input, data1d_t * output){

    uint32_t i;

    output->length = input.length;
    output->data = (float*)swap_alloc(sizeof(float)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = input.data[i]*n.inv_div_val[i];
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

    for(i = 0; i < data->length; i++) {
        data->data[i] = data->data[i] * layer.moving_inv_std_dev[i] + layer.std_beta[i];
    }
}


void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data) {
    uint32_t i, j, ilen = 0;
    uint32_t length = data->height * data->width;

    for(i = 0; i < data->channels; i++, ilen += length) {
        for(j = 0; j < length; j++) {
            data->data[ilen+j] = data->data[ilen+j] * layer.moving_inv_std_dev[i] + layer.std_beta[i];
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
    output->data     = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );

    for(c=0, l=0; c < input.channels; c++){
        for(i=0; i < input.height; i++) {
            for(j=0; j < input.width; j++, l++ ){
                output->data[l] = input.data[i*input.channels*input.width+input.channels*j+c];
            }
        }
    }
}




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
        twiddle_re = cos(angle);
        twiddle_im = sin(angle);
    }
  }
}

/*
 * void create_spectrogram(spectrogram_layer_t config, data1d_t input, data3d_t *output)
 * Generates the spectrogram from the input signal by applying FFTs
 * and further processing.
 * Parameters:
 *   - config: Spectrogram layer configuration
 *   - input:  1D input signal  
 *   - output: 3D output spectrogram (W = n_mels, H = b_blocks, Ch = 1)
 */
void create_spectrogram(spectrogram_layer_t config, data1d_t input, data3d_t * output){
    register int i,j,k;
    float aux;
    
    float data_re[config.n_fft];
    float data_im[config.n_fft];

    output->width    = config.n_mels;
    output->height   = config.n_blocks;
    output->channels = 1;
    output->data     = (float*)swap_alloc( sizeof(float)*output->channels*output->height*output->width );
    
    for(i=0;i<config.n_blocks;i++){
        // Copy the values ​​to the input of the fft
        const unsigned int start = i*config.step;
        for(j=0;j<config.n_fft;j++){
            data_re[j] = input.data[start+j];
            data_im[j] = 0;
        }

        // Calculate fft
        fft(data_re,data_im,config.n_fft);

        // Get the module of the fft
        for(j=0;j<config.n_fft;j++){
            const float aux_re = data_re[j];
            const float aux_im = data_im[j];
            data_re[j] = sqrt(aux_re*aux_re + aux_im*aux_im);
        }

        // N_MELS processing
        const unsigned int start2 = i*config.n_mels;
        for(j=0;j<config.n_mels;j++){
            const unsigned int start3 = j*config.len_nfft_nmels;
            aux = 0;
            for(k=0;k<config.len_nfft_nmels;k++){
                aux += data_re[start3+k];
            }
            aux /= config.len_nfft_nmels;
            if(config.convert_to_db){
                output->data[start2+j] = 10*log10(aux);
            }else{
                output->data[start2+j] = aux;
            }
        }
    }
}
