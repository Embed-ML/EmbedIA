/**
 * Function: conv2d_layer
 * Lines: 158-187
 */

/* @embedia-note
 * FIXED8 IMPLEMENTATION NOTES:
 * - Uses DDFIXED_MAC for multiply-accumulate to prevent overflow during convolution
 * - Accumulates in 32-bit (dfixed) then saturates to 8-bit (fixed) at output
 * - FX2DFX converts bias from fixed to dfixed before adding to accumulated sum
 * - DFX2FX_RND_SAT performs rounding and saturation when converting back to fixed8
 * - Memory layout: filters are stored contiguously (channel-major order)
 */
void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output) {
    int32_t delta, i, j, k, l, f_pos, i_pos;
    int16_t f, c;
    int32_t value;

    // calculate output size and allocate memory
    calc_alloc_conv2d_output(layer.n_filters, layer.kernel, layer.strides, layer.padding, input, output);

    for(f=0; f<layer.n_filters; f++){
        delta = f*output->height*output->width;
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

                            DDFIXED_MAC(value, layer.filters[f].weights[f_pos], input.data[i_pos]);
                        }
                    }
                }
                value += FX2DFX(layer.filters[f].bias);
                output->data[delta + i*output->width + j] = DFX2FX_RND_SAT(value);
            }
        }
    }
}
