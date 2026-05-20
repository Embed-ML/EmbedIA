void conv2d_padding_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output) {
    int32_t delta, i, j, k, l, f_pos, i_pos;
    int16_t f, c, i_pad, j_pad, pad_h, pad_w;
    dfixed acc;

    // calculate output size and allocate memory
    calc_alloc_conv2d_output(layer.n_filters, layer.kernel, layer.strides, layer.padding, input, output);

    pad_h = compute_padding(layer.strides.h, input.height, layer.kernel.h, output->height);
    pad_w = compute_padding(layer.strides.w, input.width,  layer.kernel.w, output->width);

    const uint16_t scale_q = layer.qparam.scale_q;
    const int8_t zero_point = layer.qparam.zero_point;

    for(f=0; f<layer.n_filters; f++){
        delta = f*(output->height)*(output->width);

        for(i=0; i<output->height; i++){
            for(j=0; j<output->width; j++){
                acc = 0;
                for(c=0; c<layer.channels; c++){
                    for(k=0; k<layer.kernel.h; k++){
                        for(l=0; l<layer.kernel.w; l++){
                            i_pad = i * layer.strides.h + k - pad_h;
                            j_pad = j * layer.strides.w + l - pad_w;
                            // Check for valid input access within padded bounds
                            if (i_pad >= 0 && i_pad < input.height && j_pad >= 0 && j_pad < input.width) {
                                f_pos = (c * layer.kernel.h * layer.kernel.w) + k * layer.kernel.w + l;
                                i_pos = (c * input.height * input.width) + i_pad * input.width + j_pad;

                                int8_t weight_q = layer.filters[f].weights[f_pos];
                                // fixed16 × int8 → int32 (acumulación exacta)
                                acc += (int32_t)input.data[i_pos] * (int16_t)(weight_q - zero_point);
                            }
                        }
                    }
                }
                // Descuantizar una vez + bias
                dfixed result = (dfixed)((acc * (int32_t)scale_q + QUANT_SCALE_HALF) >> SCALE_TO_FX_SHIFT);
                result += FX2DFX(layer.filters[f].bias);

                output->data[delta + i*output->width + j] = DFX2FX_SAT(result);
		    }
		}
	}
}
