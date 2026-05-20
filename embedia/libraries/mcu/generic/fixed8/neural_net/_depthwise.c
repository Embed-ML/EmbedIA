/* @embedia-note
 * DEPTHWISE CONVOLUTION (SEPARABLE FIRST STAGE):
 * - Applies spatial filtering per channel without mixing channels
 * - Padding calculation: handles both SAME and VALID modes
 * - Stride support: allows downsampling during convolution
 * - Uses DDFIXED_MAC for accumulation (32-bit intermediate prevents overflow)
 * - DFX2FX_RND_SAT converts accumulated sum back to fixed8 with rounding and saturation
 * - Memory layout optimized: base pointers computed once per channel
 */
static void depthwise(separable_conv2d_layer_t layer, data3d_t input, data3d_t *output) {
    uint32_t c,i,j,k,l;
    int32_t pad_h, pad_w;
    int32_t i_pad,j_pad;

    uint32_t kernel_h = layer.depth_kernel_sz.h;
    uint32_t kernel_w = layer.depth_kernel_sz.w;

    uint32_t in_w = input.width;
    uint32_t in_h = input.height;

    uint32_t out_w = output->width;
    uint32_t out_h = output->height;

    uint32_t stride_h = layer.strides.h;
    uint32_t stride_w = layer.strides.w;

    if(layer.padding == PAD_SAME) {
        int32_t pad_total_h =
            (out_h - 1) * stride_h + kernel_h - in_h;

        int32_t pad_total_w =
            (out_w - 1) * stride_w + kernel_w - in_w;

        pad_h = (pad_total_h > 0) ? pad_total_h/2 : 0;
        pad_w = (pad_total_w > 0) ? pad_total_w/2 : 0;
    }
    else {
        pad_h = 0;
        pad_w = 0;
    }

    for(c=0;c<input.channels;c++)
    {
        const fixed *in_base = &input.data[c * in_h * in_w];
        const fixed *w_base  = &layer.depth_weights[c * kernel_h * kernel_w];

        for(i=0;i<out_h;i++) {
            for(j=0;j<out_w;j++) {
                int32_t sum = 0;

                for(k=0;k<kernel_h;k++) {
                    for(l=0;l<kernel_w;l++) {
                        i_pad = (int32_t)(i*stride_h + k - pad_h);
                        j_pad = (int32_t)(j*stride_w + l - pad_w);

                        if(i_pad>=0 && i_pad<in_h && j_pad>=0 && j_pad<in_w) {
                            const fixed *in_ptr = in_base + i_pad*in_w + j_pad;
                            const fixed *w_ptr  = w_base + k*kernel_w + l;

                            DDFIXED_MAC(sum,*w_ptr,*in_ptr);
                        }
                    }
                }

                sum += FX2DFX(layer.depth_bias[c]);

                uint32_t out_idx = c*out_h*out_w + i*out_w + j;
                output->data[out_idx] = DFX2FX_RND_SAT(sum);
            }
        }
    }
}