/* @embedia-note
 * POINTWISE CONVOLUTION (1x1 SEPARABLE SECOND STAGE):
 * - Combines channels from depthwise output using 1x1 kernels
 * - No spatial filtering: only mixes channels at each (i,j) position
 * - Delta parameter: offset for writing multiple filters to same output buffer
 * - Bias pre-converted to dfixed (FX2DFX) for efficient addition
 * - Critical optimization: inner loop over channels uses MAC instruction
 */
static void pointwise(separable_conv2d_layer_t layer, filter_t filter, data3d_t input, data3d_t *output, uint32_t delta)
{
    uint32_t i,j,c;

    uint32_t in_w = input.width;
    uint32_t in_h = input.height;

    uint32_t out_w = output->width;
    uint32_t out_h = output->height;

    uint32_t bias = FX2DFX(filter.bias);

    for(i=0;i<out_h;i++)
    {
        for(j=0;j<out_w;j++)
        {
            int32_t sum = 0;

            for(c=0;c<layer.point_channels;c++)
            {
                fixed *in_ptr = &input.data[c*in_h*in_w +  i*in_w + j];
                DDFIXED_MAC(sum, filter.weights[c], *in_ptr);
            }

            sum += bias;

            output->data[delta + i*out_w + j] = DFX2FX_RND_SAT(sum);
        }
    }
}