/**
 * Function: sum_val
 * Lines: 398-424
 */

EMBEDIA_INLINE realx_t sum_val(const real_t *data, uint32_t length)
{
    if (length == 0u) {
        return REALX_ZERO;
    }

    realx_t sum0 = REALX_ZERO;
    realx_t sum1 = REALX_ZERO;

    const real_t *ptr = data;
    uint32_t remaining = length;

    for (; remaining >= 4u; remaining -= 4u){
        sum0 = DFIXED_ADD(sum0, FX2DFX(*ptr++));
        sum1 = DFIXED_ADD(sum1, FX2DFX(*ptr++));
        sum0 = DFIXED_ADD(sum0, FX2DFX(*ptr++));
        sum1 = DFIXED_ADD(sum1, FX2DFX(*ptr++));
    }

    sum0 = DFIXED_ADD(sum0, sum1);
    for (; remaining > 0u; remaining--)
    {
        sum0 = DFIXED_ADD(sum0, FX2DFX(*ptr++));
    }

    return sum0;
}
