EMBEDIA_INLINE realx_t mean_val(const real_t *data, uint32_t length)
{
    if (length == 0u) {
        return REALX_ZERO;
    }

    return sum_val(data, length) / (real_t)length;
}