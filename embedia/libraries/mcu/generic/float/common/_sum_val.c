EMBEDIA_INLINE realx_t sum_val(const real_t *data, uint32_t length)
{
    if (length == 0u) {
        return REAL_ZERO;
    }

    realx_t sum = REALX_ZERO;

    const real_t *ptr = data;
    uint32_t remaining = length;

    for (; remaining >= 4u; remaining -= 4u)
    {
        sum += *ptr++; sum += *ptr++;
        sum += *ptr++; sum += *ptr++;
    }

    for (; remaining > 0u; remaining--)
    {
        sum += *ptr++;
    }

    return sum;
}