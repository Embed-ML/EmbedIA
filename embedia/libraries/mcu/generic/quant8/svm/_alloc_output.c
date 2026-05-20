/**
 * Function: alloc_output
 * Lines: 151-159
 */

static inline compute_t *alloc_output(data1d_t *output, uint16_t n_classes)
{
    compute_t *buf = (compute_t *)swap_alloc(n_classes * sizeof(compute_t));
    output->data   = (compute_t*)buf;
    output->length = n_classes;
    for (uint16_t i = 0; i < n_classes; i++)
        buf[i] = FIX_ZERO;
    return buf;
}
