/**
 * Function: softmax_activation
 * Lines: 456-476
 */

void softmax_activation(fixed *data, uint32_t length){
    uint32_t i;
    fixed m = -FIX_MAX;

    // Find max for numerical stability
    for (i = 0; i < length; i++) {
        if (data[i] > m) m = data[i];
    }

    // Compute sum of exponentials
    fixed sum = FL2FX_CONST(0.0);
    for (i = 0; i < length; i++) {
        sum += fixed_exp(data[i] - m);
    }

    // Normalize
    fixed offset = m + fixed_log(sum);
    for (i = 0; i < length; i++) {
        data[i] = fixed_exp(data[i] - offset);
    }
}
