/*
void softmax_activation_old(fixed *data, uint32_t length){
    uint32_t i;
    fixed m = FIX_MIN;

    // Find max for numerical stability
    for (i = 0; i < length; i++) {
        if (data[i] > m) m = data[i];
    }

    // Compute sum of exponentials
    fixed sum = FIX_ZERO;
    for (i = 0; i < length; i++) {
        sum += fixed_exp(data[i] - m);
    }

    // Normalize
    fixed offset = m + fixed_log(sum);
    for (i = 0; i < length; i++) {
        data[i] = fixed_exp(data[i] - offset);
    }
}
*/

void softmax_activation(fixed *data, uint32_t length) {
    fixed m = FIX_MIN;
    for (uint32_t i = 0; i < length; i++)
        if (data[i] > m) m = data[i];

    fixed sum = 0;
    for (uint32_t i = 0; i < length; i++) {
        data[i] = fixed_exp(data[i] - m);
        sum += data[i];
    }

    if (sum <= 0) sum = FIX_ONE;

    /* Recíproco via Newton-Raphson, 2 iteraciones, sin FIXED_DIV */
    int n = 0;
    fixed xn = sum;
    while (xn >= FIX_ONE) { xn >>= 1; n++; }
    while (xn <  FIX_HALF) { xn <<= 1; n--; }

    fixed r = FL2FX_CONST(2.9142f) - (xn << 1);
    r = FIXED_MUL(r, (FIX_ONE << 1) - FIXED_MUL(xn, r));
    r = FIXED_MUL(r, (FIX_ONE << 1) - FIXED_MUL(xn, r));
    fixed inv_sum = (n >= 0) ? r >> n : r << (-n);

    for (uint32_t i = 0; i < length; i++)
        data[i] = FIXED_MUL(data[i], inv_sum);
}
