void softsign_activation(fixed *data, uint32_t length) {
    uint32_t i;
    for (i = 0; i < length; i++) {
        const fixed x     = data[i];
        data[i] = FIXED_DIV(x, FIXED_ABS(x) + FIX_ONE);
    }

}