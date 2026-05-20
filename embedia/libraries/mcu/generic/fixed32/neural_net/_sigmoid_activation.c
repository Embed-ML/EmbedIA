void sigmoid_activation(fixed *data, uint32_t length) {
    for (uint32_t i = 0; i < length; i++) {
        /* x/2 = >> 1, sin FIXED_MUL */
        fixed th = fixed_tanh(data[i] >> 1);
        /* (tanh + 1) / 2 = (th + FIX_ONE) >> 1, sin FIXED_DIV */
        data[i] = (th + FIX_ONE) >> 1;
    }
}
