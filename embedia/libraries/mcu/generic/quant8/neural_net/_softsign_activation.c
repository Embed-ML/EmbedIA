void softsign_activation(fixed *data, uint32_t length) {

    static const fixed seed_table[9] = {
        FL2FX_CONST(0.70),
        FL2FX_CONST(0.35),
        FL2FX_CONST(0.175),
        FL2FX_CONST(0.0875),
        FL2FX_CONST(0.04375),
        FL2FX_CONST(0.021875),
        FL2FX_CONST(0.0109375),
        FL2FX_CONST(0.00546875),
        FL2FX_CONST(0.002734375)
    };

    for (uint32_t i = 0; i < length; i++) {
        fixed x     = data[i];
        fixed abs_x = FIXED_ABS(x);

        if (abs_x >= FL2FX(127.0)) {
            data[i] = x > 0 ? FIX_ONE : -FIX_ONE;
            continue;
        }

        fixed den = abs_x + FIX_ONE;

        unsigned int int_part = (unsigned int)FIXED_TO_INT(den);
        unsigned int k = 0;
        while ((int_part >> (k + 1)) > 0) k++;

        fixed r = seed_table[k];
        r = FIXED_MUL(r, (FIX_ONE << 1) - FIXED_MUL(den, r));
        r = FIXED_MUL(r, (FIX_ONE << 1) - FIXED_MUL(den, r));
        //r = FIXED_MUL(r, (FIX_ONE << 1) - FIXED_MUL(den, r));
        // fixed16 solo necesita 3 iteraciones
        //r = FIXED_MUL(r, (FIX_ONE << 1) - FIXED_MUL(den, r));
        data[i] = FIXED_MUL(x, r);
    }
}