/*void softsign_activation_old(fixed *data, uint32_t length){
    uint32_t i;
    for(i=0;i<length;i++){
        data[i] = FIXED_DIV(data[i],(fixed_abs(data[i])+FIX_ONE));
    }
}*/
void softsign_activation(fixed *data, uint32_t length) {

    static const fixed seed_table[9] = {
        FL2FX_CONST(0.75000), /* k=0: den ∈ [1,   2) */
        FL2FX_CONST(0.37500), /* k=1: den ∈ [2,   4) */
        FL2FX_CONST(0.18750), /* k=2: den ∈ [4,   8) */
        FL2FX_CONST(0.09375), /* k=3: den ∈ [8,  16) */
        FL2FX_CONST(0.04688), /* k=4: den ∈ [16,  32) */
        FL2FX_CONST(0.02344), /* k=5: den ∈ [32,  64) */
        FL2FX_CONST(0.01172), /* k=6: den ∈ [64, 128) */
        FL2FX_CONST(0.00586), /* k=7: den ∈ [128,256) */
        FL2FX_CONST(0.00293), /* k=8: den ∈ [256,512) */
    };

    for (uint32_t i = 0; i < length; i++) {
        fixed x     = data[i];
        fixed abs_x = FIXED_ABS(x);

        if (abs_x >= FL2FX_CONST(127.0)) {
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
        r = FIXED_MUL(r, (FIX_ONE << 1) - FIXED_MUL(den, r));
        r = FIXED_MUL(r, (FIX_ONE << 1) - FIXED_MUL(den, r));

        data[i] = FIXED_MUL(x, r);
    }
}
