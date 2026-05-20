/*void softplus_activation_old(fixed *data, uint32_t length){
    uint32_t i;

    for(i=0;i<length;i++){
        data[i] = fixed_log( fixed_exp(data[i])+1 );
    }
}*/
void softplus_activation(fixed *data, uint32_t length) {

    /* log(2) en Q16.16 */
    static const fixed FIX_LOG2 = FL2FX_CONST(0.69314718);

    for (uint32_t i = 0; i < length; i++) {
        fixed x = data[i];

        /* Zona lineal: softplus(x) ≈ x para x > umbral
         * softplus(8) = 8.000335, error < 0.004%        */
        if (x >= FL2FX_CONST(8.0)) {
            data[i] = x;
            continue;
        }

        /* Zona cero: softplus(x) ≈ 0 para x < -umbral
         * softplus(-8) = 0.000335, despreciable          */
        if (x <= FL2FX_CONST(-8.0)) {
            data[i] = 0;
            continue;
        }

        /* Zona de transición [-8, 8]:
         * softplus(x) = log(1 + exp(x))
         *             = log(exp(x) * (exp(-x) + 1))
         *             = x + log(1 + exp(-x))
         *
         * Para x >= 0: usar x + log(1 + exp(-x))
         *   exp(-x) ∈ (0, 1] → log(1 + exp(-x)) ∈ (0, log2]
         *   evita exp grande
         *
         * Para x < 0:  usar log(1 + exp(x)) directamente
         *   exp(x) ∈ (0, 1] → misma ventaja               */
        if (x >= 0) {
            /* x + log(1 + exp(-x)) */
            fixed e = fixed_exp(-x);               /* exp(-x) ∈ (0,1] */
            data[i] = x + fixed_log(e + FIX_ONE);
        } else {
            /* log(1 + exp(x)) */
            fixed e = fixed_exp(x);                /* exp(x) ∈ (0,1)  */
            data[i] = fixed_log(e + FIX_ONE);
        }
    }
}