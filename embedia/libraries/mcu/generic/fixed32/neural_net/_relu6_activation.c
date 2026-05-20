/**
 * Function: relu6_activation
 * Lines: 748-757
 */

void relu6_activation(fixed *data, uint32_t length) {
#define FIXED_SIX INT_TO_FIXED(6)
    uint32_t i;
    for (i = 0; i < length; i++) {
        if (data[i] < FIX_ZERO)
            data[i] = FIX_ZERO;
        else if (data[i] > FIXED_SIX)
            data[i] = FIXED_SIX;
    }
}
