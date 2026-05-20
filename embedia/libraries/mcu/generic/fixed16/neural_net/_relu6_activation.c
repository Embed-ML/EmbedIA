/**
 * Function: relu6_activation
 * Lines: 496-504
 */

void relu6_activation(fixed *data, uint32_t length) {
#define FIXED_SIX INT_TO_FIXED(6)
    for (uint32_t i = 0; i < length; i++) {
        if (data[i] < 0)
            data[i] = 0;
        else if (data[i] > FIXED_SIX)
            data[i] = FIXED_SIX;
    }
}
