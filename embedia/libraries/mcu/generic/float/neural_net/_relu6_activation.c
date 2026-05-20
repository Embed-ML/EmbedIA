/**
 * Function: relu6_activation
 * Lines: 756-765
 */

void relu6_activation(float *data, uint32_t length) {

    uint32_t i;
    for (i = 0; i < length; i++) {
        if (data[i] < 0.0)
            data[i] = 0.0;
        else if (data[i] > 6.0)
            data[i] = 6.0;
    }
}
