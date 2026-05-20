/**
 * Function: softmax_activation
 * Lines: 715-736
 */

void softmax_activation(float *data, uint32_t length){
    uint32_t i;
    float m = -INFINITY;

    // Find max for numerical stability
    for (i = 0; i < length; i++) {
        if (data[i] > m) m = data[i];
    }

    // Compute sum of exponentials
    float sum = (0.0);
    for(i = 0; i < length; i++) {
        sum += exp(data[i] - m);
    }

    // Normalize

    float offset = m + log(sum);
    for(i = 0; i < length; i++) {
        data[i] = exp(data[i] - offset);
    }
}
