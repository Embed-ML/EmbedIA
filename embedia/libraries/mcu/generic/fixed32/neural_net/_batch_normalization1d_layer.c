/**
 * Function: batch_normalization1d_layer
 * Lines: 862-867
 */

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data) {
    uint32_t i;
    for (i = 0; i < data->length; i++) {
        data->data[i] = FIXED_MUL(data->data[i], layer.moving_inv_std_dev[i]) + layer.std_beta[i];
    }
}
