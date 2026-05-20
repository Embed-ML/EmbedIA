/**
 * Function: batch_normalization3d_layer
 * Lines: 882-891
 */

void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data) {
    uint32_t c, j, base_idx;
    uint32_t plane_size = data->height * data->width;
    for (c = 0; c < data->channels; c++) {
        base_idx = c * plane_size;
        for (j = 0; j < plane_size; j++) {
            data->data[base_idx + j] = data->data[base_idx + j] * layer.moving_inv_std_dev[c] + layer.std_beta[c];
        }
    }
}
