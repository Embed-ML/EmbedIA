/**
 * Function: batch_normalization3d_layer
 * Lines: 633-652
 */

void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data) {
    uint32_t i, j, ilen = 0;
    uint32_t length = data->height * data->width;
	dfixed d_data;

    for(i = 0; i < data->channels; i++, ilen += length) {
        for(j = 0; j < length; j++) {
            d_data = DFIXED_MUL(data->data[ilen+j], layer.moving_inv_std_dev[i]) + layer.std_beta[i];
			
			if (d_data > DFIX_MAX)
				d_data = FIX_MAX;
			else if (d_data < DFIX_MIN)
				d_data = FIX_MIN;
			else 
				d_data = DFIXED_TO_FIXED(d_data);

			data->data[ilen+j] = d_data;
		}
    }
}
