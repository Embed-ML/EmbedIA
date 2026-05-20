/**
 * Function: batch_normalization1d_layer
 * Lines: 610-626
 */

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data) {
    uint32_t i;
	dfixed d_data;

	for(i = 0; i < data->length; i++) {
		d_data = DFIXED_MUL(data->data[i], layer.moving_inv_std_dev[i]) + layer.std_beta[i];

		if (d_data > DFIX_MAX)
			d_data = FIX_MAX;
		else if (d_data < DFIX_MIN)
			d_data = FIX_MIN;
		else 
			d_data = DFIXED_TO_FIXED(d_data);

		data->data[i] = d_data;
	}
}
