import numpy as np


class TypeConverter:
    _size = 0
    _name = ''

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    def fit(self, values):
        pass

    def transform(self, values):
        return values

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inv_transform(self, values):
        return values


class FloatConverter(TypeConverter):
    _name = 'Float32'


class FixedTypeConverter(TypeConverter):
    int_bits = 0
    frac_bits = 0
    dtype = np.uint32

    def __init__(self, int_bits, frac_bits):
        self.set_bits(int_bits, frac_bits)
        self._name = 'Fixed%d.%d' % (int_bits, frac_bits)

    def set_bits(self, int_bits, frac_bits):
        if int_bits + frac_bits not in [8, 16, 32]:
            raise ValueError("int_bits + frac_bits must be 8, 16 or 32")
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self._size = (int_bits + frac_bits + 7) // 8
        if self._size in range(1, 5):
            self.dtype = [np.int8, np.int16, np.int32, np.int32][self._size - 1]

    def transform(self, values):
        values = np.array(values, float)
        new_values = np.round(values * (2 ** self.frac_bits)).astype(self.dtype)
        return new_values  # & (2**(self.int_bits+self.frac_bits)-1)

    def inv_transform(self, values):
        values = np.array(values, self.dtype)
        new_values = values / (2 ** self.frac_bits)
        return new_values.astype(float)


class Float16TypeConverter(TypeConverter):
    _size = 2  # Size in bytes flor 16 bits floating point
    _name = 'Float16'

    def transform(self, values):
        # Convert float 32 to float 16 IEEE floating point
        float16_values = np.float16(values).view(np.uint16)
        return float16_values

    def inv_transform(self, values):
        # Convert float 16 to float 32 IEEE floating point
        float_values = values.view(np.float16)
        return float_values

class BFloat16TypeConverter(TypeConverter):
    _size = 2  # bfloat16 size
    _name = 'BFloat16'

    def transform(self, values):
        float_32_bits = np.float32(values).view(np.uint32)
        return float_32_bits >> 16

    def inv_transform(self, values):
        values = (values.astype(np.uint32) << 16)
        return values.view(np.float32)


class QuantizedTypeConverter(TypeConverter):
    _size = 0
    min_val = 0
    max_val = 0
    max_qint = 0
    scale = 1.0
    zero_pt = 0
    dtype = np.uint16
    symetric = True
    lower_percentile = 1
    upper_percentile = 99

    def __init__(self, bits, symetric=True):
        self.set_bits(bits)
        self._name = 'Quant%d%c' % (bits, ['A', 'S'][symetric])
        self.symetric = symetric

    def set_bits(self, bits):
        if bits == 8:
            self.dtype = np.uint8
        elif bits == 16:
            self.dtype = np.uint16
        else:
            raise ValueError("bits must be 8, 16")
        self.bits = bits
        self._size = bits / 8
        self.max_qint = 2 ** self.bits - 1

    # def _remove_outliers(self, data):
    #     Q1 = np.percentile(data, 25)
    #     Q3 = np.percentile(data, 75)
    #     IQR = Q3 - Q1
    #     lower_limit = Q1 - 1.5 * IQR
    #     upper_limit = Q3 + 1.5 * IQR
    #     return data[(data >= lower_limit) & (data <= upper_limit)]

    def fit(self, values):
        if self.symetric:
            self.max_val = np.max(np.abs(values))
            self.min_val = -self.max_val
        else:
            self.min_val = np.min(values)
            self.max_val = np.max(values)

        self.scale = (self.max_val - self.min_val) / self.max_qint
        if self.scale == 0:
            self.scale = 1
        self.zero_pt = -self.min_val / self.scale
        if self.zero_pt < 0:
            self.zero_pt = 0
        elif self.zero_pt > self.max_qint:
            self.zero_pt = self.max_qint
        else:
            self.zero_pt = round(self.zero_pt)

    # def fit_b(self, values):
    #     #15/17
    #     if self.symetric:
    #         self.max_val = np.percentile(np.abs(values), self.upper_percentile)
    #         self.min_val = -self.max_val
    #     else:
    #         self.min_val = np.percentile(np.min(values), self.upper_percentile)
    #         self.max_val = np.percentile(np.max(values), self.lower_percentile)
    #     self.scale = (self.max_val - self.min_val) / self.max_qint
    #     if self.scale == 0:
    #         self.scale = 1
    #     self.zero_pt= -self.min_val / self.scale
    #     if self.zero_pt < 0:
    #         self.zero_pt = 0
    #     elif self.zero_pt > self.max_qint:
    #         self.zero_pt = self.max_qint
    #     else:
    #         self.zero_pt = round(self.zero_pt)
    #
    # def fit_c(self, values):
    #     #15/17
    #     values = self._remove_outliers(values)
    #     if self.symetric:
    #         self.max_val = np.percentile(np.abs(values), self.upper_percentile)
    #         self.min_val = -self.max_val
    #     else:
    #         self.min_val = np.percentile(np.min(values), self.upper_percentile)
    #         self.max_val = np.percentile(np.max(values), self.lower_percentile)
    #     self.scale = (self.max_val - self.min_val) / self.max_qint
    #     if self.scale == 0:
    #         self.scale = 1
    #     self.zero_pt= -self.min_val / self.scale
    #     if self.zero_pt < 0:
    #         self.zero_pt = 0
    #     elif self.zero_pt > self.max_qint:
    #         self.zero_pt = self.max_qint
    #     else:
    #         self.zero_pt = round(self.zero_pt)

    def transform(self, values):
        # f = scale * (q - zero_point)
        # q = f / scale + zero_pt
        if isinstance(values, list):
            values = np.array(values)
        scaled_values = values/self.scale + self.zero_pt

        quantized_values = np.round(scaled_values).astype(self.dtype)
        return quantized_values

    def inv_transform(self, quant_values):
        # f = scale * (q - zero_point)
        if isinstance(quant_values, list):
            quant_values = np.array(quant_values)
        values = quant_values.astype(np.float32)
        # Desnormalizar los valores al rango original
        descaled_values = self.scale * (values - self.zero_pt)
        return descaled_values


class TypeConverterManager:
    type_converters = list()
    def __init__(self):
        pass

    def add_type_converter(self, type_converter):
        self.type_converters.append(type_converter)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.type_converters):
            converter = self.type_converters[self._index]
            self._index += 1
            return converter
        raise StopIteration

    def test(self, converter, values):
        new_values = converter.inv_transform(converter.fit_transform(values))
        diff_values = values - new_values
        ae = np.absolute(diff_values)
        se = np.square(diff_values)

        return (np.mean(ae), np.std(ae), np.mean(se), np.std(se))


if __name__ == "__main__":
    conv_man = TypeConverterManager()

    conv_man.add_type_converter(FloatConverter())
    conv_man.add_type_converter(FixedTypeConverter(17, 15))
    conv_man.add_type_converter(FixedTypeConverter(9, 7))
    conv_man.add_type_converter(QuantizedTypeConverter(16))
    conv_man.add_type_converter(Float16TypeConverter())
    conv_man.add_type_converter(BFloat16TypeConverter())
    conv_man.add_type_converter(QuantizedTypeConverter(8, False))
    conv_man.add_type_converter(QuantizedTypeConverter(8, True))
    for i in range(0, 3):
        conv_man.add_type_converter(FixedTypeConverter(4+i, 4-i))

    values = np.array([8.5, 16.33, 255.56, -8.5, -16.33, -300.56])
    print("Original floating point values:", values)

    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Converter", "MAE", "MAE Dev", "MSE", "MSE Dev"]
    table.align["MAE"] = "r"
    table.align["MAE Dev"] = "r"
    table.align["MSE"] = "r"
    table.align["MSE Dev"] = "r"

    for converter in conv_man:
        (mae, smae, mse, smse) = conv_man.test(converter, values)
        table.add_row([f"{converter.name}",
                       f"{mae:10.4f}", f"{smae:.4f}",
                       f"{mse:10.4f}", f"{smse:.4f}"])

    print(table)



    # def print_hex(values):
    #     hex_strings = [hex(i) for i in values]
    #     print(', '.join(hex_strings))
    #
    # conv = Quantized8TypeConverter()
    # fv = conv.transform(values)
    # iv = conv.inv_transform(fv)
    # print_hex(fv)
    # print(iv)
