import numpy as np
from dataclasses import dataclass

# ============================================================
# Base Type Converter
# ============================================================
class TypeConverter:

    def __init__(self):
        # Propiedades generales (por instancia)
        self._size = 0
        self._name = ""

        # Contador general de saturaciones (clipping)
        self.saturation_count = 0


    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    def fit(self, values):
        pass

    def transform(self, values):
        # Conversor base: no transforma
        self.saturation_count = 0
        return values

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inv_transform(self, values):
        return values

    def extended_transform(self, values):
        return fit_transform(values)

    @property
    def is_quantized(self):
        return isinstance(self, QuantizedTypeConverter)

    @property
    def is_fixed_point(self):
        return isinstance(self, FixedTypeConverter)

    @property
    def is_float(self):
        return isinstance(self, FloatConverter)

    @property
    def extended_type(self):
        return "Unknown"

    def export_params(self, mode='q15'):
        raise NotImplementedError("This converter does not support quantization params export")


# ============================================================
# Float32 Converter
# ============================================================
class FloatConverter(TypeConverter):

    def __init__(self):
        super().__init__()
        self._name = "Float32"
        self._size = 4

    @property
    def extended_type(self):
        return "double"


# ============================================================
# Signed Fixed Point Converter
# ============================================================
class FixedTypeConverter(TypeConverter):

    def __init__(self, int_bits, frac_bits):
        super().__init__()

        self.int_bits = 0
        self.frac_bits = 0
        self.dtype = np.int32

        self.set_bits(int_bits, frac_bits)
        self._name = "Fixed%d.%d" % (int_bits, frac_bits)

    def set_bits(self, int_bits, frac_bits):

        total_bits = int_bits + frac_bits
        if total_bits not in [8, 16, 32]:
            raise ValueError("int_bits + frac_bits must be 8, 16 or 32")

        self.int_bits = int_bits
        self.frac_bits = frac_bits

        # Tamaño en bytes
        self._size = (total_bits + 7) // 8

        # Tipo entero subyacente
        if self._size == 1:
            self.dtype = np.int8
        elif self._size == 2:
            self.dtype = np.int16
        else:
            self.dtype = np.int32

    def transform(self, values):

        values = np.array(values, dtype=np.float32)

        # Escalado a punto fijo
        scaled = np.round(values * (2 ** self.frac_bits))

        # Rango representable según bits totales
        total_bits = self.int_bits + self.frac_bits
        qmin = -(2 ** (total_bits - 1))
        qmax = (2 ** (total_bits - 1)) - 1

        # Saturación
        clipped = np.clip(scaled, qmin, qmax)

        # Contar saturaciones
        self.saturation_count = int(np.sum(scaled != clipped))

        return clipped.astype(self.dtype)

    def extended_transform(self, values):

        values = np.array(values, dtype=np.float32)

        # Escalado a punto fijo
        scaled = np.round(values * (2 ** (2*self.frac_bits)))

        # Rango representable según bits totales
        total_bits = 2*self.int_bits + 2*self.frac_bits
        qmin = -(2 ** (total_bits - 1))
        qmax = (2 ** (total_bits - 1)) - 1

        # Saturación
        clipped = np.clip(scaled, qmin, qmax)

        # Contar saturaciones
        self.saturation_count = int(np.sum(scaled != clipped))

        return clipped.astype(self.dtype)

    def inv_transform(self, values):

        values = np.array(values, dtype=self.dtype)
        new_values = values / (2 ** self.frac_bits)

        return new_values.astype(float)

    @property
    def extended_type(self):
        return "dfixed"


# ============================================================
# Unsigned Fixed Point Converter
# ============================================================
class UnsignedFixedTypeConverter(FixedTypeConverter):
    """
    Variante unsigned para datos en [0.0, 1.0] como ventanas de Hann.
    Ejemplo: UnsignedFixedTypeConverter(0, 8) → Q0.8, uint8, rango [0, 255]
    """

    def __init__(self, int_bits, frac_bits):
        super().__init__(int_bits, frac_bits)
        self._name = "UnsignedFixed%d.%d" % (int_bits, frac_bits)

        # Sobreescribir dtype a unsigned
        if self._size == 1:
            self.dtype = np.uint8
        elif self._size == 2:
            self.dtype = np.uint16
        else:
            self.dtype = np.uint32

    def transform(self, values):
        values = np.array(values, dtype=np.float32)

        scaled = np.round(values * (2 ** self.frac_bits))

        # Rango unsigned: [0, 2^total_bits - 1]
        total_bits = self.int_bits + self.frac_bits
        qmin = 0
        qmax = (2 ** total_bits) - 1  # 255 para Q0.8

        clipped = np.clip(scaled, qmin, qmax)
        self.saturation_count = int(np.sum(scaled != clipped))

        return clipped.astype(self.dtype)

    def inv_transform(self, values):
        values = np.array(values, dtype=self.dtype)
        return (values / (2 ** self.frac_bits)).astype(float)

    # extended_transform no aplica para ventana, pero por completitud:
    def extended_transform(self, values):
        raise NotImplementedError("extended_transform no aplica para tipo unsigned")

# ============================================================
# Float16 Converter (IEEE)
# ============================================================
class Float16TypeConverter(TypeConverter):

    def __init__(self):
        super().__init__()
        self._size = 2
        self._name = "Float16"

    def transform(self, values):

        self.saturation_count = 0
        float16_values = np.float16(values).view(np.uint16)
        return float16_values

    def inv_transform(self, values):

        values = np.array(values, dtype=np.uint16)
        return values.view(np.float16)

    @property
    def extended_type(self):
        return "float"

# ============================================================
# BFloat16 Converter
# ============================================================
class BFloat16TypeConverter(TypeConverter):

    def __init__(self):
        super().__init__()
        self._size = 2
        self._name = "BFloat16"

    def transform(self, values):

        self.saturation_count = 0
        float_32_bits = np.float32(values).view(np.uint32)
        return (float_32_bits >> 16).astype(np.uint16)

    def inv_transform(self, values):

        values = np.array(values, dtype=np.uint16)
        bits = (values.astype(np.uint32) << 16)
        return bits.view(np.float32)

    @property
    def extended_type(self):
        return "float"

# ============================================================
# Quantized Converter (Affine Quantization)
# ============================================================
from dataclasses import dataclass
import numpy as np


@dataclass
class QuantExportParams:
    scale: float  # Escala float (debug)
    zero_point: int  # Zero point int8
    scale_q: int  # Escala cuantizada (depende del modo)
    mode: str  # "q15", "q31", "float"


class QuantizedTypeConverter(TypeConverter):
    # Constantes simbólicas (deben coincidir con C)
    QUANT_SCALE_BITS_Q15 = 15
    QUANT_SCALE_ONE_Q15 = (1 << QUANT_SCALE_BITS_Q15)  # 32768
    QUANT_SCALE_MAX_Q15 = 65535  # uint16_t max

    QUANT_SCALE_BITS_Q31 = 31
    QUANT_SCALE_ONE_Q31 = (1 << QUANT_SCALE_BITS_Q31)  # 2^31
    QUANT_SCALE_MAX_Q31 = ((1 << 31) - 1)  # int32_t max for Q0.31

    def __init__(self, bits=8, symetric=True, signed=False):
        super().__init__()

        # Parámetros principales
        self.symetric = symetric
        self.signed = signed

        # Valores internos
        self.scale = 1.0
        self.zero_pt = 0
        self.min_val = 0.0
        self.max_val = 0.0

        # Configurar bits
        self.set_bits(bits, signed)

        # Nombre descriptivo
        self._name = "Quant%d%c%s" % (
            bits,
            ["A", "S"][symetric],
            ["", "S"][signed]
        )

    def set_bits(self, bits, signed=False):

        if bits == 8:
            self.dtype = np.int8 if signed else np.uint8
        elif bits == 16:
            self.dtype = np.int16 if signed else np.uint16
        else:
            raise ValueError("bits must be 8 or 16")

        self.bits = bits
        self._size = bits // 8

        # Rango cuantizado
        if signed:
            self.max_qint = 2 ** (bits - 1) - 1
            self.min_qint = -(2 ** (bits - 1))
        else:
            self.max_qint = 2 ** bits - 1
            self.min_qint = 0

    def fit(self, values):
        values = np.array(values, dtype=np.float32)

        if self.symetric:
            abs_max = np.max(np.abs(values))
            if abs_max < 1e-8:
                abs_max = 1e-8
            self.scale = abs_max / 127.0
            self.zero_pt = 0 if self.signed else 128
            return

        # Percentile clipping
        low_p = 0.5
        high_p = 99.5
        min_val = np.percentile(values, low_p)
        max_val = np.percentile(values, high_p)

        # Calcular escala y zero point
        q_range = self.max_qint - self.min_qint  # 255 para int8
        scale = (max_val - min_val) / q_range

        if scale < 1e-8:
            scale = 1e-8

        # Zero point sin restricciones artificiales
        zp_float = self.min_qint - min_val / scale
        zp = np.clip(int(round(zp_float)), self.min_qint, self.max_qint)

        # Ajustar si zp está muy descentrado (opcional)
        # Para int8: idealmente zp debería estar cerca de 0
        # Si quieres forzar un rango más equilibrado:
        if abs(zp) > 100:  # advertencia, no corrección forzada
            print(f"WARNING: zp={zp} muy descentrado para rango [{min_val:.3f}, {max_val:.3f}]")

        self.min_val = min_val
        self.max_val = max_val
        self.scale = scale
        self.zero_pt = zp

    def transform(self, values):

        values = np.array(values, dtype=np.float32)

        scaled_values = values / self.scale + self.zero_pt
        rounded = np.round(scaled_values)

        clipped = np.clip(rounded, self.min_qint, self.max_qint)

        # Saturación count
        self.saturation_count = int(np.sum(rounded != clipped))

        return clipped.astype(self.dtype)

    def inv_transform(self, quant_values):

        quant_values = np.array(quant_values, dtype=np.float32)
        return self.scale * (quant_values - self.zero_pt)

    def export_params(self, mode="q15"):
        """
        Exporta parámetros de cuantización para C.

        Modos soportados:
        - "float": valores float (debug)
        - "q15":   scale_q en Q0.15 (uint16_t, rango 0-65535)
        - "q31":   scale_q en Q0.31 (uint32_t, rango 0-2^31-1)
        """

        if mode == "float":
            return QuantExportParams(
                scale=self.scale,
                zero_point=self.zero_pt,
                scale_q=0,
                mode="float"
            )

        if mode == "q15":
            scale_q = int(round(self.scale * self.QUANT_SCALE_ONE_Q15))
            scale_q = max(1, min(scale_q, self.QUANT_SCALE_MAX_Q15))

            return QuantExportParams(
                scale=self.scale,
                zero_point=self.zero_pt,
                scale_q=scale_q,
                mode="q15"
            )

        if mode == "q31":
            # Q0.31: uint32_t, 1.0 = 2^31
            # Escala DIRECTA: scale_q = scale * 2^31
            scale_q = int(round(self.scale * self.QUANT_SCALE_ONE_Q31))

            # Clamp al rango válido y evitar cero
            scale_q = max(1, min(scale_q, self.QUANT_SCALE_MAX_Q31))

            return QuantExportParams(
                scale=self.scale,
                zero_point=self.zero_pt,
                scale_q=scale_q,
                mode="q31"
            )

        raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Converter Manager
# ============================================================
class TypeConverterManager:

    def __init__(self):
        # Lista por instancia (no compartida)
        self.type_converters = []

    def add_type_converter(self, type_converter):
        self.type_converters.append(type_converter)

    def __iter__(self):
        return iter(self.type_converters)

    def test(self, converter, values):

        new_values = converter.inv_transform(converter.fit_transform(values))

        diff_values = values - new_values
        ae = np.absolute(diff_values)
        se = np.square(diff_values)

        return (
            np.mean(ae),
            np.std(ae),
            np.mean(se),
            np.std(se),
            converter.saturation_count
        )


# ============================================================
# Main Test
# ============================================================
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
        conv_man.add_type_converter(FixedTypeConverter(4 + i, 4 - i))

    values = np.array([8.5, 16.33, 255.56, -8.5, -16.33, -300.56])
    print("Original floating point values:", values)

    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Converter", "MAE", "MAE Dev", "MSE", "MSE Dev", "Saturations"]

    for converter in conv_man:
        mae, smae, mse, smse, sat = conv_man.test(converter, values)

        table.add_row([
            converter.name,
            f"{mae:10.4f}",
            f"{smae:.4f}",
            f"{mse:10.4f}",
            f"{smse:.4f}",
            sat
        ])

    print(table)
