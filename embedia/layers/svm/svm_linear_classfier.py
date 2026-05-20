from embedia.core.svm_base_layer import SvmBaseLayer
from embedia.model_generator.project_options import ModelDataType

class SvmLinearClassifier(SvmBaseLayer):

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)
        self._use_data_structure = True  # this layer require data structure initialization

    def calculate_params(self):
        """
        Calculates parameters for LinearSVC (different from standard SVC).

        LinearSVC stores:
        - coef_: shape (n_classes, n_features)
        - intercept_: shape (n_classes,)

        Returns:
            tuple: (#trainable params, #non-trainable params)
        """
        # Get model dimensions
        n_classes = self.wrapper.n_classes
        n_features= self.wrapper.n_features

        # Trainable parameters (coefficients + intercepts)
        trainable = (n_classes * n_features) + n_classes

        # Non-trainable parameters (config)
        non_trainable = 3  # penalty type, loss function, dual flag

        return (trainable, non_trainable)

    def calculate_MAC(self):
        """
        Calculates MAC operations for LinearSVC prediction.

        LinearSVC computes:
        - For each class: dot_product(input, coef_[class]) + intercept

        Returns:
            int: Total MAC operations
        """
        n_classes = self.wrapper.n_classes
        n_features = self.wrapper.n_features

        # Dot product: n_features MACs per class
        # Plus 1 for adding the intercept
        return n_classes * (n_features + 1)

    def calculate_memory(self):
        """
        Calculates memory usage for LinearSVC model.

        Memory components:
        - coef_: n_classes * n_features * 4 bytes
        - intercept_: n_classes * 4 bytes
        - config: 3 bytes (penalty, loss, dual as bytes)

        Returns:
            int: Memory size in bytes
        """
        n_classes = self.wrapper.n_classes
        n_features = self.wrapper.n_features
        dtype_size = 4  # float32

        # Memory breakdown
        components = [
            (n_classes * n_features),  # coef_
            n_classes,  # intercept_
            3  # config flags
        ]

        return (components[0] * dtype_size) + (components[1] * dtype_size) + components[2]

    @property
    def function_implementation(self):
        """
        Generate C code for SVM direct classifier (OvR/dense) initialization.
        """
        # ─────────────────────────────────────────────────────────────────
        # Model data
        # ─────────────────────────────────────────────────────────────────
        struct_type = self.struct_data_type
        name = self.name
        coefficients = self.wrapper.coefficients
        intercepts = self.wrapper.intercepts
        n_classes = self.wrapper.n_classes
        n_features = self.wrapper.n_features

        is_mixed_type = (self.options.data_type == ModelDataType.QUANT8)
        use_comments = (self.options.data_type != ModelDataType.FLOAT)

        # ─────────────────────────────────────────────────────────────────
        # Type converters
        # ─────────────────────────────────────────────────────────────────
        (data_type, coefs_converter) = self.model.get_type_converter()
        conv_coeffs = coefs_converter.fit_transform(coefficients)

        if is_mixed_type:
            (icepts_type, icepts_converter) = self.model.get_type_converter(ModelDataType.FIXED16)
            params = coefs_converter.export_params(mode="q15")
            qp_coefs = f'{{ {params.scale_q}, {params.zero_point} }}'
        else:
            (icepts_type, icepts_converter) = self.model.get_type_converter()
            qp_coefs = ''

        conv_icepts = icepts_converter.fit_transform(intercepts)

        # ─────────────────────────────────────────────────────────────────
        # C code generation
        # ─────────────────────────────────────────────────────────────────
        cb = self.c_builder
        cb.add()

        with cb.bgn(f'{struct_type} init_{name}_data(void)'):

            # Intercepts  [n_classes]  — always compute_t (fixed16 or float)
            cb.add_array(
                f'static EMBEDIA_MODEL_STORAGE {icepts_type}',
                f'{name}_icepts',
                conv_icepts.flatten().tolist(),
                comments=[cb.to_array(intercepts.flatten(), fmt='.6f')] if use_comments else None,
                header_comment=f'[{n_classes}]'
            )
            cb.add()

            # Coefficient matrix  [n_classes x n_features]
            flat_coefs = []
            row_comments_c = [] if use_comments else None

            for i, row in enumerate(conv_coeffs):
                flat_coefs.extend(row.tolist())
                if use_comments:
                    row_comments_c.append(
                        f'class {i} | {cb.to_array(coefficients[i], fmt=".6f")}'
                    )

            cb.add_array(
                f'static EMBEDIA_MODEL_STORAGE {data_type}',
                f'{name}_coefs',
                flat_coefs,
                cols=n_features,
                comments=row_comments_c,
                header_comment=f'[{n_classes} x {n_features}]'
            )
            cb.add()

            # Struct initializer
            fields = [
                f'{n_classes}, {n_features}',
                f'{name}_coefs',
                f'{name}_icepts',
            ]
            if is_mixed_type:
                fields.append(qp_coefs)

            cb.add_struct(
                f'static EMBEDIA_MODEL_STORAGE {struct_type}',
                f'{name}_layer',
                fields
            )

            cb.add()
            cb.add(f'return {name}_layer;')

        return cb.get_code()
    
    def invoke(self, input_name, output_name):
        return f'''svm_direct_classifier_layer(&{self.name}_data, &{input_name}, &{output_name});'''