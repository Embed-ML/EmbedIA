from embedia.core.svm_base_layer import SvmBaseLayer

from embedia.model_generator.project_options import ModelDataType

class SvmClassifier(SvmBaseLayer):
        
    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization

    def calculate_params(self):
        """
        Calculates trainable and non-trainable parameters.
        Now supports sparse representation.
        """

        n_SV = self._wrapper.n_support_vectors
        n_features = self._wrapper.n_features
        n_pairs = self._wrapper.n_classes * (self._wrapper.n_classes - 1) // 2

        # Sparse view
        sparse_pairs = self.convert_coefficients_sparse(self.wrapper.coefficients)

        n_nonzero = sum(len(p) for p in sparse_pairs)

        trainable = (
                (n_SV * n_features) +  # support vectors
                n_nonzero +  # sparse coefficients
                n_pairs  # intercepts
        )

        non_trainable = 4  # kernel config

        return (trainable, non_trainable)

    def calculate_MAC(self):
        """
        Calculates MAC operations per inference.
        Uses sparse coefficients.
        """

        n_SV = self._wrapper.n_support_vectors
        n_features = self._wrapper.n_features
        n_pairs = self._wrapper.n_classes * (self._wrapper.n_classes - 1) // 2

        kernel_type = self._wrapper.kernel_type
        #gamma, coef0, degree = self._wrapper.kernel_params

        # ===== Kernel cost =====

        if kernel_type == 'linear':
            mac_per_kernel = n_features

        elif kernel_type == 'poly':
            mac_per_kernel = n_features + 2  # gamma*x + coef0 + pow

        elif kernel_type == 'rbf':
            mac_per_kernel = 2 * n_features + 1  # diff^2 + sum + exp

        elif kernel_type == 'sigmoid':
            mac_per_kernel = n_features + 2

        else:
            mac_per_kernel = n_features

        kernel_macs = n_SV * mac_per_kernel

        # ===== Decision cost (sparse) =====


        sparse_pairs = self.convert_coefficients_sparse(self.wrapper.coefficients)

        decision_macs = sum(len(p) for p in sparse_pairs)

        return kernel_macs + decision_macs

    def calculate_memory(self):
        """
        Calculates model memory footprint (bytes).
        Sparse-aware.
        """

        n_SV = self._wrapper.n_support_vectors
        n_features = self._wrapper.n_features
        n_pairs = self._wrapper.n_classes * (self._wrapper.n_classes - 1) // 2

        sparse_pairs = self.convert_coefficients_sparse(self._wrapper.coefficients)

        n_nonzero = sum(len(p) for p in sparse_pairs)

        # Sizes
        float_size = 4
        index_size = 2  # uint16

        # ===== Components =====

        mem_vectors = n_SV * n_features * float_size

        # sparse: (idx + coef)
        mem_coefs = n_nonzero * (index_size + float_size)

        mem_intercepts = n_pairs * float_size

        mem_kernel = (
                1 +  # uint8 kernel type
                3 * float_size  # gamma, coef0, degree
        )

        return mem_vectors + mem_coefs + mem_intercepts + mem_kernel

    @property
    def function_implementation(self):
        """
        Generate C code for SVM model initialization (OvO strategy).
        """
        import numpy as np
        from embedia.wrappers.svm_base import SVMStrategy

        # ─────────────────────────────────────────────────────────────────
        # Model data
        # ─────────────────────────────────────────────────────────────────
        vectors = self.wrapper.support_vectors
        intercepts = self.wrapper.intercepts
        n_classes = self.wrapper.n_classes
        n_SV = self.wrapper.n_support_vectors
        n_features = self.wrapper.n_features
        n_pairs = n_classes * (n_classes - 1) // 2
        name = self.name
        struct_type = self.struct_data_type

        kernel_type = self.wrapper.kernel_type
        gamma, intercept_k, degree = self.wrapper.kernel_params

        is_mixed_type = (self.options.data_type == ModelDataType.QUANT8)
        use_comments = (self.options.data_type != ModelDataType.FLOAT)

        # ─────────────────────────────────────────────────────────────────
        # Kernel type mapping
        # ─────────────────────────────────────────────────────────────────
        kernel_type_c = {
            'linear': 'SVM_KERNEL_LINEAR',
            'poly': 'SVM_KERNEL_POLY',
            'rbf': 'SVM_KERNEL_RBF',
            'sigmoid': 'SVM_KERNEL_SIGMOID',
        }.get(kernel_type.lower(), 'SVM_KERNEL_LINEAR')

        # ─────────────────────────────────────────────────────────────────
        # Type converters
        # ─────────────────────────────────────────────────────────────────
        (data_type, vectors_converter) = self.model.get_type_converter()
        (_, coefs_converter) = self.model.get_type_converter()

        conv_vectors = vectors_converter.fit_transform(vectors)

        if is_mixed_type:
            (icepts_type, icepts_converter) = self.model.get_type_converter(ModelDataType.FIXED16)
            (_, kernel_converter) = self.model.get_type_converter(ModelDataType.FIXED16)
            qp_vectors = f'{{ {vectors_converter.export_params(mode="q15").scale_q}, {vectors_converter.export_params(mode="q15").zero_point} }}'
            qp_coefs = f'{{ {coefs_converter.export_params(mode="q15").scale_q}, {coefs_converter.export_params(mode="q15").zero_point} }}'
        else:
            (icepts_type, icepts_converter) = self.model.get_type_converter()
            (_, kernel_converter) = self.model.get_type_converter()
            qp_vectors = ''
            qp_coefs = ''

        gamma_c = kernel_converter.transform(gamma)
        intercept_kc = icepts_converter.transform(intercept_k)
        conv_icepts = icepts_converter.fit_transform(intercepts)

        # ─────────────────────────────────────────────────────────────────
        # Sparse OvO pairs
        # ─────────────────────────────────────────────────────────────────
        sparse_pairs = self.convert_coefficients_sparse(
            self.wrapper.coefficients, threshold=1e-6, sort=True
        )
        for pair in sparse_pairs:
            for i in range(len(pair)):
                pair[i] = (pair[i][0], coefs_converter.transform(pair[i][1]))

        # ─────────────────────────────────────────────────────────────────
        # C code generation
        # ─────────────────────────────────────────────────────────────────
        cb = self.c_builder
        cb.add()

        with cb.bgn(f'{struct_type} init_{name}_data(void)'):

            # Intercepts
            cb.add_array(
                f'static EMBEDIA_MODEL_STORAGE {icepts_type}',
                f'{name}_icepts',
                conv_icepts.flatten().tolist(),
                comments=[cb.to_array(intercepts.flatten(), fmt='.6f')] if use_comments else None,
                header_comment=f'[{len(conv_icepts.flatten())}]'
            )
            cb.add()

            # Support vectors
            flat_vectors = []
            row_comments_v = [] if use_comments else None

            for i, vec in enumerate(conv_vectors):
                flat_vectors.extend(vec.tolist())
                if use_comments:
                    row_comments_v.append(f'SV{i:<3d} | {cb.to_array(vectors[i], fmt=".6f")}')

            cb.add_array(
                f'static EMBEDIA_MODEL_STORAGE {data_type}',
                f'{name}_vectors',
                flat_vectors,
                cols=n_features,
                comments=row_comments_v,
                header_comment=f'[{n_SV} x {n_features}]'
            )
            cb.add()

            # Sparse coefficient pairs
            pair_struct_entries = []

            for pair_idx, sparse_entries in enumerate(sparse_pairs):
                pair_name = f'{name}_pair_{pair_idx}_data'

                cb.add(f'static EMBEDIA_MODEL_STORAGE svm_coef_sparse_t {pair_name}[] = {{')
                for sv_idx, coef in sparse_entries:
                    cb.add(f'    {{ {sv_idx}, {coef} }},')
                cb.add('};')
                cb.add()

                pair_struct_entries.append(f'{{ {len(sparse_entries)}, {pair_name} }}')

            cb.add(f'static EMBEDIA_MODEL_STORAGE svm_pair_sparse_t {name}_pairs[] = {{')
            for entry in pair_struct_entries:
                cb.add(f'    {entry},')
            cb.add('};')
            cb.add()

            cb.add_struct(
                f'static EMBEDIA_MODEL_STORAGE {struct_type}',
                f'{name}_layer',
                [
                    f'{n_classes}, {n_features}, {n_SV}, {n_pairs}',
                    f'{{ {kernel_type_c}, {gamma_c}, {intercept_kc}, {degree} }}',
                    f'{name}_vectors',
                    f'{name}_pairs',
                    f'{name}_icepts',
                    f'{qp_vectors}',
                    f'{qp_coefs}'
                ]
            )

            cb.add()
            cb.add(f'return {name}_layer;')

        return cb.get_code()

    def invoke(self, input_name, output_name):
        kernel_type = self._wrapper.kernel_type.lower()
        return f'''svm_{kernel_type}_classifier_layer(&{self.name}_data, &{input_name}, &{output_name});'''