from embedia.utils.c_helper import declare_array
from embedia.model_generator.project_options import ModelDataType
from embedia.core.decision_tree_base_layer import DecisionTreeBaseLayer
import numpy as np


class DecisionTreeClassifier(DecisionTreeBaseLayer):
    support_quantization = False  # support quantized data

    def __init__(self, model, wrapper, **kwargs):
        super().__init__(model, wrapper, **kwargs)

        self._use_data_structure = True  # this layer require data structure initialization

    def calculate_params(self):
        """
        Calculates trainable and non-trainable parameters of the layer.
        """

        n_nodes = self.wrapper.node_count

        # each node stores:
        # threshold, feature_id, right_offset, class_id
        params_per_node = 4

        trainable = 0
        non_trainable = n_nodes * params_per_node

        return (trainable, non_trainable)

    def calculate_ACOPS(self):

        def _max_depth(wrapper):
            left = wrapper.node_left
            right = wrapper.node_right

            def depth(node):
                if left[node] == -1:
                    return 1
                return 1 + max(depth(left[node]), depth(right[node]))

            return depth(0)

        depth = _max_depth(self.wrapper)

        ops_per_node = 5

        return (depth-1)*ops_per_node

    def _implementation_normal_tree(self):
        """
        Generate C code for the decision tree layer initialization function.
        Uses CBuilder.add_array and add_struct for clean, declarative generation.

        The tree nodes are reordered via DFS so that each node's right child
        can be referenced by a relative offset instead of an absolute index,
        saving memory (offset fits in a smaller integer type than a pointer).
        """

        def _reorder_tree(wrapper):
            """
            DFS reorder: returns (order, mapping) where:
              order   = list of original node indices in DFS visit order
              mapping = dict from original index -> new DFS index
            """
            left, right = wrapper.node_left, wrapper.node_right
            order, mapping = [], {}

            def dfs(node):
                mapping[node] = len(order)
                order.append(node)
                if left[node] != -1:
                    dfs(left[node])
                    dfs(right[node])

            dfs(0)
            return order, mapping

        wrapper = self.wrapper
        name = self.name
        node_count = wrapper.node_count
        num_features = wrapper.n_features
        num_classes = wrapper.n_classes

        (data_type, data_converter) = self.model.get_type_converter()
        (conv_data, quant_params) = self.convert_to_embedia_data(data_converter,
                                                                 wrapper.node_thresholds)

        order, mapping = _reorder_tree(wrapper)

        # build node initializer strings in DFS order
        node_inits = []
        for old_i in order:
            feature = wrapper.node_features[old_i]
            threshold = conv_data[old_i]
            value = wrapper.node_values[old_i]

            if feature < 0:
                # leaf node — no split
                feature_str = 'DT_LEAF_NODE'
                right_offset = 0
            else:
                feature_str = str(feature)
                right_offset = mapping[wrapper.node_right[old_i]] - mapping[old_i]

            node_inits.append(f'{{ {threshold}, {feature_str}, {right_offset}, {value} }}')

        cb = self.c_builder

        cb.add()
        with cb.bgn(f'{self.struct_data_type} init_{name}_data(void)'):

            # nodes array — one node per line for readability
            cb.add_array(
                f'static EMBEDIA_MODEL_STORAGE Node',
                'nodes',
                node_inits,
                header_comment=f'[{node_count}]'
            )

            cb.add()

            cb.add_struct(
                f'static EMBEDIA_MODEL_STORAGE decision_tree_classifier_layer_t',
                'tree',
                [f'{num_features}, {num_classes},nodes{quant_params}']
            )

            cb.add()
            cb.add('return tree;')

        return cb.get_code()

    @property
    def function_implementation(self):
        """
        Generate C code with the initialization function of the additional
        structure required by the layer.
        Note: it is important to note the automatically generated function
        prototype (defined in the DataLayer class).

        Returns
        -------
        str
            C function for data initialization
        """

        init_fn = self._implementation_normal_tree()

        return init_fn


    def invoke(self, input_name, output_name):
        """
        Generates C code for the invocation of the EmbedIA function that
        implements the layer/element. The C function must be previously
        implemented in "neural_net.c" and by convention should be called
        "class name" + "_layer".
        For example, for the EmbedIA Dense class associated to the Keras
        Dense layer, the function "dense_layer" must be implemented in
        "neural_net.c"

        Parameters
        ----------
        input_name : str
            name of the input variable to be used in the invocation of the C
            function that implements the layer.
        output_name : str
            name of the output variable to be used in the invocation of the C
            function that implements the layer.

        Returns
        -------
        str
            C code with the invocation of the function that performs the
            processing of the layer in the file "neural_net.c".

        """

        return f'''decision_tree_classifier_layer({self.name}_data, {input_name}, &{output_name});
'''
