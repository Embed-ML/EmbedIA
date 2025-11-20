from embedia.core.layer_wrapper import LayerWrapper, OutputPredictionType
import numpy as np

class ScikitLearnWrapper(LayerWrapper):
    #@property
    #def name(self):
    #    return self._target.__class__.name
    pass


class SKLNormWrapper(ScikitLearnWrapper):
    @property
    def div_values(self):
        return None

    @property
    def sub_values(self):
        return None

    @property
    def input_shape(self):
        return self.div_values.shape

    @property
    def output_shape(self):
        return self.div_values.shape


class SKLMinMaxScalerWrapper(SKLNormWrapper):

    @property
    def div_values(self):
        return self._target.data_range_

    @property
    def sub_values(self):
        return self._target.data_min_

    @property
    def funcion_name(self):
        return 'min_max'


class SKLMaxAbsScalerWrapper(SKLNormWrapper):
    @property
    def div_values(self):
        return self._target.max_abs_

    @property
    def funcion_name(self):
        return 'max_abs'


class SKLStandardScalerWrapper(SKLNormWrapper):
    @property
    def div_values(self):
        return self._target.scale_

    @property
    def sub_values(self):
        return self._target.mean_

    @property
    def funcion_name(self):
        return 'standard'


class SKLRobustScalerWrapper(SKLNormWrapper):
    @property
    def div_values(self):
        return self._target.scale_

    @property
    def sub_values(self):
        return self._target.center_

    @property
    def funcion_name(self):
        return 'robust'


class SKLKnnWrapper(ScikitLearnWrapper):
    SUPPORTES_DISTANCES = ['euclidean', 'manhattan', 'cosine', 'chebyshev', 'braycurtis', 'canberra']

    @property
    def n_classes(self):
        return len(self._target.classes_)

    @property
    def n_neighbors(self):
        return self._target.n_neighbors

    @property
    def n_samples(self):
        return self._target.n_samples_fit_

    @property
    def n_features(self):
        return self._target.n_features_in_

    @property
    def fit_x(self):
        return self._target._fit_X

    @property
    def y(self):
        return self._target._y

    @property
    def input_shape(self):
        return self._target._fit_X[0].shape

    @property
    def output_shape(self):
        return (None, len(self._target.classes_))


    def _uniform_function_name(self, dist_name, extra_param):
        dist_name = dist_name.lower()
        if (dist_name == 'manhattan' or dist_name == 'cityblock' or
                dist_name == 'l1' or (dist_name == 'minkowski' and extra_param == 1)):
            return 'manhattan'
        if dist_name == 'euclidean' or dist_name == 'l2' or (dist_name == 'minkowski' and extra_param == 2):
            return 'euclidean'

        return dist_name

    @property
    def distance_function(self):

        extra_param = self._target.p
        fn_name = self._target.metric

        fn_name = self._uniform_function_name(fn_name, extra_param)
        if fn_name not in self.SUPPORTES_DISTANCES or (fn_name=='minkowski' and extra_param not in [1,2]):
            raise Exception('Unknown distance function for KNN algorithm')

        return fn_name

    @property
    def activation(self):
        return None



class SKLSvmWrapper(ScikitLearnWrapper):

    def _expand_dual_coef(self, model):
        """
        Converts scikit-learn's compressed SVC coefficients to expanded format.
        Returns:
        - expanded_coef: Array of shape [n_pairs][n_SV] with complete coefficients
        - intercepts: Array of shape [n_pairs] with bias terms
        """
        n_classes = len(model.classes_)
        n_pairs = n_classes * (n_classes - 1) // 2
        total_sv = sum(model.n_support_)

        # Initialize expanded matrix
        expanded_coef = np.zeros((n_pairs, total_sv))

        # Class to dual_coef_ row mapping
        class_to_coef_row = {c: i for i, c in enumerate(model.classes_[:-1])}

        pair_idx = 0
        sv_starts = np.concatenate(([0], np.cumsum(model.n_support_)[:-1]))

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                # SV ranges for classes i and j
                start_i, end_i = sv_starts[i], sv_starts[i] + model.n_support_[i]
                start_j, end_j = sv_starts[j], sv_starts[j] + model.n_support_[j]

                # Coefficients for class i
                if i in class_to_coef_row:
                    row_i = class_to_coef_row[i]
                    expanded_coef[pair_idx, start_i:end_i] = model.dual_coef_[row_i, start_i:end_i]

                # Coefficients for class j
                if j > 0:  # Class 0 has no dedicated row
                    row_j = j - 1
                    expanded_coef[pair_idx, start_j:end_j] = model.dual_coef_[row_j, start_j:end_j]

                pair_idx += 1
        return expanded_coef

    @property
    def kernel(self):
        """Returns a tuple containing all kernel parameters in order:
        (kernel_type, gamma, coef0, degree) where:
        - kernel_type: one of 'linear', 'poly', 'rbf', 'sigmoid'
        - gamma: kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        - coef0: independent term in kernel function
        - degree: degree of the polynomial kernel
        """
        return (
            self._target.kernel,
            self._target._gamma if hasattr(self._target, '_gamma') else self._target.gamma,
            self._target.coef0,
            self._target.degree
        )

    @property
    def input_shape(self):
        """Shape of input features: (None, n_features)"""
        return (None, self._target.n_features_in_)

    @property
    def output_shape(self):
        """Shape of output predictions: (None, n_classes)"""
        return (None, len(self._target.classes_))

    @property
    def classes(self):
        """Array of class labels"""
        return self._target.classes_

    @property
    def support(self):
        """Indices of support vectors"""
        return self._target.support_

    @property
    def n_support(self):
        """Number of support vectors per class"""
        return self._target.n_support_

    @property
    def offsets_classes(self):
        """Precomputed offsets for support vectors per class"""
        offsets = [0]
        for count in self._target.n_support_:
            offsets.append(offsets[-1] + count)
        return np.array(offsets)

    @property
    def n_features(self):
        """Number of input features"""
        return self._target.n_features_in_

    @property
    def support_vectors(self):
        """Array of all support vectors"""
        return self._target.support_vectors_

    @property
    def coefficients(self):
        """Tuple of (expanded coefficients, intercepts)"""
        return (self._expand_dual_coef(self._target), self._target.intercept_)

    @property
    def activation(self):
        """Activation function (always None for SVM)"""
        return None

    @property
    def output_prediction_type(self):
        """Type of prediction output (class probabilities)"""
        return OutputPredictionType.CLASS_PROBABILITIES


class SKLSvmLinearWrapper(ScikitLearnWrapper):

    @property
    def input_shape(self):
        """Shape of input features: (None, n_features)"""
        return (None, self._target.n_features_in_)

    @property
    def output_shape(self):
        """Shape of output predictions: (None, n_classes)"""
        return (None, len(self._target.classes_))

    @property
    def classes(self):
        """Array of class labels"""
        return self._target.classes_

    @property
    def n_features(self):
        """Number of input features"""
        return self._target.n_features_in_

    @property
    def coefficients(self):
        """Model parameters in OVR format:
        Returns:
            tuple: (coef_array, intercept_array) where
                - coef_array: shape [n_classes, n_features]
                - intercept_array: shape [n_classes,]
        """
        return (self._target.coef_, self._target.intercept_)

    @property
    def activation(self):
        """Activation function (always None for SVM)"""
        return None

    @property
    def output_prediction_type(self):
        """Type of prediction output (class probabilities)"""
        return OutputPredictionType.CLASS_PROBABILITIES



class SKLDecisionTreeClassifierWrapper(ScikitLearnWrapper):
    @property
    def node_count(self):
        return self._target.tree_.node_count

    @property
    def node_feature(self):
        """Get the feature indices for each node in the decision tree.
        Returns:
            numpy.ndarray: Array where each element represents:
                - For internal nodes: Index of the feature to split on (>=0)
                - For leaf nodes: -1 (modified from original -2 to match common conventions)
        """
        features = self._target.tree_.feature.copy()  # Get a copy of the feature indices
        features[features < 0] = -1  # Replace all negative values (-2 for leaves) with -1
        return features

    @property
    def node_threshhold(self):
        """Get the threshold values for each node in the decision tree.
        Returns:
            numpy.ndarray: Array of threshold values used for splitting at each node.
                For leaf nodes, this value is typically meaningless (often -2).
        """
        return self._target.tree_.threshold

    @property
    def value(self):
        """Get the predicted class values for each node in the decision tree.
        For each node, determines the class with maximum probability/score.
        Returns:
            list: Array where each element represents:
                - For leaf nodes: The predicted class index
                - For internal nodes: The class with highest probability at that node
        """
        value = []
        for v in self._target.tree_.value:
            value.append(v[0].argmax())
        return value

    @property
    def node_children_left(self):
        """Get the left children indices for each node in the decision tree.
        Returns:
            numpy.ndarray: Array where each element represents:
                - For internal nodes: Index of left child node
                - For leaf nodes: -1 (indicating no child)
        """
        return self._target.tree_.children_left

    @property
    def node_children_right(self):
        """Get the right children indices for each node in the decision tree.
        Returns:
            numpy.ndarray: Array where each element represents:
                - For internal nodes: Index of right child node
                - For leaf nodes: -1 (indicating no child)
        """
        return self._target.tree_.children_right

    @property
    def n_features(self):
        """Get the number of features used by the decision tree.
        Returns:
            int: Total number of features the tree was trained on.
        """
        return self._target.tree_.n_features

    @property
    def input_shape(self):
        """Get the expected input shape for the decision tree model.
        Returns:
            tuple: Shape in format (batch_size, n_features) where:
                - batch_size is None (variable size)
                - n_features is number of input features
        """
        return (None, self._target.tree_.n_features)

    @property
    def output_shape(self):
        """Get the output shape of the decision tree model.
        Returns:
            tuple: Shape in format (batch_size, n_outputs) where:
                - batch_size is None (variable size)
                - n_outputs is number of output classes
        """
        return (None, self._target.n_outputs_)

    @property
    def output_prediction_type(self):
        return OutputPredictionType.DIRECT_CLASS_ID


class SKLLogisticRegressionWrapper(ScikitLearnWrapper):
    
    """
    Wrapper para el modelo LogisticRegression de scikit-learn."
    Extrae los parámetros que la implementacione en C necesita
    """ 
    @property
    def n_classes(self):
        """Numero de clases de salida"""
        return len(self._target.classes_)

    @property
    def n_features(self):
        """Numero de características de entrada"""
        return self._target.n_features_in_

    @property
    def classes(self):
        """Array con las etiquetas de las clases"""
        return self._target.classes_

    @property
    def weights(self):
        """
        Puntero a los pesos del modelo.
        Shape: (n_classes, n_features).
        """
        return self._target.coef_
    
    @property
    def bias(self):
        """
        Puntero a los sesgos/bias del modelo
        Shape: (n_classes,)
        """
        return self._target.intercept_
    
    @property
    def input_shape(self):
        """
        La forma de una sola muestra de entada
        """
        return (self.n_features,)

    @property
    def output_shape(self):
        """
        La forma de salida va a ser una predicción
        """
        return (1, )

    def actuvation(self):
        return None
    
