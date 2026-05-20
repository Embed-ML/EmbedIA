/**
 * @struct filter_t
 * @brief Represents a single filter (kernel) in a convolutional layer.
 *
 * Contains weights and bias for one filter.
 */
typedef struct {
    EMBEDIA_MODEL_STORAGE storage_t * weights;  /**< Pointer to the filter weights (quantized, in flash) */
    compute_t bias;             /**< Bias value for the filter (runtime precision) */
} filter_t;

/**
 * @struct conv2d_layer_t
 * @brief Represents a 2D convolutional layer.
 */
typedef struct {
    uint16_t n_filters;     /**< Number of filters in the layer */
    EMBEDIA_MODEL_STORAGE filter_t * filters;     /**< Array of filters */
    uint16_t channels;      /**< Number of input channels */
    size2d_t kernel;        /**< Kernel size (height, width) */
    uint8_t padding;        /**< Padding type: PAD_SAME or PAD_VALID */
    size2d_t strides;       /**< Stride size (vertical, horizontal) */
    qparam_t qparam;
} conv2d_layer_t;


/**
 * @struct conv1d_layer_t
 * @brief Represents a 1D convolutional layer for temporal data processing.
 *
 * @details Used for time series analysis, audio processing, and sequential data
 *
 * - n_filters    Number of filters in the layer
 * - filters      Array of filters (weights and biases)
 * - channels     Number of input channels/features
 * - kernel_size  Kernel size (temporal dimension)
 * - padding      Padding type: PAD_SAME or PAD_VALID
 * - stride       Stride size (temporal stride)
 */
typedef struct {
    uint16_t n_filters;     /**< Number of filters in the layer */
    EMBEDIA_MODEL_STORAGE filter_t * filters;     /**< Array of filters */
    uint16_t channels;      /**< Number of input channels */
    uint16_t kernel_size;   /**< Kernel size (length) */
    uint8_t padding;        /**< Padding type: PAD_SAME or PAD_VALID */
    uint16_t stride;        /**< Stride size */
    qparam_t qparam;
} conv1d_layer_t;


/**
 * @struct depthwise_conv2d_layer_t
 * @brief Represents a depthwise 2D convolutional layer.
 *
 * Each input channel is convolved with a separate filter.
 */
typedef struct {
    EMBEDIA_MODEL_STORAGE storage_t * weights;      /**< Weights for depthwise filters (quantized, in flash) */
    EMBEDIA_MODEL_STORAGE compute_t * bias;         /**< Bias values per channel (runtime precision) */
    uint16_t channels;          /**< Number of input channels (and filters) */
    size2d_t kernel_sz;         /**< Kernel size (height, width) */
    uint8_t padding;            /**< Padding type: PAD_SAME or PAD_VALID */
    size2d_t strides;           /**< Stride size (vertical, horizontal) */
    qparam_t qparam;            /**< Quantization params for ST2CO conversion */
} depthwise_conv2d_layer_t;


/**
 * @struct separable_conv2d_layer_t
 * @brief Represents a separable 2D convolutional layer.
 *
 * Composed of a depthwise convolution followed by a pointwise (1x1) convolution.
 */
typedef struct {
    uint16_t n_filters;             /**< Number of pointwise filters (output channels) */
    filter_t * point_filters;       /**< Array of 1x1 filters (pointwise) */
    uint16_t point_channels;        /**< Number of input channels for pointwise step */
    size2d_t point_kernel_sz;       /**< Kernel size for pointwise convolution (should be 1x1) */
    EMBEDIA_MODEL_STORAGE storage_t * depth_weights;   /**< Depthwise weights (quantized, in flash) */
    EMBEDIA_MODEL_STORAGE compute_t * depth_bias;      /**< Bias values per channel (runtime precision) */
    uint16_t depth_channels;        /**< Number of input channels for depthwise step */
    size2d_t depth_kernel_sz;       /**< Kernel size for depthwise convolution */
    uint8_t padding;                /**< Padding type: PAD_SAME or PAD_VALID */
    size2d_t strides;               /**< Stride size for both steps */
    qparam_t qparam;                /**< Quantization params for ST2CO conversion */
} separable_conv2d_layer_t;


/**
 * @struct dense_layer_t
 * @brief Represents a fully connected (dense) layer.
 */
typedef struct {
    uint16_t input_size;        /**< Number of input neurons */
    uint16_t output_size;       /**< Number of output neurons */
    EMBEDIA_MODEL_STORAGE storage_t *weights;            /**< Weight matrix (quantized, in flash) */
    EMBEDIA_MODEL_STORAGE compute_t *biases;             /**< Bias vector (runtime precision) */
    qparam_t qparam;            /**< Quantization params for ST2CO conversion */
} dense_layer_t;


/**
 * @struct pooling2d_layer_t
 * @brief Configuration for 2D pooling layers (max or average).
 */
typedef struct {
    uint16_t size;      /**< Pooling window size (assumed square: size x size) */
    uint16_t strides;   /**< Stride of the pooling window */
} pooling2d_layer_t;


/**
 * @struct pooling1d_layer_t
 * @brief Configuration for 1D pooling layers (max or average).
 */
typedef struct {
    uint16_t size;      /**< Pooling window size */
    uint16_t strides;   /**< Stride of the pooling window */
} pooling1d_layer_t;


/** @} */ // end of layer_structures




/**
 * @struct batch_normalization_layer_t
 * @brief Parameters for batch normalization layer.
 *
 * Implements: output = (input - moving_mean) * moving_inv_std_dev + beta
 * Optimized as: output = input * moving_inv_std_dev + std_beta
 */
typedef struct {
    uint32_t length;                        /**< Number of channels (length of parameter vectors) */
    EMBEDIA_MODEL_STORAGE compute_t *moving_inv_std_dev;        /**< Precomputed: gamma / sqrt(variance + epsilon) */
    EMBEDIA_MODEL_STORAGE compute_t *std_beta;                  /**< Precomputed: beta - moving_mean * moving_inv_std_dev */
} batch_normalization_layer_t;

