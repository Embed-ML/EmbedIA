import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    # Input,
    Dense,
    Conv2D,
    SeparableConv2D,
    DepthwiseConv2D,
    Flatten,
    BatchNormalization,
    # Activation,
    ReLU,
    LeakyReLU,
    Softmax,
    MaxPooling2D,
    AveragePooling2D
)


def build_all_layers_model():
    model = Sequential()

    model._name = "EmbedIA_all_layers"

    model.add(
        Conv2D(8, kernel_size=(2, 2), input_shape=(28, 28, 1),
               activation='LeakyReLU', name='Conv2D_1')
        )
    model.add(DepthwiseConv2D(8, (2, 2), activation='linear', name='DepthwiseConv2D_1'))

    model.add(MaxPooling2D(pool_size=(2, 2), name='MaxPool_1'))
    #model.add(BatchNormalization(name='Batch_1'))

    model.add(
        Conv2D(12, kernel_size=(2, 2), activation='ReLU', name='Conv2D_2')
        )
    model.add(AveragePooling2D(pool_size=(2, 2), name='AvgPool_2'))
    model.add(ReLU(name='ReLU_2'))

    model.add(
        SeparableConv2D(15, kernel_size=(2, 2), activation='sigmoid',
                        name='SepConv2D_3')
        )

    # model.add(AveragePooling2D(pool_size=(2, 2), name='AvgPool_3'))

    model.add(Flatten(name='Flatten_4'))
    model.add(Dense(32, name='Dense_4', activation='tanh'))
    model.add(Dense(10, name='Dense_5', activation='linear'))
    model.add(Softmax(name='SoftMax_6'))

    model.compile()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Carga los datos de MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normaliza los datos de entrada
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Agrega una dimensión de canal a los datos de entrada
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Convierte las etiquetas a one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Entrena el modelo
    model.fit(x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test, y_test))

    idx =-5
    # print(model.layers[idx].get_weights()[0].shape)
    # print(model.layers[idx].get_weights()[1].shape)
    # print(model.layers[idx].get_weights()[2].shape)
    # print(model.layers[idx].get_weights()[2])
    return model
