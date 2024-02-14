# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:32:41 2023

@author: cesar
"""

from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import cifar10
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt  # Importar matplotlib

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import confusion_matrix
import seaborn as sns

import joblib

(IW, IH) = (16, 16)

# Cargar los datos de CIFAR-10 y dividirlos en conjuntos de entrenamiento y prueba
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_resized = []
x_test_resized = []
for img in x_train:
    img = Image.fromarray(img)
    img = img.resize((IW, IH))
    x_train_resized.append(np.array(img))
for img in x_test:
    img = Image.fromarray(img)
    img = img.resize((IW, IH))
    x_test_resized.append(np.array(img))

# Normalizar los datos de imagen
x_train = np.array(x_train_resized).astype("float32") / 255.0
x_test = np.array(x_test_resized).astype("float32") / 255.0

# Convertir las etiquetas a vectores de un solo valor
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model = models.Sequential()
#model.add(layers.Input(shape=(IW, IH, 3)))
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IH, IW, 3)))  # Convolutional Layer
model.add(layers.DepthwiseConv2D((3, 3), activation='relu'))  # DepthwiseConv2D Layer
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Convolutional Layer
model.add(layers.DepthwiseConv2D((3, 3), activation='relu'))  # DepthwiseConv2D Layer
model.add(layers.Flatten())  # Flatten Layer
model.add(layers.Dense(64, activation='relu'))  # Dense Layer
model.add(layers.Dense(10, activation='softmax'))  # Output Layer with 10 classes

# Compilar el modelo
model.compile(optimizer='adam',
              #loss='sparse_categorical_crossentropy',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
# Definir la función de programación para ajustar el valor de aprendizaje
def lr_schedule(epoch):
    return 0.001

# Definir el modificador de programación de tasa de aprendizaje
lr_scheduler = LearningRateScheduler(lr_schedule)

# Definir la parada temprana
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Entrenar el modelo con parada temprana y modificador de programación de tasa de aprendizaje
history = model.fit(x_train, y_train, batch_size=200, epochs=50,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, lr_scheduler])



# Graficar las curvas de entrenamiento
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluar el modelo
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Predicciones del modelo
y_pred = model.predict(x_test)

# Obtener las clases predichas
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Visualizar la matriz de confusión con seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# Guardar el modelo y muestras
SAMPLES_FILE = 'cifar10_20samples_16x16.sav'
model.save('cifar10_depthwise_model.h5')
joblib.dump((x_test[0:20, ::], y_test[0:20, ::].argmax(1)), SAMPLES_FILE)
