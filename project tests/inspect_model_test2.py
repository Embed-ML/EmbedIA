# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 00:46:45 2022

@author: cesar
"""
import sys
# add parent folder to path in order to find EmbedIA folder
sys.path.insert(0, '..')
import joblib
import numpy as np
from embedia.utils.model_inspector import ModelInspector
from tensorflow.keras.models import load_model


# MODEL_FILE = 'models/MNIST_14x14_model_2DIMG_norm0-1_tr-acc93.79_te-acc93.61.h5'
# SAMPLES_FILE = 'samples/MNIST_20samples_14x14_2D.sav'
# SAMPLE_INSPECT = 'mnist_img_2D'

# model = load_model(MODEL_FILE)
# model._name = "mnist_img_2D"

MODEL_FILE = 'models/MNIST_12x12_model(LReLU).h5'
SAMPLES_FILE = 'samples/MNIST_20samples_12x12.sav'
SAMPLE_INSPECT = 'mnist_img_3D'

model = load_model(MODEL_FILE)
model._name = "mnist12x12"


model.summary()

(samples, ids) = joblib.load(SAMPLES_FILE)

inspector = ModelInspector(model)

s_id = 0
# print(inspector.as_string(samples[s_id]))


sample = samples[s_id]
class_id = ids[s_id]
#sample =  np.array([sample[:, :,0], sample[:, :, 1]])




inspector.save(SAMPLE_INSPECT+f'sample{s_id}.txt', sample, ln_break=-1)

res = model.predict(samples)

print((res * 100).astype('int'))