import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os,sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np


def generator(bool,X_train):
    if bool== True:
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True)
        return datagen.fit(X_train)
    if bool == False:
        return None

def new_y(y_train,y_test, labels):
    y_train_new = []
    for i in y_train:
        y_train_new.append(labels.index(i))
    y_train = y_train_new
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test_new = []
    for i in y_test:
        y_test_new.append(labels.index(i))
    y_test = y_test_new
    y_test = tf.keras.utils.to_categorical(y_test)
    return y_train, y_test

def model(method, X_train, y_train, image_size=100):
    if method== 1:
        effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))
        model = effnet.output
        model = tf.keras.layers.GlobalAveragePooling2D()(model)
        model = tf.keras.layers.Dropout(rate=0.5)(model)
        model = tf.keras.layers.Dense(4,activation='softmax')(model)
        model = tf.keras.models.Model(inputs=effnet.input, outputs = model)
        model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])

        model.fit(X_train,y_train,validation_split=0.2, epochs =12, verbose=1, batch_size=32)

        return model
    if method == "svc":
        clf = SVC(max_iter=99)
        clf.fit(train_data)
        return clf
            


def predict_model(model, X_test, y_test):
    pred = model.predict(X_test)
    #pred = np.argmax(pred,axis=1)
    #y_test_new = np.argmax(y_test,axis=1)
    return classification_report(y_test,pred)


from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, decode_predictions, preprocess_input
from sklearn.metrics import classification_report,confusion_matrix
