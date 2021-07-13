from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, save_model, load_model
import os, sys
from tensorflow import keras
import numpy as np
import cv2

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)

def predict(image1):
    modelpath = path + os.sep + 'models' + os.sep + 'model_2.h5'
    model = load_model(modelpath)
    image = load_img(image1, target_size=(100, 100))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    #image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label

def prediction(image1):
    image = load_img(image1, target_size=(100, 100))

import keras
from PIL import Image, ImageOps
import numpy as np


def predict_image(path_to_model, route):
    new_model = keras.models.load_model(path_to_model)
    image = cv2.imread(route, flags=cv2.IMREAD_COLOR)
    smallimage = cv2.resize(image, (180, 180))
    pred = new_model.predict(preprocess_input(np.array(smallimage).reshape(1, 180, 180, 3)))
    return pred


def classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability