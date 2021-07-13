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


#Path
dir = os.path.dirname
path = dir(dir(dir(os.path.abspath(__file__))))
print(path)
sys.path.append(path)

labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

def img_train_df(image_size=100):
    '''
    Taking the train images from folders and putting into a dataframe 
    '''

    dicc_train = []
    for i in labels:
        folderpath_train = path + os.sep + "data"+ os.sep +'Training' + os.sep + i
        for j in tqdm(os.listdir(folderpath_train)):
            img = cv2.imread(os.path.join(folderpath_train,j))
            img = cv2.resize(img,(image_size, image_size))
            dicc_train.append({"Image":img, "Label":i, "Fullpath":os.path.join(folderpath_train,j)})

    df_train = pd.DataFrame(dicc_train)
    return df_train


def img_test_df(image_size=100):
    '''
    Taking the test images from folders and putting into a dataframe 
    '''

    dicc_test = []
    for i in labels:
        folderpath_test = path + os.sep + 'data'+ os.sep +'Testing'+ os.sep + i
        for j in tqdm(os.listdir(folderpath_test)):
            img = cv2.imread(os.path.join(folderpath_test,j))
            img = cv2.resize(img,(image_size,image_size))
            dicc_test.append({"Image":img, "Label":i, "Fullpath":os.path.join(folderpath_test,j)})
    df_test = pd.DataFrame(dicc_test)
    return df_test


def X_train(image_size):
    X = []
    y = []
    for i in labels:
        folderpath_test = path + os.sep + 'data'+ os.sep +'Training'+ os.sep + i
        for j in tqdm(os.listdir(folderpath_test)):
            img = cv2.imread(os.path.join(folderpath_test,j))
            img = cv2.resize(img,(image_size,image_size))
            X.append(img)
            y.append(i)
    return np.array(X), np.array(y)
    
def X_test(image_size):
    X = []
    y = []
    for i in labels:
        folderpath_test = path + os.sep + 'data'+ os.sep +'Testing'+ os.sep + i
        for j in tqdm(os.listdir(folderpath_test)):
            img = cv2.imread(os.path.join(folderpath_test,j))
            img = cv2.resize(img,(image_size,image_size))
            X.append(img)
            y.append(i)
    return np.array(X), np.array(y)


def arraytrain(df_train):
    ''' Photos train to array'''
    X_train = np.array(df_train['Image'])
    y_train = np.array(df_train['Label'])
    return X_train, y_train


def arraytest(df_test):
    ''' Photos test to array'''
    X_test = np.array(df_test['Image'])
    y_test = np.array(df_test['Label'])
    return X_test, y_test

def access_train2(h, w, batch_size):
    train_dir = path + os.sep + "data" + os.sep + "Training"
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training", 
    seed=153,
    image_size=(h, w),
    batch_size=batch_size)
    return train_data

def access_val2(h, w, batch_size):
    train_dir = path + os.sep + "data" + os.sep + "Training"
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation", 
    seed=153,
    image_size=(h, w),
    batch_size=batch_size)
    return val_data

def access_test2(h, w, batch_size):
    test_dir = path + os.sep + "data" + os.sep + "Testing"
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed = 153,
    image_size = (h, w),
    batch_size =batch_size)
    return test_data

