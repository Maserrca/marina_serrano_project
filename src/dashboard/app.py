'''
“Welcome”: here the user must see some informationabout theproject and your profile.2
ii.“Visualization”: here the user must see some graphs/images of yourdata.iii.“Json API-Flask”: 
here the user must see a table/dataframefrom yourcleaned data. Optional if you do not have dataframedata.iv.
“Model Prediction”: here the user must see model informationandmust be able to execute a prediction using 
your savedmodel. If theprediction must be done with text, then the user mustbe able to writetext. 
If the prediction must be done with an image/file,then the usermust be able to upload an image/file.v.
“Models From SQL Database”: here the user will seethe modelscomparison from MySQL. The table of comparison iscalled“model_comparasion”. Its columns are:  [“model”, “parameters”,“recall”,“score”] or [“model”, “parameters”, “rmse”, “r2”]vi.Others you need. For example, the table “predictions”if you have it
'''

import os
import sys
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests
import plotly.express as px
from src.utils.visualization_tb import *
from tensorflow import keras
from src.utils.dashboard_tb import *
import cv2
from PIL import Image, ImageOps

#----------------- We define the function here because it is causing problems again ----------------
def classic(img, weights_file):
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


header = st.beta_container()
dataset = st.beta_container()
df = None


menu = st.sidebar.selectbox('Menu:', options=["Welcome","Dataset", "Prediction", "Examples", 'Machine Prediction'])

with header:
    st.title('Do I have a tumor?')

if menu == 'Welcome':
    st.title('Welcome!')
    st.write('This prediction proyect wants to use RMI images of brain tumors to be able to indentify the presence of a tumor and classify its kind between pituitary, meningioma and glioma')

if menu == 'Dataset':
    st.write('The transformation of the photos to array with its diagnostic')
    r = requests.get("http://localhost:6060/give_me_id?token_id=M53994161").json()
    a = path + os.sep + 'data' + os.sep + 'df_photos.csv'
    df = pd.DataFrame(r)
    st.write(df)

if menu == 'Prediction':
    st.title("Upload + Classification Example")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        resized = image.resize((150, 150))
        array = np.array(resized)
        image_array = cv2.resize(array,(100, 100))
        st.write("Trying...")
        model_path = path + os.sep + 'models' + os.sep + 'model_2.h5'
        model = keras.models.load_model(model_path)
        label = model.predict(array)
        if label == 0:
            st.write("The MRI scan has a Glioma Tumor")
        elif label == 1:
            st.write("The MRI scan shows no Tumor")
        elif label == 2:
            st.write("The MRI scan has a Meningioma Tumor")
        elif label == 3:
            st.write("The MRI scan has Pituitary")

if menu == 'Machine Prediction':
    st.title("Upload + Classification Example")
    uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        model_go = path + os.sep + 'models' + os.sep + 'keras_model.h5'
        label = classic(image, model_go)
        if label == 0:
            st.write("The MRI scan has a Glioma Tumor")
        elif label == 1:
            st.write("The MRI scan has a Meningioma Tumor")
        elif label == 2:
            st.write("The MRI scan has a Pituitary Tumor")
        elif label == 3:
            st.write("The MRI scan shows no tumor")

if menu == "Examples":
    image = Image.open(path + os.sep + 'img' + os.sep + 'no_tumor.jpg')
    st.image (image,use_column_width=False)
    st.write("No Tumor")
    image = Image.open(path + os.sep + 'img' + os.sep + 'glioma_tumor.jpg')
    st.image (image,use_column_width=False)
    st.write("Glioma Tumor")
    image = Image.open(path + os.sep + 'img' + os.sep + 'meningioma_tumor.jpg')
    st.image (image,use_column_width=False)
    st.write("Meningioma Tumor")
    image = Image.open(path + os.sep + 'img' + os.sep + 'pituitary_tumor.jpg')
    st.image (image,use_column_width=False)
    st.write("Pituitary Tumor")
    pass
