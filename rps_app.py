import tensorflow as tf
import streamlit as st

model = tf.keras.models.load_model(
    "/Users/addisonpratt/Desktop/Sports Classifier/my_model.hdf5"
)

st.write(
    """
   # Sports Classifier Prediction
"""
)

st.write(
    "This model predicts what type of sport is being played in the picture that is input"
)

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np


def import_and_predict(image_data, model):

    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (
        cv2.resize(img, dsize=(75, 75), interpolation=cv2.INTER_CUBIC)
    ) / 255.0

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("It is a paper!")
    elif np.argmax(prediction) == 1:
        st.write("It is a rock!")
    else:
        st.write("It is a scissor!")

    st.text("Probability (0: Paper, 1: Rock, 2: Scissor")
    st.write(prediction)
