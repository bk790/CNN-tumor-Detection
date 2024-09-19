import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model(r'C:\Users\bk\Desktop\dl\deep-learningo\CNN\tumor_detection\tumordata\model.h5')

# Function for making predictions
def make_prediction(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize the image
    res = model.predict(img)
    return "Tumor Detected" if res > 0.5 else "No Tumor"

# Streamlit app
st.title("Tumor Detection App")
st.write("Upload an image to check for tumor detection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    result = make_prediction(img)
    st.success(result)
