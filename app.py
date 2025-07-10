import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

# Constants
MODEL_PATH = "model.h5"
MODEL_ID = "1p9pqC-Ba4aKdNcQploHjnaCVip5J07qe"  # Replace with your own model ID if needed

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model..."):
            url = f"https://drive.google.com/uc?id={MODEL_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

# Load the model
model = download_and_load_model()

# Streamlit UI
st.title("🌍 Deep Learning Land Cover Classifier")
st.write("Upload a satellite image to classify it into one of the trained land cover categories.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Adjust based on your model's expected input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Display prediction
    st.write("✅ **Prediction Output:**")
    st.write(prediction)

    # Optionally decode class if using softmax
    try:
        class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']  # Replace with your own class labels
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.success(f"Predicted class: **{predicted_class}** with confidence **{confidence:.2f}%**")
    except:
        st.warning("Prediction could not be interpreted into class labels. Check model output shape.")
