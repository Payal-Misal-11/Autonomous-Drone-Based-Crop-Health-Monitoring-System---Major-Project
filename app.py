import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle

# Load trained model
model = tf.keras.models.load_model("tomato_leaf_disease_model.h5")

# Define class labels (Change if required)
class_labels = ['Healthy', 'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold']

# Streamlit App UI
st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload an image of a tomato leaf to check for diseases.")

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict using the model
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display the Image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Show prediction result
    st.write(f"### 🔍 Prediction: {predicted_class}")
