import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.title("🩺 TB Detection Model")

# Google Drive model download
model_file = "TB_Model_4_CNN.keras"
gdrive_file_id = "1qu7taAllAbMcQsfZDyOylkL7Dbhp3Gx_"

if not os.path.exists(model_file):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}", model_file, quiet=False)

# Load model
model = load_model(model_file)
st.success("Model loaded successfully!")

# Image Upload
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))  # Adjust size as per your model
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    result = "Tuberculosis Detected" if prediction[0][0] > 0.5 else "Normal"
    
    st.subheader("Prediction:")
    st.write(result)


