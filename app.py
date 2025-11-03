import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="🎭 Deepfake Detection", layout="centered")

st.title("🎭 Deepfake Detection App")
st.write("Upload an image to check whether it's **real or deepfake** using a trained CNN model.")

# ✅ Download model from Hugging Face (only first time)
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="khr007adarsh/deepfake-detector",  # 🔁 Replace with your Hugging Face username/repo
        filename="deepfake_model.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ✅ File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    # Preprocess for model
    img_resized = cv2.resize(image, (128, 128))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Display result
    confidence = prediction[0][0]
    if confidence < 0.5:
        st.success(f"🟢 Prediction: Real ({(1 - confidence) * 100:.2f}% confidence)")
    else:
        st.error(f"🔴 Prediction: Fake ({confidence * 100:.2f}% confidence)")
