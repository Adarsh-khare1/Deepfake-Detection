import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
os.system("ldd --version")
os.system("apt-get update && apt-get install -y libgl1-mesa-glx")


# Load the trained model
model = tf.keras.models.load_model('deepfake_model.h5')

st.title("🎭 Deepfake Detection App")
st.write("Upload a video or image to check if it's real or fake.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    # Preprocess for model
    img_resized = cv2.resize(image, (128, 128))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # ✅ Predict (moved inside the if block)
    prediction = model.predict(img_array)

    # ✅ Display result
    if prediction[0][0] < 0.5:
        st.subheader(f"Prediction: 🟢 Real ({(1 - prediction[0][0]) * 100:.2f}% confidence)")
    else:
        st.subheader(f"Prediction: 🔴 Fake ({prediction[0][0] * 100:.2f}% confidence)")
