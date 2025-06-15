import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from utils.preprocessing import preprocess_canvas_image
import matplotlib.pyplot as plt

st.title("Handwritten Digit Recognizer")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/digit_model.h5")

model = load_model()

st.subheader("Draw a digit")
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data, caption="Your Drawing", width=150)
    if st.button("Predict Drawn Digit"):
        input_data = preprocess_canvas_image(canvas_result.image_data)
        pred = model.predict(input_data)
        st.write("(Know that the model is 97.4% accurate so there may be room for error.)")
        st.success(f"Predicted Digit: {np.argmax(pred)}")


