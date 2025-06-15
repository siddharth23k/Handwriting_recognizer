import cv2
import numpy as np

def preprocess_canvas_image(image_data):
    gray = cv2.cvtColor(image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    return normalized.reshape(1, 784)

