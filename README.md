# Handwritten Digit Recognizer

An interactive web app that predicts handwritten digits (0–9) using a trained neural network. Users can draw digits on a canvas or upload digit images, and the model will recognize the number in real time.


## Preview
<img width="782" alt="Screenshot 2025-06-15 at 11 26 49 PM" src="https://github.com/user-attachments/assets/b362eeb4-1f4d-43ab-ba26-c029cc0d0f5d" />


## Features:

- Draw digits using mouse or touchscreen  
- Upload handwritten digit images  
- Real-time digit prediction with confidence scores  
- Uses a trained neural network (Keras)  


## Model Details:

- **Framework:** TensorFlow + Keras  
- **Architecture:**
  - Dense(100, activation='relu')
  - Dense(100, activation='relu')
  - Dense(10, activation='softmax')  
- **Trained on:** MNIST Dataset  
- **Input shape:** 784 (28×28 grayscale images flattened)  
- **Accuracy:** -99.5% on validation set and ~97.4% on test set


## Run Locally:

1. Clone the repository

git clone https://github.com/siddharth23k/Handwriting_Predictor.git
cd Handwriting_Predictor

2. Install dependencies

pip install -r requirements.txt

3.Run the app

streamlit run app.py
