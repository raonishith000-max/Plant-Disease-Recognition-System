# Plant Disease Recognition System

This project presents a Plant Disease Recognition System developed as part of a
B.E. mini project. The system uses a CNN-based deep learning approach
(EfficientNet) to detect and classify plant diseases from leaf images.

## Project Overview
- Deep learning model trained using TensorFlow with transfer learning
- EfficientNet-based CNN architecture
- Image-based plant disease classification
- Offline inference using TensorFlow Lite
- Android application developed using Android Studio (Kotlin)

## Android Application
- Allows users to capture plant leaf images using the device camera
- Supports image selection from the gallery
- Performs disease prediction completely offline
- Displays disease name, confidence score, cause, and cure

## Dataset
The model was trained using a publicly available plant leaf disease dataset
obtained from Kaggle.

## Model Information
The trained TensorFlow Lite (.tflite) model file is not included in this repository
due to file size limitations. The model is used locally within the Android
application for offline inference.

## Note
This repository is provided for academic and evaluation purposes only.
