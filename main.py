import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import layers
import keras
from keras.applications.densenet import DenseNet121, preprocess_input

# Load the trained models
model1 = tf.keras.models.load_model('models_S/S_with_DA.h5')
model2 = tf.keras.models.load_model('models_S/S_without_DA.h5')
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
model3 = tf.keras.models.load_model('models_T/T_without_DA.h5')
model4 = tf.keras.models.load_model('models_T/T_with_DA.h5')
model5 = tf.keras.models.load_model('models_T/T_with_DA_and_FT.h5')

models = [model1, model2, model3, model4, model5]

# Define image size and classes
IMG_SIZE = 32
CLASSES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

def predict_image(image, model, needs_conv_model):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.expand_dims(image, axis=0)
    if needs_conv_model:
        image = preprocess_input(image)
        image = base_model.predict(image)
    prediction = model.predict(image)
    predicted_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Streamlit app
st.set_page_config(page_title="Image Classification Web App", page_icon=":camera:", layout="wide")


st.title("Image Classification Web App")
st.markdown("""
## Welcome to the Image Classification Web App!

This application allows you to classify images into one of the following 10 categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

To get started, select a model for classification and upload an image.
""")

st.sidebar.title("Instructions")
st.sidebar.write("""
1. **Upload an Image**: Use the uploader to select an image file (jpg, jpeg, or png).
2. **Select a Model**: Choose a model from the dropdown menu.
3. **See the Results**: View the predicted class and the confidence score.
""")

model_choice = st.sidebar.selectbox("Select Model", ["Root Model", "Root Model with Data Augmentation", "Transfer Learning Model", "Transfer Learning Model with Data Augmentation", "Transfer Learning Model with Data Augmentation and Fine Tunning" ])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mapping the choice
    if model_choice == "Root Model":
        model_idx = 0
        IMG_SIZE = 32
        needs_conv_model = False
    elif model_choice == "Root Model with Data Augmentation":
        model_idx = 1
        IMG_SIZE = 32
        needs_conv_model = False
    elif model_choice == "Transfer Learning Model":
        model_idx = 2
        IMG_SIZE = 150
        needs_conv_model = True
    elif model_choice == "Transfer Learning Model with Data Augmentation":
        model_idx = 3
        IMG_SIZE = 150
        needs_conv_model = False
    elif model_choice == "Transfer Learning Model with Data Augmentation and Fine Tunning":
        model_idx = 3
        IMG_SIZE = 150
        needs_conv_model = False        
    else:
        model_idx = -1
        st.error("The selected model is not available.")
    if model_idx != -1:
        selected_model = models[model_idx]

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying the image using the selected model...")

        with st.spinner('Processing...'):
            try:
                label, confidence = predict_image(image, selected_model, needs_conv_model)
                st.success(f"**Prediction**: {label}")
                st.info(f"**Confidence**: {confidence*100:.2f}%")
            except Exception as e:
                st.error(f"An error occurred: {e}")

        # Add additional information
        st.markdown("### How the model works")
        st.write("""
        - **Root Model**: A basic neural network model trained on the dataset.
        - **Root Model with Data Augmentation**: The same as the root model but trained with augmented data for better generalization.
        - **Transfer Learning Model**: Uses DenseNet121 pretrained on ImageNet for feature extraction.
        - **Transfer Learning Model with Data Augmentation**: The same as the transfer learning model but trained with augmented data for better generalization.
        - **Transfer Learning Model with Data Augmentation and Fine Tunning**: The same as the transfer learning model with data augmentation but with layers unfrozen, more specific.
        """)

        # Add footer
        st.sidebar.info("Developed by Afonso Fernandes and Luís Oliveira.")
else:
    st.info("Please upload an image to get started.")
