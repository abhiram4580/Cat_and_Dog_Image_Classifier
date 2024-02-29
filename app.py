import streamlit as st
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image

# Load your InceptionResNetV2-based model
model = load_model('model.h5')

# Set the image size based on your model's input shape
img_size = (299, 299)

def preprocess_image(image):
    img = image.resize(img_size)
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict(image):
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    return predictions

def main():
    # Dark-themed layout
    dark_css = """
        <style>
            body {
                color: white;
                background-color: #1e1e1e;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
            }
        </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)

    st.title("Cat or Dog Image Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")

        predictions = predict(image)

        # Assuming 0 corresponds to cat and 1 corresponds to dog
        cat_prob = predictions[0][0]
        dog_prob = predictions[0][1]

        st.write(f"Probability of being a Cat: {cat_prob:.2f}")
        st.write(f"Probability of being a Dog: {dog_prob:.2f}")

        if dog_prob > cat_prob:
            st.markdown("## It's a Dog! 🐶")
        else:
            st.markdown("## It's a Cat! 😺")

if __name__ == "__main__":
    main()
