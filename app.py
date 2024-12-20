import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import plotly.express as px

# Load model
@st.cache_resource
def load_trained_model():
    return load_model('Early_Mind.h5')

model = load_trained_model()

# Define a function to preprocess the image

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to model input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img

# Page Configuration
st.set_page_config(
    page_title="EarlyMind: Alzheimer's Detection",
    page_icon="🧠",
    layout="wide",
)

# App Header
st.title("🧠 EarlyMind: Alzheimer's Disease Detection")
st.subheader("AI-powered early detection system for Alzheimer's Disease")
st.markdown("Upload an MRI image and let our model predict early signs of Alzheimer's.")

# Tabs for navigation
tabs = st.tabs(["🏠 Home", "📊 Visualization", "📚 About",  "📝 Feedback"])

# Tab: Home
with tabs[0]:
    uploaded_file = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded MRI Image", use_column_width=True)

        st.info("Processing the image...")
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        img_processed = preprocess_image(img)
        predictions = model.predict(img_processed)
        predicted_class = np.argmax(predictions)
        class_names = ['Non Demented', 'Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia']
        result = class_names[predicted_class]

        st.success(f"Prediction: {result}")
        st.bar_chart(predictions.flatten(), height=200)

        # Additional advice
        if result == "Non Demented":
            st.info("No signs of dementia detected. Maintain a healthy lifestyle!")
        else:
            st.warning("Signs of dementia detected. Consult a medical professional.")

        # Download results
        st.download_button(
            "Download Prediction",
            f"The model predicts: {result}",
            file_name="prediction_result.txt"
        )

# Tab: Visualization
with tabs[1]:
    st.header("📊 Model Prediction Visualizations")

    if uploaded_file:
        # Prediction Probabilities Visualization
        probabilities = predictions.flatten()
        fig = px.bar(x=class_names, y=probabilities, title="Prediction Probabilities", labels={"x": "Class", "y": "Probability"})
        st.plotly_chart(fig)

# Tab: About
with tabs[2]:
    st.header("📚 About EarlyMind")
    st.markdown(
        """
        EarlyMind is a deep learning application designed to assist in the early detection of Alzheimer's Disease using MRI images. It leverages state-of-the-art AI technologies to provide accurate predictions and insights.
        
        - **Model Architecture**: ResNet50
        - **Data**: Preprocessed MRI images labeled into dementia categories.
        - **Purpose**: Aid healthcare professionals with quick, reliable insights.
        """
    )
    st.markdown("For inquiries, contact: kyulumumo@gmail.com")

# Tab: Feedback
with tabs[3]:
    st.header("📝 Feedback")
    st.markdown("We value your feedback! Please provide your comments and suggestions below.")
    feedback = st.text_area("Your feedback:")
    if st.button("Submit"):
        st.success("Thank you for your feedback!")

# Sidebar for additional navigation and information
st.sidebar.header("Navigation")
st.sidebar.markdown("Use the tabs above to navigate through the app.")
st.sidebar.header("Contact")
st.sidebar.markdown("For support, contact: kyulumumo@gmail.com")