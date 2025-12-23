import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_mri_model(path="mri_model.h5"):
    return load_model(path)

model = load_mri_model()

# ------------------------------
# Class Names
# ------------------------------
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ------------------------------
# Prediction Function
# ------------------------------
def predict_mri(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    predicted_class = class_names[idx]
    confidence = preds[idx] * 100

    return predicted_class, confidence, preds

# ------------------------------
# Streamlit App UI
# ------------------------------
st.set_page_config(page_title="Brain MRI Tumor Classifier", layout="wide")

st.title("🧠 Brain MRI Tumor Classifier")
st.write(
    """
Upload an MRI image and the model will predict the type of tumor with confidence.  
The uploaded image and the class probabilities are displayed side by side.
"""
)

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224), color_mode="rgb")

    predicted_class, confidence, preds = predict_mri(img)

    st.markdown(f"### **Prediction:** {predicted_class} ({confidence:.2f}%)")

    # Display Image and Bar Plot side by side
    col1, col2 = st.columns(2)

    # Set figure height same as image display
    fig_height = 150  # pixels, same as image display height

    with col1:
        st.image(img, caption="Uploaded MRI Image", use_container_width=True, output_format="PNG")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))  # square figure to match image height
        ax.bar(class_names, preds, color="#4F81BD")
        ax.set_ylim([0, 1])
        ax.set_ylabel("Confidence")
        ax.set_title("Class Probabilities")
        for i, p in enumerate(preds):
            ax.text(i, p + 0.01, f"{p*100:.1f}%", ha="center", fontweight='bold')
        st.pyplot(fig, use_container_width=True)


st.markdown("---")
st.markdown("Made with ❤️ by Kajal Kourani | EfficientNetB0 + Keras + Streamlit")
