import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page configuration and background color
st.set_page_config(
    page_title="TOMATO DISEASE PREDICTOR",
    page_icon=":tomato:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for background and text color
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hide Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Disable file uploader warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# Function to load the TensorFlow model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('6.keras')
    return model

# Main logic
def main():
    # Load the model
    model = load_model()

    # Sidebar navigation
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

    # Main Page
    if app_mode == "Home":
        st.title("TOMATO PLANT DISEASE RECOGNITION")
        st.image("tomato.jpg", use_column_width=True)
        st.markdown(
            """
            Welcome to the Plant Disease Recognition System! 🌿🔍
            
            Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, 
            and our system will analyze it to detect any signs of diseases. Together, let's protect our crops 
            and ensure a healthier harvest!
            
            ### How It Works
            1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
            2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
            3. **Results:** View the results and recommendations for further action.
            
            ### Why Choose Us?
            - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
            - **User-Friendly:** Simple and intuitive interface for seamless user experience.
            - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
            
            ### Get Started
            Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
            
            """
        )

    # About Page
    elif app_mode == "About":
        st.title("About")
        st.markdown(
            """
            #### About Dataset
            This dataset is recreated using offline augmentation from the original dataset.
            The original dataset can be found on this GitHub repo. This dataset consists of about 
            87K RGB images of healthy and diseased crop leaves which is categorized into 38 different 
            classes. The total dataset is divided into 80/20 ratio of training and validation set preserving 
            the directory structure. A new directory containing 33 test images is created later for prediction purpose.
            
            #### Content
            - train set
            - test set
            - validation set
            """
        )

    # Disease Recognition Page
    elif app_mode == "Disease Recognition":
        st.title("TOMATO DISEASE RECOGNITION")
        test_image = st.file_uploader("Choose an Image:")
        show_image = st.button("Show Image")
        
        if show_image:
            if test_image is None:
                st.warning("Please upload an image file first")
            else:
                st.image(test_image, width=400)

        # Predict button
        if st.button("Predict"):
            if test_image is None:
                st.warning("Please upload an image file first")
            else:
                with st.spinner("Calculating Prediction..."):
                    predictions = import_and_predict(test_image, model)
                    class_names = [
                        'Tomato___Bacterial_spot',
                        'Tomato___Early_blight',
                        'Tomato___Late_blight',
                        'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot',
                        'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]
                    max_index = np.argmax(predictions)
                    predicted_class = class_names[max_index]
                    confidence = predictions[0][max_index] * 100
                    st.success(f"Prediction: {predicted_class} with confidence {confidence:.2f}%")

# Function to perform prediction
def import_and_predict(image_data, model):
    size = (256, 256)
    image = Image.open(image_data)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if __name__ == "__main__":
    main()
