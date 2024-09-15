import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import datetime
from streamlit_cropper import st_cropper
import io

# Add causes and remedies for each disease class
disease_info = {
    'Tomato___Bacterial_spot': {
        'Cause': 'Bacterial infection caused by Xanthomonas campestris pv. vesicatoria.',
        'Remedy': 'Use copper-based fungicides and remove infected leaves. Avoid overhead watering.'
    },
    'Tomato___Early_blight': {
        'Cause': 'Fungal infection caused by Alternaria solani.',
        'Remedy': 'Apply fungicides like chlorothalonil or copper, rotate crops, and remove infected leaves.'
    },
    'Tomato___Late_blight': {
        'Cause': 'Fungal infection caused by Phytophthora infestans.',
        'Remedy': 'Use fungicides, destroy infected plants, and avoid planting in infected soil.'
    },
    'Tomato___Leaf_Mold': {
        'Cause': 'Fungal infection caused by Passalora fulva.',
        'Remedy': 'Improve air circulation and apply fungicides like mancozeb or copper.'
    },
    'Tomato___Septoria_leaf_spot': {
        'Cause': 'Fungal infection caused by Septoria lycopersici.',
        'Remedy': 'Remove infected leaves, apply fungicides, and avoid overhead watering.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'Cause': 'Infestation of two-spotted spider mites.',
        'Remedy': 'Use insecticidal soap or miticides and increase humidity around plants.'
    },
    'Tomato___Target_Spot': {
        'Cause': 'Fungal infection caused by Corynespora cassiicola.',
        'Remedy': 'Use fungicides like azoxystrobin, and practice crop rotation.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'Cause': 'Viral infection transmitted by whiteflies.',
        'Remedy': 'Control whitefly populations with insecticides and remove infected plants.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'Cause': 'Viral infection caused by Tomato mosaic virus.',
        'Remedy': 'Remove infected plants and sanitize tools to prevent the spread.'
    },
    'Tomato___healthy': {
        'Cause': 'No disease detected.',
        'Remedy': 'Continue with good cultivation practices and monitor for signs of disease.'
    }
}

# Set up SQLite database connection
conn = sqlite3.connect('disease_predictions.db')
c = conn.cursor()

# Create a table to store predictions
c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        disease TEXT,
        confidence REAL,
        timestamp TEXT
    )
''')
conn.commit()

# Function to save prediction into the database
def save_prediction(disease, confidence):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO predictions (disease, confidence, timestamp) VALUES (?, ?, ?)",
              (disease, confidence, timestamp))
    conn.commit()

# Function to load the TensorFlow model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('6.keras')
    return model

# Function to perform prediction
def import_and_predict(image_bytes, model):
    image = Image.open(io.BytesIO(image_bytes))
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Main logic
def main():
    # Load the model
    model = load_model()

    # Sidebar navigation
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Prediction History"])

    # Home Page
    if app_mode == "Home":
        st.title("TOMATO PLANT DISEASE RECOGNITION")
        st.image("tomato.jpg", use_column_width=True)
        st.markdown("""
            Welcome to the Plant Disease Recognition System! 🌿🔍
            Our mission is to help identify plant diseases efficiently. Upload an image of a plant, 
            and our system will analyze it to detect any signs of diseases.
        """)

    # About Page
    elif app_mode == "About":
        st.title("About")
        st.markdown("""
            #### About Dataset
            This dataset consists of about 87K RGB images of healthy and diseased crop leaves.
        """)

    # Disease Recognition Page
    elif app_mode == "Disease Recognition":
        st.title("TOMATO DISEASE RECOGNITION")

        # Provide options to either take a picture or upload an image
        option = st.radio("Choose Input Method", ("Upload Image", "Take Picture"))

        if option == "Upload Image":
            test_image = st.file_uploader("Choose an Image:")
            if test_image is not None:
                image = Image.open(test_image)

                # Display uploaded image (no cropping here)
                st.subheader("Uploaded Image Preview")
                st.image(image, width=350)

                # Convert the uploaded image to bytes for prediction
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='PNG')
                image_bytes = image_bytes.getvalue()

                if st.button("Predict"):
                    with st.spinner("Calculating Prediction..."):
                        predictions = import_and_predict(image_bytes, model)
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

                        # Show cause and remedy for the predicted disease
                        st.markdown(f"**Cause:** {disease_info[predicted_class]['Cause']}")
                        st.markdown(f"**Remedy:** {disease_info[predicted_class]['Remedy']}")

                        # Save the prediction to the database
                        save_prediction(predicted_class, confidence)

        elif option == "Take Picture":
            camera_image = st.camera_input("Take a picture with your camera")
            if camera_image is not None:
                image = Image.open(camera_image)

                # Ask user if they want to crop the image (only for Capture mode)
                crop_option = st.checkbox("Do you want to crop the image?")

                if crop_option:
                    # Add cropping functionality
                    st.subheader("Crop the Image")
                    cropped_image = st_cropper(image, box_color='#000000', aspect_ratio=(4, 3))

                    # Display cropped image
                    st.subheader("Cropped Image Preview")
                    st.image(cropped_image, use_column_width=True)

                    # Convert the cropped image to bytes
                    cropped_image_bytes = io.BytesIO()
                    cropped_image.save(cropped_image_bytes, format='PNG')
                    cropped_image_bytes = cropped_image_bytes.getvalue()

                    image_bytes = cropped_image_bytes
                else:
                    # Use the full image if cropping is not selected
                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format='PNG')
                    image_bytes = image_bytes.getvalue()

                if st.button("Predict"):
                    with st.spinner("Calculating Prediction..."):
                        predictions = import_and_predict(image_bytes, model)
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

                        # Show cause and remedy for the predicted disease
                        st.markdown(f"**Cause:** {disease_info[predicted_class]['Cause']}")
                        st.markdown(f"**Remedy:** {disease_info[predicted_class]['Remedy']}")

                        # Save the prediction to the database
                        save_prediction(predicted_class, confidence)

    # Prediction History Page
    elif app_mode == "Prediction History":
        st.title("Prediction History")

        # Retrieve and display predictions from the database
        c.execute("SELECT * FROM predictions")
        rows = c.fetchall()

        if rows:
            st.table(rows)
        else:
            st.write("No prediction history found.")

if __name__ == '__main__':
    main()
