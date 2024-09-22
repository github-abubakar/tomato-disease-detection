import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import datetime
import io
from streamlit_cropper import st_cropper
import pandas as pd  # Add this line
import base64  # Add this to encode images



# Add causes and remedies for each disease class
disease_info = {
    'Tomato___Bacterial_spot': {
        'Description': "Bacterial spot is a disease caused by **Xanthomonas campestris pv. vesicatoria**. "
                       "It primarily affects leaves, stems, and fruits. Symptoms include dark, water-soaked spots that "
                       "can merge to create larger areas of decay, leading to defoliation and reduced yield.",
        'Cause': 'This disease is caused by a bacterial infection from Xanthomonas campestris pv. vesicatoria. '
                 'The bacteria thrive in warm, moist environments, entering the plant through natural openings or wounds.',
        'Remedy': 'To manage bacterial spot, use copper-based fungicides as a preventive measure. Regularly inspect plants '
                  'and remove infected leaves immediately to prevent spread. Implement practices to reduce leaf wetness, '
                  'such as avoiding overhead watering and improving air circulation around the plants.'
    },

    'Tomato___Early_blight': {
        'Description': "Early blight, caused by the fungus **Alternaria solani**, typically appears as dark, target-like spots on the lower leaves. "
                       "These spots can expand and lead to yellowing and premature leaf drop, reducing the plant's overall vigor and yield.",
        'Cause': 'Early blight results from a fungal infection caused by Alternaria solani. Favorable conditions for this disease include warm temperatures and wet foliage, especially when combined with high humidity.',
        'Remedy': 'Apply effective fungicides like chlorothalonil or copper at the first sign of disease. Rotate crops yearly to disrupt the disease cycle and remove any infected plant debris to minimize spore survival.'
    },

    'Tomato___Late_blight': {
        'Description': "Late blight is a serious disease caused by **Phytophthora infestans**, characterized by large, dark lesions on leaves, "
                       "which can rapidly lead to plant collapse in humid conditions. Infected fruits develop brown lesions and can rot quickly.",
        'Cause': 'This disease is caused by the fungal pathogen Phytophthora infestans, which thrives in cool, moist environments. '
                 'Infection can occur through spores dispersed by wind or water, especially during wet conditions.',
        'Remedy': 'To control late blight, use preventative fungicides and practice crop rotation. Remove and destroy infected plants immediately. Avoid planting tomatoes in previously infected soil and ensure proper spacing for air circulation.'
    },

    'Tomato___Leaf_Mold': {
        'Description': "Leaf mold, caused by **Passalora fulva**, thrives in warm, humid environments. It results in yellowing of leaves, "
                       "and a dense layer of fuzzy gray mold appears on the undersides. This can significantly reduce photosynthesis and yield.",
        'Cause': 'Leaf mold is caused by the fungal pathogen Passalora fulva, which prefers humid conditions and can spread quickly in greenhouses or high-density planting.',
        'Remedy': 'To manage leaf mold, improve air circulation around the plants by proper spacing and pruning. Apply fungicides such as mancozeb or copper to infected plants and consider planting resistant varieties.'
    },

    'Tomato___Septoria_leaf_spot': {
        'Description': "Septoria leaf spot is caused by **Septoria lycopersici**, leading to small, dark spots with lighter centers on leaves. "
                       "Severe infections can result in extensive leaf drop and reduced fruit quality.",
        'Cause': 'This fungal disease is caused by Septoria lycopersici, which thrives in wet conditions and can spread through water splashes or contaminated tools.',
        'Remedy': 'Remove and destroy infected leaves promptly. Apply fungicides as needed and avoid overhead watering to reduce leaf wetness, which encourages spore germination.'
    },

    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'Description': "Two-spotted spider mites are tiny pests that feed on the sap of tomato plants, causing stippling, yellowing, and leaf drop. "
                       "They thrive in hot, dry conditions and can reproduce rapidly, leading to significant damage. Management strategies include "
                       "insecticidal soap, increasing humidity, and introducing predatory mites.",
        'Cause': 'Infestation is caused by two-spotted spider mites, which are often found in dry, dusty environments where they reproduce quickly and feed on plant sap, weakening the plants.',
        'Remedy': 'To control spider mites, use insecticidal soap or miticides as soon as signs of infestation appear. Increasing humidity around the plants can help deter these pests. Introducing predatory mites can also naturally control their population.'
    },

    'Tomato___Target_Spot': {
        'Description': "Target spot is caused by **Corynespora cassiicola**. It presents as dark spots with concentric rings on leaves, "
                       "and can lead to leaf drop and reduced yield.",
        'Cause': 'This disease is caused by Corynespora cassiicola, which thrives in warm, humid conditions and spreads through rain or overhead irrigation.',
        'Remedy': 'To manage target spot, apply fungicides like azoxystrobin early in the disease cycle. Practice crop rotation and remove any infected plant debris to minimize spore survival in the soil.'
    },

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'Description': "This viral disease is transmitted by whiteflies and causes yellowing, curling, and stunted growth of leaves. "
                       "Infected plants may produce few or no fruits.",
        'Cause': 'Tomato yellow leaf curl virus is transmitted primarily by whiteflies, which carry the virus from infected plants to healthy ones. High temperatures and drought stress can exacerbate symptoms.',
        'Remedy': 'Control whitefly populations using insecticides or natural predators. Remove infected plants immediately to prevent spread and consider planting resistant varieties to mitigate risk.'
    },

    'Tomato___Tomato_mosaic_virus': {
        'Description': "Tomato mosaic virus causes mottling, yellowing, and stunted growth in infected plants. It spreads through infected seeds, tools, "
                       "and even human contact.",
        'Cause': 'This viral infection is caused by the tomato mosaic virus, which can be transmitted through contaminated seeds or by using infected tools. Poor sanitation practices can lead to widespread outbreaks.',
        'Remedy': 'To manage this virus, remove and destroy infected plants to prevent further spread. Sanitize tools between uses to minimize transmission and select disease-resistant varieties when planting.'
    },

    'Tomato___healthy': {
        'Description': "Healthy tomato plants exhibit vibrant green leaves and robust growth. Proper cultivation practices, including appropriate watering, fertilization, "
                       "and pest management, are essential to maintain plant health.",
        'Cause': 'No disease detected, indicating that the plant is in optimal health with no visible signs of infection or distress.',
        'Remedy': 'Continue with good cultivation practices, including regular watering, balanced fertilization, and pest management to ensure the plants remain healthy. Monitor for signs of disease regularly.'
    }
}




# Assuming all images are in the same directory as your script
disease_images = {
    'Tomato___Bacterial_spot': './/bacterial_spot.jpg',
    'Tomato___Early_blight': 'early_blight.jpg',
    'Tomato___Late_blight': 'late_blight.jpg',
    'Tomato___Leaf_Mold': 'leaf_mold.jpg',
    'Tomato___Septoria_leaf_spot': 'septoria_leaf_spot.jpg',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'spider_mites.jpg',
    'Tomato___Target_Spot': 'target_spot.jpg',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'yellow_leaf_curl_virus.jpg',
    'Tomato___Tomato_mosaic_virus': 'mosaic_virus.jpg',
    'Tomato___healthy': 'healthy.jpg'
}

# Database setup
conn = sqlite3.connect('disease_predictions.db')
c = conn.cursor()

# Create tables if they do not exist
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        disease TEXT,
        confidence REAL,
        timestamp TEXT,
        input_image BLOB
    )
''')

conn.commit()

# Function to save prediction into the database
def save_prediction(user_id, disease, confidence, image_bytes):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO predictions (user_id, disease, confidence, timestamp, input_image) VALUES (?, ?, ?, ?, ?)",
              (user_id, disease, confidence, timestamp, image_bytes))
    conn.commit()

# Function to load the TensorFlow model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('9.keras')
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

# User Authentication
def authenticate_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone()

def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# Function to clear prediction history for the logged-in user
def clear_prediction_history(user_id):
    c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
    conn.commit()

# Function to display image in table
def image_to_base64(image_bytes):
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    return f'<img src="data:image/png;base64,{encoded}" width="50" height="50">'




# Main logic
def main():
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None

    st.sidebar.title("Dashboard")
    if st.session_state['user_id'] is None:
        app_mode = st.sidebar.selectbox("Select Page", ["Login", "Register"])
    else:
        app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Prediction History", "Logout"])

    if app_mode == "Login":
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')

        if st.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state['user_id'] = user[0]  # Store user ID in session state
                st.success("Login successful!")
                st.rerun()  # Refresh the page to show the home page
            else:
                st.error("Invalid username or password.")

    elif app_mode == "Register":
        st.title("Register")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')

        if st.button("Register"):
            if register_user(username, password):
                st.success("User registered successfully! You can now log in.")
            else:
                st.error("Username already exists.")

    elif app_mode == "Logout":
        st.session_state['user_id'] = None
        st.success("Logged out successfully.")
        st.rerun()  # Refresh the page after logout

    # Only show the following pages if user is logged in
    if st.session_state['user_id'] is not None:
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
            st.title("About Tomato Diseases")
            st.markdown("""
                This dataset consists of about 36K RGB images of healthy and diseased crop leaves.
            """)

            # Dropdown list for diseases
            selected_disease = st.selectbox("Select a disease to learn more:", list(disease_info.keys()))

            # Display information based on selection
            if selected_disease:
                description = disease_info[selected_disease]['Description']
                cause = disease_info[selected_disease]['Cause']
                remedy = disease_info[selected_disease]['Remedy']

                st.markdown(f"### {selected_disease}")
                st.image(disease_images[selected_disease], width=350)  # Display the disease image
                st.markdown(f"**Description:** {description}")
                st.markdown(f"**Cause:** {cause}")
                st.markdown(f"**Remedy:** {remedy}")


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
                            model = load_model()
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

                            # Save the prediction in the database
                            save_prediction(st.session_state['user_id'], predicted_class, confidence, image_bytes)

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
                            model = load_model()
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

                            # Save the prediction in the database
                            save_prediction(st.session_state['user_id'], predicted_class, confidence, image_bytes)
        
        # Prediction History Page
        elif app_mode == "Prediction History":
            username = c.execute("SELECT username FROM users WHERE id=?", (st.session_state['user_id'],)).fetchone()[0]
            st.title(f"Prediction History of - {username}")

            # Fetch prediction history for the logged-in user
            def fetch_prediction_history():
                c.execute("SELECT input_image, disease, confidence, timestamp FROM predictions WHERE user_id=? ORDER BY timestamp DESC", (st.session_state['user_id'],))
                return c.fetchall()

            data = fetch_prediction_history()

            if len(data) == 0:
                st.info("No prediction history found.")
            else:
                # Clear prediction history button
                if st.button("Clear Prediction History"):
                    clear_prediction_history(st.session_state['user_id'])
                    st.success("Prediction history cleared.")
                    data = fetch_prediction_history()
                    
                if data:
                    # Create a DataFrame for predictions
                    history_df = pd.DataFrame(data, columns=["Input Image", "Class Name", "Accuracy", "Date"])

                    # Convert accuracy to percentage format
                    history_df['Accuracy'] = history_df['Accuracy'].map("{:.2f}%".format)

                    # Convert input image bytes to base64 and display in the table
                    history_df['Input Image'] = history_df['Input Image'].apply(image_to_base64)

                    # Add custom CSS for table alignment
                    st.markdown(
                        """
                        <style>
                        table {
                            width: 100%;
                        }
                        th {
                            text-align: left !important;
                        }
                        td {
                            text-align: left !important;
                            vertical-align: middle !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    # Display the table with images and other data
                    st.markdown(history_df.to_html(escape=False, index=False), unsafe_allow_html=True)


# Run the app
if __name__ == '__main__':
    main()
