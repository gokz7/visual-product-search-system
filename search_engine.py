import streamlit as st                           
import numpy as np                               
import tensorflow as tf                          
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input 
import faiss                                     
import pickle                                    
import os
from PIL import Image                            

# ==========================================
# SYSTEM INITIALIZATION
# ==========================================
@st.cache_resource
def load_core_engine():
    print("Loading AI Brains and Search Index...")
    
    # 1. Load the Classifier (To predict the category text)
    classifier = load_model('best_classifier_model.h5')
    
    # 2. Load the Extractor (To find the visual matches)
    extractor = load_model('resnet50_extractor.h5')
    
    # 3. Load the Database
    index = faiss.read_index('product_index.index')
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
        
    return classifier, extractor, index, filenames

classifier_model, extractor_model, faiss_index, filenames = load_core_engine()

# Alphabetical list of the categories your AI was trained on
CLASS_NAMES = ['Accessories', 'Apparel', 'Footwear', 'Free Items', 'Home', 'Personal Care', 'Sporting Goods']

# ==========================================
# AI LOGIC: PREDICTION & EXTRACTION
# ==========================================
def analyze_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # 1. Predict the Category Text
    category_predictions = classifier_model.predict(preprocessed_img)
    predicted_class_index = np.argmax(category_predictions)
    predicted_category = CLASS_NAMES[predicted_class_index]
    confidence = category_predictions[0][predicted_class_index] * 100
    
    # 2. Extract the Visual Shape Features
    features = extractor_model.predict(preprocessed_img)
    faiss.normalize_L2(features)
    
    return predicted_category, confidence, features

# ==========================================
# WEB INTERFACE (FRONTEND)
# ==========================================
st.set_page_config(page_title="Visual Product Search", page_icon="👕", layout="wide")

st.title("👕 Visual Product Search Engine")
st.write("Upload a clothing item to find the closest visual matches in our database.")

uploaded_file = st.file_uploader("Choose a clear, tightly cropped image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.divider() 
    
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        st.subheader("Your Image:")
        user_image = Image.open(uploaded_file)
        st.image(user_image, use_container_width=True)
        
    with right_col:
        with st.spinner("Analyzing visual features and predicting category..."):
            temp_path = "temp_upload.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Run the AI
            predicted_category, confidence, query_vector = analyze_image(temp_path)
            
            # --- NEW CATEGORY BADGE ---
            st.success(f"**AI Prediction:** This looks like **{predicted_category}** (Confidence: {confidence:.2f}%)")
            st.write("---")
            
            # Search the FAISS index
            D, I = faiss_index.search(query_vector, k=5) 
            
            st.subheader("Top 5 Nearest Visual Matches:")
            res_cols = st.columns(5)
            
            for i in range(5):
                match_index_id = I[0][i]
                match_filename = filenames[match_index_id]
                local_image_path = os.path.join(r"D:\Visual Product Search System\data\images", match_filename)
                
                with res_cols[i]:
                    try:
                        match_img = Image.open(local_image_path)
                        st.image(match_img, use_container_width=True)
                        st.caption(f"Match: {D[0][i] * 100:.2f}%")
                    except FileNotFoundError:
                        st.error(f"Image missing: {match_filename}")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)